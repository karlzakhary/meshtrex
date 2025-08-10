#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_subgroup_extended_types_int8: require

// Task shader for transient extraction pass 2
// Processes blocks from PVS_curr-prev (newly visible blocks)
// Implements Kreskowski-style compact cell list generation

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// Input bindings
layout(binding = 0, std140) uniform ViewParameters {
    mat4 viewProj;
    uvec4 volumeDim;      // Volume dimensions (x, y, z, _)
    uvec4 blockDim;       // Block dimensions (x, y, z, _)
    uvec4 blockGridDim;   // Grid dimensions in blocks (x, y, z, _)
    float isovalue;
} viewParams;

// Define this to bypass PVS and process all blocks
#define BYPASS_PVS 0

// Volume data image
layout(binding = 1, r8ui) uniform readonly uimage3D volumeImage;

// Marching cubes lookup tables as storage buffers (shared with mesh shader)
layout(binding = 2, std430) readonly buffer NumVerticesTable { uint8_t numVerticesTable[]; };

// Min-max hierarchy for early rejection
layout(binding = 5) uniform sampler3D minMaxTexture;

// PVS_curr-prev buffer - contains newly visible block IDs
layout(binding = 15, std430) readonly buffer DifferencePVSBuffer {
    uint count;
    uint blockIds[];
} pvsDifference;

// Output payload to mesh shader (Kreskowski-style compact list)
struct TaskPayload {
    uint blockID;                    // Block being processed  
    uint halfIndex;                  // Which half of the block (0 or 1)
    uint8_t denseOccupancyIndex[256]; // Dense list of occupied cell indices (8-bit)
    uint8_t offsetAndLength[64];     // Interleaved offset/length pairs for mesh workgroups
};

taskPayloadSharedEXT TaskPayload OUT;

// Shared memory for stream compaction and vertex counting
shared int macroBlockSharedVertices[256];

// Helper function to get number of unique vertices for a cube configuration
uint getNumUniqueVertices(uint cubeIndex) {
    return uint(numVerticesTable[cubeIndex]);
}

// Sample volume data at a given position
float sampleVolume(vec3 pos) {
    ivec3 coord = ivec3(pos);
    if (any(lessThan(coord, ivec3(0))) || any(greaterThanEqual(coord, ivec3(viewParams.volumeDim.xyz)))) {
        return 0.0;
    }
    return float(imageLoad(volumeImage, coord).r);
}

void main() {
    uint workgroupIndex = gl_WorkGroupID.x;
    uint threadID = gl_LocalInvocationID.x;
    
    // --- 1. Determine which block this workgroup is processing ---
    uint blockID;
    uint halfIndex;
    uint maxBlockID = viewParams.blockGridDim.x * viewParams.blockGridDim.y * viewParams.blockGridDim.z;

#if BYPASS_PVS
    if (workgroupIndex >= maxBlockID * 2) { return; }
    blockID = workgroupIndex / 2;
    halfIndex = workgroupIndex % 2;
#else
    if (workgroupIndex >= pvsDifference.count * 2) { return; }
    uint blockListIndex = workgroupIndex / 2;
    halfIndex = workgroupIndex % 2;
    blockID = pvsDifference.blockIds[blockListIndex];
#endif

    if (blockID >= maxBlockID) { return; } // Skip invalid block
    
    // Decode block coordinates
    uint blocksPerRow = viewParams.blockGridDim.x;
    uint blocksPerSlice = blocksPerRow * viewParams.blockGridDim.y;
    uvec3 blockCoord = uvec3(blockID % blocksPerRow, (blockID / blocksPerRow) % viewParams.blockGridDim.y, blockID / blocksPerSlice);
    uvec3 blockBasePos = blockCoord * viewParams.blockDim.xyz;
    
    // --- 2. Partial Marching Cubes Analysis ---
    uint8_t numVertsToExtractForThread[8];
    uint8_t numVoxelsChecked = uint8_t(0);
    
    for (int cell_idx = int(threadID); cell_idx < 256; cell_idx += 32) {
        ivec3 localOffset;
        localOffset.z = cell_idx / 64;
        int rem = cell_idx % 64;
        localOffset.y = rem / 8;
        localOffset.x = rem % 8;
        
        if (halfIndex == 1) { localOffset.z += 4; }
        
        ivec3 cellSamplingIndex = ivec3(blockBasePos) + localOffset;
        
        // Sample 8 corners
        float sampledValues[8];
        sampledValues[0] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 0, 0));
        sampledValues[1] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 0, 0));
        sampledValues[2] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 1, 0));
        sampledValues[3] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 1, 0));
        sampledValues[4] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 0, 1));
        sampledValues[5] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 0, 1));
        sampledValues[6] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 1, 1));
        sampledValues[7] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 1, 1));
        
        uint32_t cubeIndex = 0;
        float scaledIsovalue = viewParams.isovalue * 255.0;
        for (uint8_t i = uint8_t(0); i < 8; ++i) {
            if (sampledValues[i] < scaledIsovalue) {
                cubeIndex |= (1u << i);
            }
        }
        
        uint8_t numVerts = uint8_t(getNumUniqueVertices(cubeIndex));
        numVertsToExtractForThread[numVoxelsChecked++] = numVerts;
        macroBlockSharedVertices[cell_idx] = numVerts;
    }
    
    // --- 3. Robust Stream Compaction (Kreskowski's method) ---
    uint8_t numOccupiedInGroup[8], offsetInCurrentGroup[8], offsetPerWarp[8];
    for (int i = 0; i < 8; ++i) {
        uvec4 votes = subgroupBallot(numVertsToExtractForThread[i] > 0);
        numOccupiedInGroup[i] = uint8_t(subgroupBallotBitCount(votes));
        offsetInCurrentGroup[i] = uint8_t(subgroupBallotExclusiveBitCount(votes));
    }
    offsetPerWarp[0] = uint8_t(0);
    for (int i = 0; i < 8; ++i) {
        if (i > 0) {
            offsetPerWarp[i] = uint8_t(offsetPerWarp[i-1] + subgroupBroadcast(numOccupiedInGroup[i-1], 31));
        }
        if (numVertsToExtractForThread[i] > 0) {
            uint writeOffset = uint(offsetPerWarp[i]) + uint(offsetInCurrentGroup[i]);
            OUT.denseOccupancyIndex[writeOffset] = uint8_t(threadID + i * 32);
        }
    }
    
    // --- 4. Batch Creation and Emission (Thread 0 only) ---
    if (threadID == 0) {
        uint32_t totalNumOccupiedVoxels = uint32_t(offsetPerWarp[7] + subgroupBroadcast(numOccupiedInGroup[7], 31));
        
        uint vertexCount = 0;
        uint cellCount = 0;
        uint32_t extractionGroupStartIndex = 0;
        uint8_t numSubgroupsCreated = uint8_t(0);
        
        for (int32_t i = 0; i < totalNumOccupiedVoxels; ++i) {
            uint32_t occupiedIndex = uint32_t(OUT.denseOccupancyIndex[i]);
            uint32_t numVerts = macroBlockSharedVertices[occupiedIndex];
            
            if (cellCount == 32 || (vertexCount + numVerts > 64)) { // Use 96 to match original
                int writeOffset = int(numSubgroupsCreated) * 2;
                OUT.offsetAndLength[writeOffset + 0] = uint8_t(extractionGroupStartIndex);
                OUT.offsetAndLength[writeOffset + 1] = uint8_t(cellCount);
                
                extractionGroupStartIndex += cellCount;
                vertexCount = 0;
                cellCount = 0;
                numSubgroupsCreated++;
            }
            cellCount++;
            vertexCount += numVerts;
        }
        
        // Unconditionally write the final batch
        int writeOffset = int(numSubgroupsCreated) * 2;
        OUT.offsetAndLength[writeOffset + 0] = uint8_t(extractionGroupStartIndex);
        OUT.offsetAndLength[writeOffset + 1] = uint8_t(cellCount);
        if (cellCount > 0) {
            numSubgroupsCreated++;
        }
        
        OUT.blockID = blockID;
        OUT.halfIndex = halfIndex;
        
        if (numSubgroupsCreated > 0) {
            EmitMeshTasksEXT(numSubgroupsCreated, 1, 1);
        }
    }
}