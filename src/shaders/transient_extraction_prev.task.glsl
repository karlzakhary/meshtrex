#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_debug_printf : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_subgroup_extended_types_int8: require


// Task shader for transient extraction pass 1
// Processes blocks from PVS_prev (previous frame's visible blocks)
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

// Volume data image
layout(binding = 1, r8ui) uniform readonly uimage3D volumeImage;

// Min-max hierarchy for early rejection
layout(binding = 2) uniform sampler3D minMaxTexture;

// PVS_prev buffer - contains visible block IDs from previous frame
layout(binding = 14, std430) readonly buffer PrevPVSBuffer {
    uint count;
    uint blockIds[];
} pvsPrev;

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

// Number of unique vertices per cell configuration
const uint numUniqueVerticesPerCell[256] = uint[256](
    0, 3, 3, 6, 3, 6, 6, 9, 3, 6, 6, 9, 6, 9, 9, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 6, 9, 12, 12, 9, 9, 12, 12, 9, 12, 9, 9, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    9, 12, 12, 9, 12, 12, 12, 9, 12, 12, 12, 9, 12, 9, 9, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 6, 12, 9, 12, 9, 12, 6,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    9, 12, 12, 9, 12, 12, 12, 9, 12, 9, 12, 6, 12, 9, 6, 3,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    9, 12, 12, 12, 12, 12, 12, 6, 12, 9, 12, 9, 12, 9, 6, 3,
    9, 12, 12, 12, 12, 12, 12, 6, 12, 12, 12, 6, 12, 6, 6, 3,
    6, 9, 9, 6, 9, 6, 6, 3, 9, 6, 6, 3, 6, 3, 3, 0
);

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
    
    // Check if we're within the PVS_prev count
    if (workgroupIndex >= pvsPrev.count * 2) {
        return;
    }
    
    // Get the block ID from PVS_prev buffer
    uint blockListIndex = workgroupIndex / 2;
    uint halfIndex = workgroupIndex % 2;
    uint blockID = pvsPrev.blockIds[blockListIndex];
    
    // Decode block coordinates
    uint blocksPerRow = viewParams.blockGridDim.x;
    uint blocksPerSlice = viewParams.blockGridDim.x * viewParams.blockGridDim.y;
    
    uint blockZ = blockID / blocksPerSlice;
    uint remaining = blockID % blocksPerSlice;
    uint blockY = remaining / blocksPerRow;
    uint blockX = remaining % blocksPerRow;
    
    uvec3 blockCoord = uvec3(blockX, blockY, blockZ);
    uvec3 blockBasePos = blockCoord * viewParams.blockDim.xyz;
    
    // Process 8 voxels per thread (256 voxels / 32 threads)
    uint8_t numVertsToExtractForThread[8];
    uint8_t numVoxelsChecked = uint8_t(0);
    
    for (int voxelConfigToCheck = int(threadID); 
         voxelConfigToCheck < 256; 
         voxelConfigToCheck += int(gl_WorkGroupSize.x)) {
        
        // Unflatten voxel index to 3D position in 8x8x4 half-block
        ivec3 unflattenedLocal3DIndex;
        unflattenedLocal3DIndex.z = voxelConfigToCheck / 64;
        int temp = voxelConfigToCheck % 64;
        unflattenedLocal3DIndex.y = temp / 8;
        unflattenedLocal3DIndex.x = temp % 8;
        
        // Add half-block offset
        if (halfIndex == 1) {
            unflattenedLocal3DIndex.z += 4;
        }
        
        ivec3 cellSamplingIndex = ivec3(blockBasePos) + unflattenedLocal3DIndex;
        
        // Check if cell is within valid bounds (need room for 8 corners)
        int32_t numVerts = 0;
        if (all(lessThan(cellSamplingIndex + ivec3(1), ivec3(viewParams.volumeDim.xyz)))) {
            // Sample 8 corners
            float sampledValues[8];
        sampledValues[0] = sampleVolume(vec3(cellSamplingIndex));
        sampledValues[1] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 0, 0));
        sampledValues[2] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 1, 0));
        sampledValues[3] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 1, 0));
        sampledValues[4] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 0, 1));
        sampledValues[5] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 0, 1));
        sampledValues[6] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 1, 1));
        sampledValues[7] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 1, 1));
        
            // Compute cube index
            uint32_t cubeIndex = 0;
            float scaledIsovalue = viewParams.isovalue * 255.0;
            for (uint8_t cornerIdx = uint8_t(0); cornerIdx < uint8_t(8); ++cornerIdx) {
                cubeIndex += (uint32_t(sampledValues[cornerIdx] < scaledIsovalue) << cornerIdx);
            }
            
            numVerts = int(numUniqueVerticesPerCell[cubeIndex]);
        }
        
        numVertsToExtractForThread[numVoxelsChecked++] = uint8_t(numVerts);
        macroBlockSharedVertices[voxelConfigToCheck] = numVerts;
    }
    
    // Perform ballot voting to find occupied voxels
    uvec4 occupationVotesForAllVoxels[8];
    uint8_t numOccupiedVoxelsInGroup[8];
    uint8_t occupiedVoxelOffsetInCurrentGroup[8];
    
    for (int voxelGroupOccupancyCheckIdx = 0; voxelGroupOccupancyCheckIdx < 8; ++voxelGroupOccupancyCheckIdx) {
        uvec4 occupationVotes = subgroupBallot(uint8_t(0) != numVertsToExtractForThread[voxelGroupOccupancyCheckIdx]);
        numOccupiedVoxelsInGroup[voxelGroupOccupancyCheckIdx] = uint8_t(subgroupBallotBitCount(occupationVotes));
        occupiedVoxelOffsetInCurrentGroup[voxelGroupOccupancyCheckIdx] = uint8_t(subgroupBallotExclusiveBitCount(occupationVotes));
    }
    
    // Calculate offsets for occupied voxel storage
    uint8_t occupiedVoxelOffsetPerWarp[8];
    occupiedVoxelOffsetPerWarp[0] = uint8_t(0);
    
    for (uint32_t groupIdxToBroadcast = 0; groupIdxToBroadcast < 8; ++groupIdxToBroadcast) {
        if (0 != groupIdxToBroadcast) {
            occupiedVoxelOffsetPerWarp[groupIdxToBroadcast] = 
                uint8_t(occupiedVoxelOffsetPerWarp[groupIdxToBroadcast - 1] + 
                       subgroupBroadcast(numOccupiedVoxelsInGroup[groupIdxToBroadcast - 1], 31));
        }
        
        if (uint8_t(0) != numVertsToExtractForThread[groupIdxToBroadcast]) {
            uint writeOffset = uint32_t(occupiedVoxelOffsetPerWarp[groupIdxToBroadcast]) + 
                               uint32_t(occupiedVoxelOffsetInCurrentGroup[groupIdxToBroadcast]);
            OUT.denseOccupancyIndex[writeOffset] = uint8_t(gl_WorkGroupSize.x * groupIdxToBroadcast + threadID);
        }
    }
    
    uint32_t totalNumOccupiedVoxels = uint32_t(occupiedVoxelOffsetPerWarp[7] + numOccupiedVoxelsInGroup[7]);
    totalNumOccupiedVoxels = subgroupBroadcast(totalNumOccupiedVoxels, 31);
    
    // Thread 0 compiles dense voxel lists for mesh shader workgroups
    if (threadID == 0) {
        uint vertexCount = 0;
        uint cellCount = 0;
        
        uint32_t extractionGroupStartIndex = 0;
        uint32_t extractionGroupLength = 0;
        
        uint8_t numSubgroupsCreated = uint8_t(0);
        
        // Create maximally occupied warps (up to 64 vertices or 32 cells)
        for (int32_t cellIdx = 0; cellIdx < totalNumOccupiedVoxels; ++cellIdx) {
            uint32_t currOccupiedVoxelIdx = uint16_t(OUT.denseOccupancyIndex[cellIdx]);
            uint32_t currNumVertices = macroBlockSharedVertices[currOccupiedVoxelIdx];
            
            if (cellCount == 32 || (vertexCount + currNumVertices > 96)) {
                int writeBaseOffset = int(numSubgroupsCreated) * 2;
                OUT.offsetAndLength[writeBaseOffset + 0] = uint8_t(extractionGroupStartIndex);
                OUT.offsetAndLength[writeBaseOffset + 1] = uint8_t(cellCount);
                
                extractionGroupStartIndex += cellCount;
                vertexCount = 0;
                cellCount = 0;
                ++numSubgroupsCreated;
            }
            
            ++cellCount;
            vertexCount += currNumVertices;
        }
        
        // Write final group
        int writeBaseOffset = int(numSubgroupsCreated) * 2;
        OUT.offsetAndLength[writeBaseOffset + 0] = uint8_t(extractionGroupStartIndex);
        OUT.offsetAndLength[writeBaseOffset + 1] = uint8_t(cellCount);
        ++numSubgroupsCreated;
        
        OUT.blockID = blockID;
        OUT.halfIndex = halfIndex;
        
        // Debug output
        if (workgroupIndex == 0 || blockID == 292) {
            debugPrintfEXT("Task shader: block %d, half %d, occupied cells: %d, meshlets: %d", 
                          blockID, halfIndex, totalNumOccupiedVoxels, numSubgroupsCreated);
        }
        
        EmitMeshTasksEXT(numSubgroupsCreated, 1, 1);
    }
}