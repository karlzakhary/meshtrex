#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_debug_printf : enable

// Workgroup size - 32 threads per workgroup (NVIDIA warp size)
#define WORKGROUP_SIZE 32
// Each workgroup processes 256 blocks (8x8x4) - half of an 8x8x8 region
#define BLOCKS_PER_WORKGROUP 256
// Each thread processes 8 blocks
#define BLOCKS_PER_THREAD (BLOCKS_PER_WORKGROUP / WORKGROUP_SIZE)
// We process blocks in 8x8x4 groups (half of 8x8x8)
#define GROUP_SIZE_X 8
#define GROUP_SIZE_Y 8
#define GROUP_SIZE_Z 4

layout(local_size_x = WORKGROUP_SIZE) in;

// UBO bindings
layout(set = 0, binding = 0) uniform ViewUniforms {
    mat4 viewProj;
    uvec4 volumeDim;      // Volume dimensions in voxels
    uvec4 blockDim;       // Block dimensions (8, 8, 4, 1)
    uvec4 blockGridDim;   // Number of blocks in each dimension
    float isovalue;
} view;

// Min-max hierarchy texture
layout(set = 0, binding = 1) uniform usampler3D minMaxHierarchy;

// For compatibility, params is just an alias to view
#define params view

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pushConstants;

// Output to mesh shader - matches Kreskowski's memory interface
taskPayloadSharedEXT struct Task {
    uint baseID;              // Base block ID for this group
    uint numOccupiedBlocks;   // Number of occupied blocks in this 8x8x4 group
    uint8_t denseOccupancyIndex[256]; // Dense array of local block indices
} taskOutput;

// Shared memory for stream compaction
shared uint s_occupiedCount;
shared uint s_occupiedOffsets[BLOCKS_PER_THREAD];

// Check if a block contains the isosurface
bool isBlockOccupied(uvec3 blockCoord) {
    // Use texelFetch with integer coordinates like Kreskowski
    ivec3 texelCoord = ivec3(blockCoord);
    uvec2 minMax = texelFetch(minMaxHierarchy, texelCoord, 0).rg;
    
    // Check if isovalue is within [min, max] range
    // Note: minMax values are already in 0-255 range, isovalue is normalized (0-1)
    float scaledIsovalue = params.isovalue * 255.0;
    bool occupied = (float(minMax.x) <= scaledIsovalue && float(minMax.y) >= scaledIsovalue);
    
    // Debug blocks near the center (sphere is at 32,32,32 in a 64x64x64 volume)
    // Block coordinates 3,4 would contain voxels 24-31, 32-39
    // if ((blockCoord.x >= 3 && blockCoord.x <= 4) && 
    //     (blockCoord.y >= 3 && blockCoord.y <= 4) && 
    //     (blockCoord.z >= 3 && blockCoord.z <= 4)) {
    //     debugPrintfEXT("Block (%d,%d,%d): minMax=(%d,%d), iso=%.1f, occupied=%d",
    //                   blockCoord.x, blockCoord.y, blockCoord.z,
    //                   minMax.x, minMax.y, scaledIsovalue, occupied ? 1 : 0);
    // }
    
    return occupied;
}

// Unpack linear index to 3D coordinates within 8x8x4 group
uvec3 unflattenIndex(uint index, uvec2 extents) {
    uvec3 result;
    result.z = index / extents.y; // extents.y = x * y
    index -= result.z * extents.y;
    result.y = index / extents.x;
    result.x = index % extents.x;
    return result;
}

void main() {
    uint threadID = gl_LocalInvocationID.x;
    uint workgroupID = gl_WorkGroupID.x;
    
    // Debug output for first workgroup
    if (workgroupID == 0 && threadID == 0) {
        debugPrintfEXT("OcclusionTask WG 0: blockGridDim=(%d,%d,%d), isovalue=%f",
                      view.blockGridDim.x, view.blockGridDim.y, view.blockGridDim.z,
                      view.isovalue);
    }
    
    // Calculate which 8x8x4 group this workgroup processes
    // We process blocks in groups of 8x8x8, split into two 8x8x4 halves
    uint groupsPerRow = (params.blockGridDim.x + 7) / 8;
    uint groupsPerSlice = groupsPerRow * ((params.blockGridDim.y + 7) / 8);
    
    // Which 8x8x8 group and which half (0 or 1)
    uint group888ID = workgroupID / 2;
    uint halfIndex = workgroupID % 2;  // 0 = lower half (z=0-3), 1 = upper half (z=4-7)
    
    // Unpack 8x8x8 group coordinates
    uvec3 group888Coord;
    group888Coord.z = group888ID / groupsPerSlice;
    uint temp = group888ID % groupsPerSlice;
    group888Coord.y = temp / groupsPerRow;
    group888Coord.x = temp % groupsPerRow;
    
    // Starting block coordinates for this 8x8x4 half
    uvec3 groupStartBlock = group888Coord * 8;
    groupStartBlock.z += halfIndex * 4;  // Offset for upper/lower half
    
    // Initialize shared memory
    if (threadID == 0) {
        s_occupiedCount = 0;
    }
    barrier();
    
    // Each thread checks its assigned blocks within the 8x8x4 group
    uint localOccupiedCount = 0;
    uint8_t localOccupiedBlocks[BLOCKS_PER_THREAD];
    
    for (uint i = 0; i < BLOCKS_PER_THREAD; i++) {
        uint blockIndex = threadID * BLOCKS_PER_THREAD + i;
        if (blockIndex < BLOCKS_PER_WORKGROUP) {
            // Convert local index to 3D offset within 8x8x4 group
            uvec3 localBlockOffset = unflattenIndex(blockIndex, uvec2(8, 64)); // 8x8 = 64
            
            // Global block coordinates
            uvec3 blockCoord = groupStartBlock + localBlockOffset;
            
            // Check if block is within volume bounds
            if (all(lessThan(blockCoord, params.blockGridDim.xyz))) {
                if (isBlockOccupied(blockCoord)) {
                    localOccupiedBlocks[localOccupiedCount] = uint8_t(blockIndex);
                    localOccupiedCount++;
                }
            }
        }
    }
    
    // Stream compaction using subgroup operations
    uvec4 ballot = subgroupBallot(localOccupiedCount > 0);
    uint numThreadsWithOccupied = subgroupBallotBitCount(ballot);
    uint threadOffset = subgroupBallotExclusiveBitCount(ballot);
    
    // Store offset for this thread
    if (localOccupiedCount > 0) {
        s_occupiedOffsets[threadOffset] = atomicAdd(s_occupiedCount, localOccupiedCount);
    }
    
    barrier();
    
    // Write occupied blocks to dense array
    if (localOccupiedCount > 0) {
        uint writeOffset = s_occupiedOffsets[threadOffset];
        for (uint i = 0; i < localOccupiedCount; i++) {
            taskOutput.denseOccupancyIndex[writeOffset + i] = localOccupiedBlocks[i];
        }
    }
    
    barrier();
    
    // Only one thread should emit mesh tasks
    if (threadID == 0) {
        // Store the workgroup ID as base - mesh shader will reconstruct block IDs
        taskOutput.baseID = workgroupID;
        taskOutput.numOccupiedBlocks = s_occupiedCount;
        
        // Calculate mesh workgroups
        uint meshWorkgroups = (s_occupiedCount + 31) / 32;
        
        // Debug output for first few workgroups
        if (workgroupID < 3) {
            debugPrintfEXT("OcclusionTask WG %d: found %d occupied blocks, emitting %d mesh WGs", 
                          workgroupID, s_occupiedCount, meshWorkgroups);
            
            // Print first few occupied block indices
            // if (s_occupiedCount > 0) {
            //     debugPrintfEXT("  First occupied blocks: ");
            //     for (uint i = 0; i < min(5, s_occupiedCount); i++) {
            //         debugPrintfEXT("%d ", taskOutput.denseOccupancyIndex[i]);
            //     }
            //     debugPrintfEXT("\n");
            // }
        }
        
        // Emit mesh shader workgroups - each processes up to 32 blocks
        if (meshWorkgroups > 0) {
            // Debug output before emission
            if (workgroupID < 2) {
                debugPrintfEXT("OcclusionTask WG %d: About to emit %d mesh workgroups",
                              workgroupID, meshWorkgroups);
            }
            
            EmitMeshTasksEXT(meshWorkgroups, 1, 1);
            
            // Debug output after emission
            if (workgroupID < 2) {
                debugPrintfEXT("OcclusionTask WG %d: EmitMeshTasksEXT completed",
                              workgroupID);
            }
        } else {
            debugPrintfEXT("OcclusionTask WG %d: No mesh workgroups emitted (count=0)", 
                          workgroupID);
        }
    }
}