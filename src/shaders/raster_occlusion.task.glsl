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
// Dynamic block processing - actual dimensions come from uniforms
// We'll calculate blocks per workgroup based on blockDim at runtime
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
    uint numOccupiedBlocks;   // Number of occupied blocks in this group (up to 256 for 8x8x4)
    uint8_t denseOccupancyIndex[256]; // Dense array of local block indices for 8x8x4 blocks
} taskOutput;

// Shared memory for stream compaction
shared uint s_occupiedCount;
shared uint s_occupiedOffsets[32]; // One per thread

// Check if a block contains the isosurface
bool isBlockOccupied(uvec3 blockCoord) {
    // Use texelFetch with integer coordinates like Kreskowski
    ivec3 texelCoord = ivec3(blockCoord);
    uvec2 minMax = texelFetch(minMaxHierarchy, texelCoord, 0).rg;
    
    // Check if isovalue is within [min, max] range
    // Note: minMax values are already in 0-255 range, isovalue is normalized (0-1)
    float scaledIsovalue = params.isovalue * 255.0;
    bool occupied = (float(minMax.x) <= scaledIsovalue && float(minMax.y) >= scaledIsovalue);
    
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
    
    // Each workgroup processes 8x8x4 = 256 blocks
    uint blocksPerWorkgroup = 256; // 8x8x4 blocks
    uint blocksPerThread = blocksPerWorkgroup / WORKGROUP_SIZE; // 256/32 = 8 blocks per thread
    
    // Calculate which block group this workgroup processes
    // We dispatch 2 workgroups per 8x8x8 region (each handles an 8x8x4 half)
    uint groupsPerRow = (params.blockGridDim.x + 7) / 8;
    uint groupsPerSlice = groupsPerRow * ((params.blockGridDim.y + 7) / 8);
    uint groupsPerVolume = groupsPerSlice * ((params.blockGridDim.z + 7) / 8);
    // Calculate which block group this workgroup processes (8x8x4 groups)
    // uint groupsPerRow = (params.blockGridDim.x + 7) / 8;
    // uint groupsPerSlice = groupsPerRow * ((params.blockGridDim.y + 7) / 8);
    
    // uint groupID = workgroupID;
    
    // // Unpack group coordinates
    // uvec3 groupCoord;
    // groupCoord.z = groupID / groupsPerSlice;
    // uint temp = groupID % groupsPerSlice;
    // groupCoord.y = temp / groupsPerRow;
    // groupCoord.x = temp % groupsPerRow;
    
    // // Starting block coordinates for this group (8x8x4)
    // uvec3 groupStartBlock = groupCoord * uvec3(8, 8, 4);
    
    // Each 8x8x8 region gets 2 workgroups (lower and upper half)
    uint regionID = workgroupID / 2;
    uint halfIndex = workgroupID % 2; // 0 = lower half, 1 = upper half
    
    // Unpack region coordinates
    uvec3 regionCoord;
    regionCoord.z = regionID / groupsPerSlice;
    uint temp = regionID % groupsPerSlice;
    regionCoord.y = temp / groupsPerRow;
    regionCoord.x = temp % groupsPerRow;
    
    // Starting block coordinates for this 8x8x4 half
    uvec3 groupStartBlock = regionCoord * uvec3(8, 8, 8) + uvec3(0, 0, halfIndex * 4);
    
    // Initialize shared memory
    if (threadID == 0) {
        s_occupiedCount = 0;
    }
    barrier();
    
    // Each thread checks its assigned blocks within the group
    uint localOccupiedCount = 0;
    uint8_t localOccupiedBlocks[8]; // 256 blocks / 32 threads = 8 blocks per thread
    
    for (uint i = 0; i < blocksPerThread; i++) {
        uint blockIndex = threadID * blocksPerThread + i;
        if (blockIndex < blocksPerWorkgroup) {
            // Convert local index to 3D offset within the 8x8x4 group
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
    
    // Stream compaction using subgroup operations (Kreskowski-style)
    // This avoids atomics by using ballot operations
    uvec4 ballot = subgroupBallot(localOccupiedCount > 0);
    uint numThreadsWithOccupied = subgroupBallotBitCount(ballot);
    uint threadOffset = subgroupBallotExclusiveBitCount(ballot);
    
    // Calculate total count using subgroup reduction instead of atomics
    uint localCountReduced = subgroupAdd(localOccupiedCount);
    
    // Thread 0 in subgroup stores the total
    if (gl_SubgroupInvocationID == 0) {
        s_occupiedCount = localCountReduced;
    }
    
    // Store offset for this thread (prefix sum within subgroup)
    if (localOccupiedCount > 0) {
        uint myOffset = subgroupExclusiveAdd(localOccupiedCount);
        s_occupiedOffsets[threadID] = myOffset;
    }
    
    barrier();
    
    // Write occupied blocks to dense array
    if (localOccupiedCount > 0) {
        uint writeOffset = s_occupiedOffsets[threadID];  // Use threadID not threadOffset
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
            // debugPrintfEXT("OcclusionTask WG %d: No mesh workgroups emitted (count=0)", 
            //               workgroupID);
        }
    }
}