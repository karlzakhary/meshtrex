#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_subgroup_extended_types_int8: require
#extension GL_EXT_debug_printf : enable

#define WORKGROUP_SIZE 32
#define BLOCKS_PER_WORKGROUP 256
#define BLOCKS_PER_THREAD (BLOCKS_PER_WORKGROUP / WORKGROUP_SIZE)

layout(local_size_x = WORKGROUP_SIZE) in;

// UBO bindings
layout(set = 0, binding = 0) uniform ViewUniforms {
    mat4 viewProj;
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} view;

layout(set = 0, binding = 1) uniform usampler3D minMaxHierarchy;

taskPayloadSharedEXT struct Task {
    uint baseID;
    uint numOccupiedBlocks;
    uint8_t denseOccupancyIndex[256];
} taskOutput;

// Unpack linear index to 3D coordinates
uvec3 unflattenIndex(uint index, uvec2 extents) {
    uvec3 result;
    result.z = index / extents.y; // extents.y = x * y
    index -= result.z * extents.y;
    result.y = index / extents.x;
    result.x = index % extents.x;
    return result;
}

bool isBlockOccupied(uvec3 blockCoord) {
    ivec3 texelCoord = ivec3(blockCoord);
    uvec2 minMax = texelFetch(minMaxHierarchy, texelCoord, 0).rg;
    float scaledIsovalue = view.isovalue * 255.0;
    return (float(minMax.x) <= scaledIsovalue && float(minMax.y) >= scaledIsovalue);
}

void main() {
    uint threadID = gl_LocalInvocationID.x;
    uint workgroupID = gl_WorkGroupID.x;

    // --- Simplified Workgroup Mapping (Matches Kreskowski) ---
    // Each workgroup processes an 8x8x4 group of blocks.
    uvec3 groupGridDim = (view.blockGridDim.xyz + uvec3(7, 7, 3)) / uvec3(8, 8, 4);
    uvec3 groupCoord = unflattenIndex(workgroupID, uvec2(groupGridDim.x, groupGridDim.x * groupGridDim.y));
    uvec3 groupStartBlock = groupCoord * uvec3(8, 8, 4);

    // --- 1. Identify Occupied Blocks ---
    // Each thread checks 8 blocks.
    bool blockOccupancyForThread[BLOCKS_PER_THREAD];
    uint numBlocksChecked = 0;
    for (int i = 0; i < BLOCKS_PER_THREAD; ++i) {
        uint localBlockIndex = threadID + i * WORKGROUP_SIZE;
        uvec3 localBlockOffset = unflattenIndex(localBlockIndex, uvec2(8, 64));
        uvec3 blockCoord = groupStartBlock + localBlockOffset;
        
        bool occupied = false;
        if (all(lessThan(blockCoord, view.blockGridDim.xyz))) {
            occupied = isBlockOccupied(blockCoord);
        }
        blockOccupancyForThread[i] = occupied;
    }

    // --- 2. Compute Dense Occupancy List ---
    uint8_t numOccupiedInGroup[BLOCKS_PER_THREAD];
    uint8_t offsetInCurrentGroup[BLOCKS_PER_THREAD];

    for(int i = 0; i < BLOCKS_PER_THREAD; ++i) {
        uvec4 votes = subgroupBallot(blockOccupancyForThread[i]);
        numOccupiedInGroup[i] = uint8_t(subgroupBallotBitCount(votes));
        offsetInCurrentGroup[i] = uint8_t(subgroupBallotExclusiveBitCount(votes));
    }

    uint8_t offsetPerWarp[BLOCKS_PER_THREAD];
    offsetPerWarp[0] = uint8_t(0);
    for(uint i = 0; i < BLOCKS_PER_THREAD; ++i) {
        if (i > 0) {
            offsetPerWarp[i] = uint8_t(offsetPerWarp[i - 1] + subgroupBroadcast(numOccupiedInGroup[i - 1], WORKGROUP_SIZE - 1));
        }
        if (blockOccupancyForThread[i]) {
            uint writeOffset = uint(offsetPerWarp[i]) + uint(offsetInCurrentGroup[i]);
            taskOutput.denseOccupancyIndex[writeOffset] = uint8_t(threadID + i * WORKGROUP_SIZE);
        }
    }
    
    // --- 3. Emit Mesh Tasks (Thread 0 Only) ---
    if (threadID == 0) {
        uint totalOccupied = offsetPerWarp[BLOCKS_PER_THREAD - 1] + numOccupiedInGroup[BLOCKS_PER_THREAD - 1];
        totalOccupied = subgroupBroadcast(totalOccupied, WORKGROUP_SIZE - 1);

        taskOutput.baseID = workgroupID;
        taskOutput.numOccupiedBlocks = totalOccupied;
        
        uint meshWorkgroups = (totalOccupied + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        if (meshWorkgroups > 0) {
            EmitMeshTasksEXT(meshWorkgroups, 1, 1);
        }
    }
}