#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

// --- Push Constants / Uniforms ---
layout(push_constant) uniform PushConstants {
    uvec4 volumeDim;     // Original volume dimensions
    uvec4 blockDim;      // Block dimensions (e.g., 8x8x8)
    uvec4 blockStride;   // Block stride for overlap (e.g., 7x7x7)
    uvec4 blockGridDim;  // Grid dimensions = ceil(volumeDim / blockStride)
    float isovalue;
} pc;

// --- Bindings ---

// Binding 0: Input Min/Max data (rg32ui -> uvec2)
// IMPORTANT: This min/max volume must have been generated with stride-based block positioning:
// - Block at grid position (bx,by,bz) covers voxels starting at (bx*stride.x, by*stride.y, bz*stride.z)
// - Each block covers blockDim voxels, so blocks overlap when stride < blockDim
layout(binding = 0, rg32ui) uniform readonly uimage3D minMaxInputVolume;

// Binding 1: Output buffer for compacted active block IDs
layout(binding = 1, std430) buffer CompactedBlockIDs {
    uint blockIDs[]; // Stores 1D indices of active blocks
};

// Binding 2: Atomic counter for total active blocks & output offset calculation
layout(binding = 2, std430) buffer ActiveBlockCount {
    uint count; // Stores the total number of active blocks found
} activeBlockCount;

// --- Workgroup Setup ---
// PMB paper suggests 128 threads for occupancy filtering.
layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
// Assumes minimum subgroup size is 32 for sizing. Adjust if needed.
#define MAX_SUBGROUPS (128 / 32) // Max subgroups in a workgroup of 128

// Stores the total active count for each subgroup
shared uint s_subgroupTotals[MAX_SUBGROUPS];
// Stores the exclusive scan result across subgroup totals (base offset for each subgroup)
shared uint s_subgroupScanResult[MAX_SUBGROUPS];
// Stores the base offset for the entire workgroup in the global output buffer
shared uint s_workgroupBaseOffset;
// Stores the total active count for the entire workgroup
shared uint s_workgroupTotalActiveCount;

// Helper function to convert 1D block index to 3D grid coordinates
uvec3 blockIndexTo3D(uint index, uvec3 gridDim) {
    uvec3 coord;
    uint planeSize = gridDim.x * gridDim.y;
    coord.z = (planeSize > 0) ? (index / planeSize) : 0;
    uint remainder = (planeSize > 0) ? (index % planeSize) : index;
    coord.y = (gridDim.x > 0) ? (remainder / gridDim.x) : 0;
    coord.x = (gridDim.x > 0) ? (remainder % gridDim.x) : remainder;
    return coord;
}

// Helper function to get the actual volume position of a block
// This is where stride comes into play - blocks are positioned at stride intervals
uvec3 getBlockVolumePosition(uvec3 blockCoord) {
    return blockCoord * pc.blockStride.xyz;
}

void main() {
    // --- Basic Setup & Activity Check ---
    uint globalBlockID1D = gl_GlobalInvocationID.x;
    uvec3 blockGridDim = pc.blockGridDim.xyz;
    uint totalBlocks = blockGridDim.x * blockGridDim.y * blockGridDim.z;

    // Early exit for threads beyond the grid
    if (globalBlockID1D >= totalBlocks) {
        return;
    }

    // Convert 1D block index to 3D grid coordinates
    uvec3 blockCoord = blockIndexTo3D(globalBlockID1D, blockGridDim);
    
    // Debug info (commented out for production):
    // uvec3 blockStartPos = getBlockVolumePosition(blockCoord);
    // uvec3 blockEndPos = blockStartPos + pc.blockDim.xyz;

    // Load min/max values for this block
    // The min/max volume should have been generated considering the stride-based positioning
    uvec2 minMax = imageLoad(minMaxInputVolume, ivec3(blockCoord)).xy;

    // Check if block is active (contains the isosurface)
    bool blockIsActive = false;
    if (minMax.x != minMax.y) { // Skip if min == max (constant block)
        blockIsActive = (pc.isovalue >= float(minMax.x) && pc.isovalue <= float(minMax.y));
    }
    uint activeFlag = blockIsActive ? 1 : 0;

    // --- Parallel Scan Implementation ---
    // This part remains the same as it's already optimized for parallel prefix sum

    // Stage 1: Intra-Subgroup Exclusive Scan
    uint subgroupPrefixSum = subgroupExclusiveAdd(activeFlag);
    uint subgroupTotal = subgroupAdd(activeFlag);

    // Stage 2: Store Subgroup Totals in Shared Memory
    if (gl_SubgroupInvocationID == gl_SubgroupSize - 1) {
        if(gl_SubgroupID < MAX_SUBGROUPS) {
            s_subgroupTotals[gl_SubgroupID] = subgroupTotal;
        }
    }

    memoryBarrierShared();
    barrier();

    // Stage 3: Inter-Subgroup Scan
    if (gl_SubgroupID == 0) {
        uint totalToScan = 0;
        if (gl_SubgroupInvocationID < MAX_SUBGROUPS) {
            totalToScan = s_subgroupTotals[gl_SubgroupInvocationID];
        }

        uint scanOfTotals = subgroupExclusiveAdd(totalToScan);

        if (gl_SubgroupInvocationID < MAX_SUBGROUPS) {
            s_subgroupScanResult[gl_SubgroupInvocationID] = scanOfTotals;
        }

        uint workgroupTotal = subgroupAdd(totalToScan);
        if (gl_LocalInvocationIndex == 0) {
            s_workgroupTotalActiveCount = workgroupTotal;
        }
    }

    memoryBarrierShared();
    barrier();

    // Stage 4: Combine Results for Local Offset
    uint blockPrefixSumForSubgroup = 0;
    if(gl_SubgroupID < MAX_SUBGROUPS) {
        blockPrefixSumForSubgroup = s_subgroupScanResult[gl_SubgroupID];
    }
    uint localOffset = blockPrefixSumForSubgroup + subgroupPrefixSum;

    // Stage 5 & 6: Get Global Offset Base and Broadcast
    if (gl_LocalInvocationIndex == 0) {
        uint wgTotal = s_workgroupTotalActiveCount;
        s_workgroupBaseOffset = atomicAdd(activeBlockCount.count, wgTotal);
    }

    memoryBarrierShared();
    barrier();

    // Stage 7: Calculate Final Global Offset
    uint workgroupBaseOffset = s_workgroupBaseOffset;
    uint globalOffset = workgroupBaseOffset + localOffset;

    // Stage 8: Write Output
    // Store the 1D block index for active blocks
    // The task shader will use this index to reconstruct the block position using stride
    if (activeFlag == 1) {
        blockIDs[globalOffset] = globalBlockID1D;
    }
}