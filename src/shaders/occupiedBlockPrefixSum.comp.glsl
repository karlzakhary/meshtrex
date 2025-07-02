#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_debug_printf: require

// --- Push Constants / Uniforms ---
layout(push_constant) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} pc;

// --- Bindings ---

// Binding 0: Input Min/Max data (rg32ui -> uvec2)
layout(binding = 0) uniform usampler3D minMaxInputVolume;

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

const int LEAF_LOD   = 0;             // your leaf statistics
const int COARSE_LOD = 2;             // 4 → 16 leaf blocks per texel

uvec2 loadMinMax(ivec3 blkCoord)
{
    /* ---------- coarse test (early reject) ------------------------ */
    ivec3  texelC = blkCoord >> COARSE_LOD;        // parent texel
    uvec2  mm = texelFetch(minMaxInputVolume, texelC, COARSE_LOD).xy;

    if (pc.isovalue < float(mm.x) || pc.isovalue > float(mm.y)) {
        return uvec2(0xFFFFFFFFu, 0u);             // mark “inactive”
    }

    /* ---------- fine test (confirm) ------------------------------- */
    mm = texelFetch(minMaxInputVolume, blkCoord, LEAF_LOD).xy;
    return mm;
}

void main() {
    // --- Basic Setup & Activity Check ---
    uint globalBlockID1D = gl_GlobalInvocationID.x;
    uvec3 blockGridDim = pc.blockGridDim.xyz; // Use .xyz accessor
    uint totalBlocks = blockGridDim.x * blockGridDim.y * blockGridDim.z;

    if (globalBlockID1D >= totalBlocks) {
        return;
    }

    uvec3 blockCoord;
    uint planeSize = blockGridDim.x * blockGridDim.y;
    blockCoord.z = (planeSize > 0) ? (globalBlockID1D / planeSize) : 0;
    uint remainder = (planeSize > 0) ? (globalBlockID1D % planeSize) : globalBlockID1D;
    blockCoord.y = (blockGridDim.x > 0) ? (remainder / blockGridDim.x) : 0;
    blockCoord.x = (blockGridDim.x > 0) ? (remainder % blockGridDim.x) : remainder;

    uvec2 minMax = loadMinMax(ivec3(blockCoord));

    bool blockIsActive = (pc.isovalue >= float(minMax.x) && pc.isovalue <= float(minMax.y));
    uint activeFlag = blockIsActive ? 1 : 0;
    
    // --- Parallel Scan Implementation ---

    // Stage 1: Intra-Subgroup Exclusive Scan
    // Calculates sum of activeFlags preceding this invocation within its subgroup.
    uint subgroupPrefixSum = subgroupExclusiveAdd(activeFlag);
    // Calculate total active count within this subgroup efficiently
    // subgroupAdd(activeFlag) sums across the subgroup.
    uint subgroupTotal = subgroupAdd(activeFlag); // Total count for this subgroup

    // Stage 2: Store Subgroup Totals in Shared Memory
    // Only one thread per subgroup (e.g., the last one) needs to write.
    if (gl_SubgroupInvocationID == gl_SubgroupSize - 1) {
        // Ensure subgroupID is within bounds of shared memory array
        if(gl_SubgroupID < MAX_SUBGROUPS) {
            s_subgroupTotals[gl_SubgroupID] = subgroupTotal;
        }
    }

    // Synchronize: Ensure all subgroup totals are written to shared memory
    // before the next stage reads them. Also synchronizes execution flow.
    memoryBarrierShared(); // Ensure shared memory writes are visible
    barrier();             // Synchronize workgroup execution

    // Stage 3: Inter-Subgroup Scan (Scan across subgroup totals)
    // This is performed by only the *first* subgroup (gl_SubgroupID == 0).
    // It scans the totals stored in s_subgroupTotals.
    if (gl_SubgroupID == 0) {
        // Each thread in subgroup 0 loads a total from a different subgroup
        uint totalToScan = 0;
        if (gl_SubgroupInvocationID < MAX_SUBGROUPS) { // Ensure we only read valid totals
            totalToScan = s_subgroupTotals[gl_SubgroupInvocationID];
        }

        // Perform an exclusive scan on these totals within subgroup 0
        uint scanOfTotals = subgroupExclusiveAdd(totalToScan);

        // Write the scan result (base offset for each subgroup) back to shared memory
        if (gl_SubgroupInvocationID < MAX_SUBGROUPS) {
            s_subgroupScanResult[gl_SubgroupInvocationID] = scanOfTotals;
        }

        // All threads in block 0 calculate and store the workgroup's total active count
        uint workgroupTotal = subgroupAdd(totalToScan); // Total sum of subgroup totals
        if (gl_LocalInvocationIndex == 0) { // Equivalent to gl_SubgroupInvocationID == 0 here
            // Workgroup total = scan result for the last subgroup + total of the last subgroup
            uint lastSubgroupID = (gl_WorkGroupSize.x / gl_SubgroupSize) - 1;
            // Need the total count from the scan operation within subgroup 0
            s_workgroupTotalActiveCount = workgroupTotal;
        }
    }

    // Synchronize: Ensure scan results and workgroup total are written to shared memory
    // and visible to all threads before proceeding.
    memoryBarrierShared();
    barrier();

    // Stage 4: Combine Results for Local Offset
    // Each thread reads the starting offset for its subgroup
    uint blockPrefixSumForSubgroup = 0;
    if(gl_SubgroupID < MAX_SUBGROUPS) { // Bounds check
        blockPrefixSumForSubgroup = s_subgroupScanResult[gl_SubgroupID];
    }
    // Add the intra-subgroup prefix sum calculated earlier
    uint localOffset = blockPrefixSumForSubgroup + subgroupPrefixSum;

    // Stage 5 & 6: Get Global Offset Base and Broadcast
    // Only the first thread in the workgroup performs the atomic add
    if (gl_LocalInvocationIndex == 0) {
        uint wgTotal = s_workgroupTotalActiveCount;
        // Atomically add this workgroup's count to the global counter
        // and retrieve the starting offset for this workgroup.
        s_workgroupBaseOffset = atomicAdd(activeBlockCount.count, wgTotal);
    }

    // Synchronize: Ensure the base offset is written to shared memory
    // and visible to all threads in the workgroup.
    memoryBarrierShared();
    barrier();

    // Stage 7: Calculate Final Global Offset
    // All threads read the workgroup's base offset
    uint workgroupBaseOffset = s_workgroupBaseOffset;
    uint globalOffset = workgroupBaseOffset + localOffset;

    // Stage 8: Write Output
    // Only active threads write their original 1D block ID to the calculated position
    if (activeFlag == 1) {
        // Optional: Bounds check globalOffset against total size of blockIDs buffer if needed
        blockIDs[globalOffset] = globalBlockID1D;
    }
}
