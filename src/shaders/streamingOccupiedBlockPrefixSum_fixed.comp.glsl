#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_debug_printf: require

// --- Push Constants ---
layout(push_constant) uniform PushConstants {
    uvec3 pageCoord;      // Page coordinate for this pass
    uint mipLevel;        // Mip level
    float isoValue;       // Isovalue for filtering
    uint blockSize;       // Block size (typically 4)
    uint pageSizeX;       // Page size X (64)
    uint pageSizeY;       // Page size Y (32)
    uint pageSizeZ;       // Page size Z (32)
    uint volumeSizeX;     // Full volume dimension X
    uint volumeSizeY;     // Full volume dimension Y
    uint volumeSizeZ;     // Full volume dimension Z
} pc;

// --- Descriptor Set Bindings ---
layout(set = 0, binding = 0, std430) readonly buffer PageTable {
    uvec2 pageEntries[];
} pageTable;

// Input Min/Max data - THIS IS PER-PAGE DATA, NOT GLOBAL!
layout(set = 0, binding = 1) uniform usampler3D minMaxInputVolume;

// Output buffer for compacted active block IDs
layout(set = 0, binding = 2, std430) buffer CompactedBlockIDs {
    uint blockIDs[]; // Stores 1D indices of active blocks
};

// Atomic counter for total active blocks & output offset calculation
layout(set = 0, binding = 3, std430) buffer ActiveBlockCount {
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

// --- Helper Functions ---

uint getPageIndex(uvec3 pageCoord, uint mipLevel) {
    // Calculate based on actual volume dimensions from push constants
    uint pagesX = (pc.volumeSizeX + pc.pageSizeX - 1) / pc.pageSizeX;
    uint pagesY = (pc.volumeSizeY + pc.pageSizeY - 1) / pc.pageSizeY;
    return pageCoord.z * pagesX * pagesY + pageCoord.y * pagesX + pageCoord.x;
}

bool isPageResident(uvec3 pageCoord, uint mipLevel) {
    uint pageIndex = getPageIndex(pageCoord, mipLevel);
    if (pageIndex >= pageTable.pageEntries.length()) return false;
    return pageTable.pageEntries[pageIndex].y != 0;
}

uvec2 loadMinMaxStreaming(ivec3 blkCoord)
{
    // IMPORTANT: For streaming, the min-max texture is PER-PAGE, not global!
    // blkCoord is in page-local block coordinates (0 to blocksPerPage-1)
    // The minMaxInputVolume texture dimensions match the page's block grid
    
    if (!isPageResident(pc.pageCoord, pc.mipLevel)) {
        // Page not resident - mark as inactive
        return uvec2(0xFFFFFFFFu, 0u);
    }
    
    // Get the per-page min-max texture dimensions
    ivec3 pageBlockGridSize = textureSize(minMaxInputVolume, 0);
    
    /* ---------- coarse test (early reject) ------------------------ */
    // Check if we have a coarse level available
    if (COARSE_LOD < textureQueryLevels(minMaxInputVolume)) {
        ivec3 texelC = blkCoord >> COARSE_LOD;        // parent texel in coarse level
        ivec3 coarseSize = pageBlockGridSize >> COARSE_LOD;
        
        if (all(greaterThanEqual(texelC, ivec3(0))) && 
            all(lessThan(texelC, coarseSize))) {
            // Sample the coarse level with normalized coordinates
            uvec2 mm = textureLod(minMaxInputVolume, (vec3(texelC) + 0.5) / vec3(coarseSize), COARSE_LOD).xy;
            
            if (pc.isoValue < float(mm.x) || pc.isoValue > float(mm.y)) {
                return uvec2(0xFFFFFFFFu, 0u);             // mark "inactive"
            }
        }
    }

    /* ---------- fine test (confirm) ------------------------------- */
    // Check bounds against the per-page texture dimensions
    if (all(greaterThanEqual(blkCoord, ivec3(0))) && 
        all(lessThan(blkCoord, pageBlockGridSize))) {
        // Sample the leaf level with normalized coordinates
        uvec2 mm = textureLod(minMaxInputVolume, (vec3(blkCoord) + 0.5) / vec3(pageBlockGridSize), LEAF_LOD).xy;
        return mm;
    }
    
    return uvec2(0xFFFFFFFFu, 0u);
}

void main() {
    // --- Basic Setup & Activity Check ---
    uint localBlockID1D = gl_GlobalInvocationID.x;
    
    // Calculate blocks within this page
    uint blocksPerPageX = pc.pageSizeX / pc.blockSize;
    uint blocksPerPageY = pc.pageSizeY / pc.blockSize;
    uint blocksPerPageZ = pc.pageSizeZ / pc.blockSize;
    uint totalBlocksInPage = blocksPerPageX * blocksPerPageY * blocksPerPageZ;

    if (localBlockID1D >= totalBlocksInPage) {
        return;
    }

    // Convert to 3D block coordinates within the page
    uvec3 blockCoord;
    blockCoord.z = localBlockID1D / (blocksPerPageX * blocksPerPageY);
    uint remainder = localBlockID1D % (blocksPerPageX * blocksPerPageY);
    blockCoord.y = remainder / blocksPerPageX;
    blockCoord.x = remainder % blocksPerPageX;

    // For streaming, we pass page-local block coordinates to loadMinMaxStreaming
    // The function expects coordinates in the range [0, blocksPerPage-1]
    uvec2 minMax = loadMinMaxStreaming(ivec3(blockCoord));

    bool blockIsActive = (pc.isoValue >= float(minMax.x) && pc.isoValue <= float(minMax.y));
    uint activeFlag = blockIsActive ? 1 : 0;
    
    // Debug output for first few blocks
    if (gl_WorkGroupID.x == 0 && localBlockID1D < 4) {
        debugPrintfEXT("Streaming filter: page(%d,%d,%d) block %d: coord=(%d,%d,%d), min=%d, max=%d, iso=%.1f, active=%d", 
                       pc.pageCoord.x, pc.pageCoord.y, pc.pageCoord.z,
                       localBlockID1D, blockCoord.x, blockCoord.y, blockCoord.z, 
                       minMax.x, minMax.y, pc.isoValue, activeFlag);
    }
    
    // --- Parallel Scan Implementation (same as original) ---

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
    if (activeFlag == 1) {
        // Store the local block ID within the page
        blockIDs[globalOffset] = localBlockID1D;
    }
}