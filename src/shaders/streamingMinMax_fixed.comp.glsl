#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_debug_printf : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 2) in;

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

layout(set = 0, binding = 0, std430) readonly buffer PageTable {
    uvec2 pageEntries[];
} pageTable;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeAtlas;

layout(set = 0, binding = 2, rg32ui) uniform writeonly uimage3D minMaxOutputVolume;

// Shared memory for intermediate min/max values per subgroup
const uint totalInvocations = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
shared uvec2 s_subgroupMinMax[totalInvocations / 32]; // Assuming min subgroup size of 32

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

uvec3 getAtlasCoord(uvec3 pageCoord, uint mipLevel) {
    uint pageIndex = getPageIndex(pageCoord, mipLevel);
    if (pageIndex >= pageTable.pageEntries.length()) {
        return uvec3(0);  // Invalid page
    }
    uint atlasCoord = pageTable.pageEntries[pageIndex].x;
    
    return uvec3(
        atlasCoord & 0x3FF,
        (atlasCoord >> 10) & 0x3FF,
        (atlasCoord >> 20) & 0x3FF
    );
}

uint sampleVolumeAtlas(uvec3 worldCoord) {
    uvec3 pageCoord = uvec3(worldCoord.x / pc.pageSizeX,
                            worldCoord.y / pc.pageSizeY,
                            worldCoord.z / pc.pageSizeZ);
    
    if (!isPageResident(pageCoord, pc.mipLevel)) {
        // Return a special value to indicate invalid sample
        return 0xFFFFFFFFu;
    }
    
    uvec3 atlasCoord = getAtlasCoord(pageCoord, pc.mipLevel);
    uvec3 localCoord = uvec3(worldCoord.x % pc.pageSizeX,
                             worldCoord.y % pc.pageSizeY,
                             worldCoord.z % pc.pageSizeZ);
    
    // atlasCoord from page table is in granularity units (divided by granularity during packing)
    // We need to multiply by granularity to get back to texel coordinates
    // The granularity is 64x32x32 as reported by the sparse image requirements
    ivec3 atlasTexel = ivec3(atlasCoord.x * 64 + localCoord.x,
                             atlasCoord.y * 32 + localCoord.y,
                             atlasCoord.z * 32 + localCoord.z);
    
    // Bounds check before imageLoad
    if (any(greaterThanEqual(atlasTexel, ivec3(1024)))) {
        return 255;
    }
    
    // Use imageLoad for integer textures
    uint value = imageLoad(volumeAtlas, atlasTexel).r;
    
    return value;
}

// Alternative approach: track if we have valid samples
struct MinMaxAccumulator {
    uint minVal;
    uint maxVal;
    bool hasValidSamples;
};

void accumulateSample(inout MinMaxAccumulator acc, uvec3 worldCoord) {
    // Check bounds first - don't sample outside volume
    if (worldCoord.x >= pc.volumeSizeX || worldCoord.y >= pc.volumeSizeY || worldCoord.z >= pc.volumeSizeZ) {
        return;
    }
    
    // Calculate which page this voxel belongs to
    uvec3 pageCoord = uvec3(worldCoord.x / pc.pageSizeX,
                            worldCoord.y / pc.pageSizeY,
                            worldCoord.z / pc.pageSizeZ);
    
    // Get the page index
    uint pageIndex = getPageIndex(pageCoord, pc.mipLevel);
    if (pageIndex >= pageTable.pageEntries.length()) {
        return;
    }
    
    // Check if page is resident
    if (pageTable.pageEntries[pageIndex].y == 0) {
        // Page not resident - this shouldn't happen for face neighbors
        // since CPU ensures they're loaded
        return;
    }
    
    // Get atlas coordinates
    uint atlasCoordPacked = pageTable.pageEntries[pageIndex].x;
    uvec3 atlasCoord = uvec3(
        atlasCoordPacked & 0x3FF,
        (atlasCoordPacked >> 10) & 0x3FF,
        (atlasCoordPacked >> 20) & 0x3FF
    );
    
    // Calculate local coordinate within the page
    uvec3 localCoord = uvec3(worldCoord.x % pc.pageSizeX,
                             worldCoord.y % pc.pageSizeY,
                             worldCoord.z % pc.pageSizeZ);
    
    // Calculate final atlas texel coordinate
    ivec3 atlasTexel = ivec3(atlasCoord.x * 64 + localCoord.x,
                             atlasCoord.y * 32 + localCoord.y,
                             atlasCoord.z * 32 + localCoord.z);
    
    // Bounds check
    if (any(greaterThanEqual(atlasTexel, ivec3(1024)))) {
        return;
    }
    
    // Sample the atlas
    uint value = imageLoad(volumeAtlas, atlasTexel).r;
    
    // Update accumulator
    if (!acc.hasValidSamples) {
        acc.minVal = value;
        acc.maxVal = value;
        acc.hasValidSamples = true;
    } else {
        acc.minVal = min(acc.minVal, value);
        acc.maxVal = max(acc.maxVal, value);
    }
}

void main() {
    const uvec3 localInvocationID = gl_LocalInvocationID;
    const uint localInvocationIndex = gl_LocalInvocationIndex;
    const uint subgroupId = gl_SubgroupID;
    const uint subgroupInvocationId = gl_SubgroupInvocationID;
    
    // Calculate the starting position of this block relative to the page
    const ivec3 pageStart = ivec3(pc.pageSizeX * pc.pageCoord.x,
                                  pc.pageSizeY * pc.pageCoord.y,
                                  pc.pageSizeZ * pc.pageCoord.z);
    const ivec3 blockStart = pageStart + ivec3(gl_WorkGroupID) * int(pc.blockSize);
    
    // Debug: Show blocks at page boundaries
    bool isPageBoundaryBlock = (gl_WorkGroupID.x == 0 || gl_WorkGroupID.x >= (pc.pageSizeX/pc.blockSize - 1) ||
                                gl_WorkGroupID.y == 0 || gl_WorkGroupID.y >= (pc.pageSizeY/pc.blockSize - 1) ||
                                gl_WorkGroupID.z == 0 || gl_WorkGroupID.z >= (pc.pageSizeZ/pc.blockSize - 1));
    
    if (gl_LocalInvocationIndex == 0 && gl_WorkGroupID.x == 3 && gl_WorkGroupID.y == 0 && gl_WorkGroupID.z == 0) {
        debugPrintfEXT("DEBUG: Processing block (3,0,0) of page(%d,%d,%d), world start=(%d,%d,%d)", 
                       pc.pageCoord.x, pc.pageCoord.y, pc.pageCoord.z,
                       blockStart.x, blockStart.y, blockStart.z);
        
        // Check which voxels we'll be sampling
        for (int dx = 0; dx <= 4; dx++) {
            ivec3 voxelCoord = blockStart + ivec3(dx, 0, 0);
            uvec3 voxelPageCoord = uvec3(voxelCoord.x / int(pc.pageSizeX),
                                         voxelCoord.y / int(pc.pageSizeY),
                                         voxelCoord.z / int(pc.pageSizeZ));
            bool resident = isPageResident(voxelPageCoord, pc.mipLevel);
            debugPrintfEXT("  Voxel at world(%d,%d,%d) -> page(%d,%d,%d) resident=%d",
                           voxelCoord.x, voxelCoord.y, voxelCoord.z,
                           voxelPageCoord.x, voxelPageCoord.y, voxelPageCoord.z,
                           resident ? 1 : 0);
        }
    }
    
    // Use accumulator approach to handle non-resident pages correctly
    MinMaxAccumulator acc;
    acc.minVal = 0xFFFFFFFFu;
    acc.maxVal = 0u;
    acc.hasValidSamples = false;
    
    // First voxel: Each thread samples one voxel in the 5x5x5 region
    {
        ivec3 voxelCoord = blockStart + ivec3(localInvocationID.x % (pc.blockSize + 1),
                                               localInvocationID.y % (pc.blockSize + 1),
                                               localInvocationID.z % (pc.blockSize + 1));
        accumulateSample(acc, uvec3(voxelCoord));
    }
    
    // Second voxel: Handle the boundary layer
    if (localInvocationIndex < 61) {
        ivec3 extraVoxel;
        
        if (localInvocationIndex < 16) {
            // Handle x boundary
            uint y = localInvocationIndex % pc.blockSize;
            uint z = localInvocationIndex / pc.blockSize;
            extraVoxel = blockStart + ivec3(pc.blockSize, y, z);
        }
        else if (localInvocationIndex < 32) {
            // Handle y boundary
            uint idx = localInvocationIndex - 16;
            uint x = idx % pc.blockSize;
            uint z = idx / pc.blockSize;
            extraVoxel = blockStart + ivec3(x, pc.blockSize, z);
        }
        else if (localInvocationIndex < 48) {
            // Handle z boundary
            uint idx = localInvocationIndex - 32;
            uint x = idx % pc.blockSize;
            uint y = idx / pc.blockSize;
            extraVoxel = blockStart + ivec3(x, y, pc.blockSize);
        }
        else if (localInvocationIndex < 52) {
            // Handle x,y edge
            uint z = localInvocationIndex - 48;
            extraVoxel = blockStart + ivec3(pc.blockSize, pc.blockSize, z);
        }
        else if (localInvocationIndex < 56) {
            // Handle x,z edge
            uint y = localInvocationIndex - 52;
            extraVoxel = blockStart + ivec3(pc.blockSize, y, pc.blockSize);
        }
        else if (localInvocationIndex < 60) {
            // Handle y,z edge
            uint x = localInvocationIndex - 56;
            extraVoxel = blockStart + ivec3(x, pc.blockSize, pc.blockSize);
        }
        else {
            // Handle the corner
            extraVoxel = blockStart + ivec3(pc.blockSize);
        }
        
        accumulateSample(acc, uvec3(extraVoxel));
    }
    
    // If we have no valid samples (all voxels were in non-resident pages),
    // use values that won't participate in the reduction
    uint invocationMin = acc.hasValidSamples ? acc.minVal : 0xFFFFFFFFu;
    uint invocationMax = acc.hasValidSamples ? acc.maxVal : 0u;
    
    // Debug: Check if we're getting valid samples
    if (gl_LocalInvocationIndex == 0 && isPageBoundaryBlock && !acc.hasValidSamples) {
        debugPrintfEXT("WARNING: No valid samples for boundary block at page(%d,%d,%d) block(%d,%d,%d)", 
                       pc.pageCoord.x, pc.pageCoord.y, pc.pageCoord.z,
                       gl_WorkGroupID.x, gl_WorkGroupID.y, gl_WorkGroupID.z);
    }
    
    // --- Step 1: Subgroup-level reduction using subgroupShuffleDown ---
    for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
        uint downMin = subgroupShuffleDown(invocationMin, offset);
        uint downMax = subgroupShuffleDown(invocationMax, offset);
        invocationMin = min(invocationMin, downMin);
        invocationMax = max(invocationMax, downMax);
    }
    
    // --- Step 2: Write partial results to shared memory ---
    if (subgroupInvocationId == 0) {
        if (subgroupId < (totalInvocations / 32)) {
            s_subgroupMinMax[subgroupId] = uvec2(invocationMin, invocationMax);
        }
    }
    
    barrier();
    
    // --- Step 3: Final reduction using the first subgroup ---
    if (subgroupId == 0) {
        uvec2 subgroupResult = uvec2(0xFFFFFFFFu, 0u);
        
        uint sharedMemIndex = subgroupInvocationId;
        uint numSubgroups = totalInvocations / gl_SubgroupSize;
        if(sharedMemIndex < numSubgroups && sharedMemIndex < (totalInvocations / 32)) {
            subgroupResult = s_subgroupMinMax[sharedMemIndex];
        }
        
        // Final reduction within the first subgroup
        for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
            uint downMin = subgroupShuffleDown(subgroupResult.x, offset);
            uint downMax = subgroupShuffleDown(subgroupResult.y, offset);
            subgroupResult.x = min(subgroupResult.x, downMin);
            subgroupResult.y = max(subgroupResult.y, downMax);
        }
        
        // --- Step 4: Write final block result ---
        if (subgroupInvocationId == 0) {
            // Write to output volume - one result per workgroup
            imageStore(minMaxOutputVolume, ivec3(gl_WorkGroupID), uvec4(subgroupResult, 0, 0));
            
            // Debug output for boundary blocks
            if (isPageBoundaryBlock && (subgroupResult.x <= 80 && subgroupResult.y >= 80)) {
                debugPrintfEXT("Boundary block active: page(%d,%d,%d) block(%d,%d,%d) min=%d max=%d", 
                               pc.pageCoord.x, pc.pageCoord.y, pc.pageCoord.z,
                               gl_WorkGroupID.x, gl_WorkGroupID.y, gl_WorkGroupID.z,
                               subgroupResult.x, subgroupResult.y);
            }
        }
    }
}