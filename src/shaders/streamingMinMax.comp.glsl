#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 2) in;

layout(push_constant) uniform PushConstants {
    uvec3 pageCoord;
    uint mipLevel;
    float isoValue;
    uint blockSize;
    uint pageSizeX;
    uint pageSizeY;
    uint pageSizeZ;
    uint volumeSizeX;
    uint volumeSizeY;
    uint volumeSizeZ;
    uint granularityX;
    uint granularityY;
    uint granularityZ;
    uint pageOverlap;
} pc;

layout(set = 0, binding = 0, std430) readonly buffer PageTable {
    uvec2 pageEntries[];
} pageTable;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeAtlas;
layout(set = 0, binding = 2, rg32ui) uniform writeonly uimage3D minMaxOutputVolume;

const uint totalInvocations = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
shared uvec2 s_subgroupMinMax[totalInvocations / 32];

uint getPageIndex(uvec3 pageCoord, uint mipLevel) {
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
    uint atlasCoord = pageTable.pageEntries[pageIndex].x;
    return uvec3(
        atlasCoord & 0x3FF,
        (atlasCoord >> 10) & 0x3FF,
        (atlasCoord >> 20) & 0x3FF
    );
}

uint sampleVolumeAtlas(uvec3 worldCoord, out bool valid) {
    valid = true;
    if (worldCoord.x >= pc.volumeSizeX || worldCoord.y >= pc.volumeSizeY || worldCoord.z >= pc.volumeSizeZ) {
        valid = false;
        return 0;
    }
    
    uvec3 pageCoord = uvec3(worldCoord.x / pc.pageSizeX,
                            worldCoord.y / pc.pageSizeY,
                            worldCoord.z / pc.pageSizeZ);
    
    if (!isPageResident(pageCoord, pc.mipLevel)) {
        valid = false;
        return 0;
    }
    
    uvec3 atlasCoord = getAtlasCoord(pageCoord, pc.mipLevel);
    uvec3 localCoord = uvec3(worldCoord.x % pc.pageSizeX,
                             worldCoord.y % pc.pageSizeY,
                             worldCoord.z % pc.pageSizeZ);
    
    ivec3 atlasTexel = ivec3(atlasCoord.x * pc.pageSizeX + localCoord.x,
                             atlasCoord.y * pc.pageSizeY + localCoord.y,
                             atlasCoord.z * pc.pageSizeZ + localCoord.z);
    
    if (any(greaterThanEqual(atlasTexel, ivec3(1024)))) {
        return 0;
    }
    
    return imageLoad(volumeAtlas, atlasTexel).r;
}

void main() {
    const uvec3 localInvocationID = gl_LocalInvocationID;
    const uint localInvocationIndex = gl_LocalInvocationIndex;
    const uint subgroupId = gl_SubgroupID;
    const uint subgroupInvocationId = gl_SubgroupInvocationID;
    
    const ivec3 pageStart = ivec3(pc.pageSizeX * pc.pageCoord.x,
                                  pc.pageSizeY * pc.pageCoord.y,
                                  pc.pageSizeZ * pc.pageCoord.z);
    const ivec3 blockStart = pageStart + ivec3(gl_WorkGroupID) * int(pc.blockSize);
    
    uint invocationMin = 0xFFFFFFFFu;
    uint invocationMax = 0u;
    
    {
        ivec3 voxelCoord = blockStart + ivec3(localInvocationID);
        bool valid;
        uint value = sampleVolumeAtlas(uvec3(voxelCoord), valid);
        if (valid) {
            invocationMin = min(invocationMin, value);
            invocationMax = max(invocationMax, value);
        }
    }
    
    if (localInvocationIndex < 61) {
        ivec3 extraVoxel;
        if (localInvocationIndex < 16) {
            uint y = localInvocationIndex % pc.blockSize;
            uint z = localInvocationIndex / pc.blockSize;
            extraVoxel = blockStart + ivec3(pc.blockSize, y, z);
        } else if (localInvocationIndex < 32) {
            uint idx = localInvocationIndex - 16;
            uint x = idx % pc.blockSize;
            uint z = idx / pc.blockSize;
            extraVoxel = blockStart + ivec3(x, pc.blockSize, z);
        } else if (localInvocationIndex < 48) {
            uint idx = localInvocationIndex - 32;
            uint x = idx % pc.blockSize;
            uint y = idx / pc.blockSize;
            extraVoxel = blockStart + ivec3(x, y, pc.blockSize);
        } else if (localInvocationIndex < 52) {
            uint z = localInvocationIndex - 48;
            extraVoxel = blockStart + ivec3(pc.blockSize, pc.blockSize, z);
        } else if (localInvocationIndex < 56) {
            uint y = localInvocationIndex - 52;
            extraVoxel = blockStart + ivec3(pc.blockSize, y, pc.blockSize);
        } else {
            uint x = localInvocationIndex - 56;
            extraVoxel = blockStart + ivec3(x, pc.blockSize, pc.blockSize);
        }
        
        if (localInvocationIndex == 60) {
            extraVoxel = blockStart + ivec3(pc.blockSize, pc.blockSize, pc.blockSize);
        }
        
        bool valid;
        uint value = sampleVolumeAtlas(uvec3(extraVoxel), valid);
        if (valid) {
            invocationMin = min(invocationMin, value);
            invocationMax = max(invocationMax, value);
        }
    }
    
    for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
        uint downMin = subgroupShuffleDown(invocationMin, offset);
        uint downMax = subgroupShuffleDown(invocationMax, offset);
        invocationMin = min(invocationMin, downMin);
        invocationMax = max(invocationMax, downMax);
    }
    
    if (subgroupInvocationId == 0) {
        if (subgroupId < (totalInvocations / 32)) {
            s_subgroupMinMax[subgroupId] = uvec2(invocationMin, invocationMax);
        }
    }
    
    barrier();
    
    if (subgroupId == 0) {
        uvec2 subgroupResult = uvec2(0xFFFFFFFFu, 0u);
        uint sharedMemIndex = subgroupInvocationId;
        uint numSubgroups = totalInvocations / gl_SubgroupSize;
        if(sharedMemIndex < numSubgroups && sharedMemIndex < (totalInvocations / 32)) {
            subgroupResult = s_subgroupMinMax[sharedMemIndex];
        }
        
        for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
            uint downMin = subgroupShuffleDown(subgroupResult.x, offset);
            uint downMax = subgroupShuffleDown(subgroupResult.y, offset);
            subgroupResult.x = min(subgroupResult.x, downMin);
            subgroupResult.y = max(subgroupResult.y, downMax);
        }
        
        if (subgroupInvocationId == 0) {
            imageStore(minMaxOutputVolume, ivec3(gl_WorkGroupID), uvec4(subgroupResult, 0, 0));
        }
    }
}