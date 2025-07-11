#pragma once

#include <stdint.h>
#include <glm/glm.hpp>

struct PageTableEntry {
    uint32_t atlasX;
    uint32_t atlasY;
    uint32_t atlasZ;
    uint32_t isResident;
};

struct StreamingConstants {
    uint32_t pageSizeX;
    uint32_t pageSizeY;
    uint32_t pageSizeZ;
    uint32_t atlasSizeX;
    uint32_t atlasSizeY;
    uint32_t atlasSizeZ;
    uint32_t volumeWidth;
    uint32_t volumeHeight;
    uint32_t volumeDepth;
    uint32_t maxMipLevel;
    float isoValue;
    uint32_t frameIndex;
    uint32_t padding[2];
};

struct IndirectDispatchCommand {
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
    uint32_t passType;
    uint32_t pageCoordX;
    uint32_t pageCoordY;
    uint32_t pageCoordZ;
    uint32_t mipLevel;
};

enum PassType {
    PASS_MIN_MAX_LEAF = 0,
    PASS_MIN_MAX_REDUCE = 1,
    PASS_ACTIVE_BLOCK_FILTER = 2,
    PASS_MESH_EXTRACTION = 3
};

struct WorkQueueHeader {
    uint32_t commandCount;
    uint32_t maxCommands;
    uint32_t frameIndex;
    uint32_t padding;
};

// Constants specific to each streaming pass type
struct StreamingMinMaxConstants {
    glm::uvec3 pageCoord;
    uint32_t mipLevel;
    glm::uvec3 srcDim;
    uint32_t padding1;
    glm::uvec3 dstDim;
    uint32_t padding2;
    uint32_t pageSize;
    float isoValue;
    uint32_t padding3[2];
};

struct StreamingFilteringConstants {
    glm::uvec3 pageCoord;
    uint32_t mipLevel;
    uint32_t blockSize;
    uint32_t pageSize;
    float isoValue;
    uint32_t padding;
};

struct StreamingExtractionConstants {
    glm::uvec3 pageCoord;
    uint32_t mipLevel;
    float isoValue;
    uint32_t blockSize;
    uint32_t pageSizeX;
    uint32_t pageSizeY;
    uint32_t pageSizeZ;
};

struct StreamingMinMaxPushConstants {
    glm::uvec3 pageCoord;
    uint32_t mipLevel;
    float isoValue;
    uint32_t blockSize;
    uint32_t pageSizeX;
    uint32_t pageSizeY;
    uint32_t pageSizeZ;
    uint32_t volumeSizeX;
    uint32_t volumeSizeY;
    uint32_t volumeSizeZ;
    uint32_t granularityX;
    uint32_t granularityY;
    uint32_t granularityZ;
    uint32_t pageOverlap; // Number of voxels of overlap between pages (for min-max halo)
};
