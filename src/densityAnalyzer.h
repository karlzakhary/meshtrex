#pragma once

#include "common.h"
#include <vector>
#include <cmath>
#include <glm/glm.hpp>

class DensityAnalyzer {
public:
    // Classification categories for blocks
    enum BlockClass {
        EMPTY = 0,
        SPARSE = 1,
        MEDIUM = 2,
        DENSE = 3
    };

    struct BlockDensity {
        uint32_t blockID;
        float complexity;           // 0.0 = empty, 1.0 = maximum complexity
        BlockClass classification;  // Classification based on statistical analysis
    };

    struct VolumeParams {
        glm::uvec3 volumeDim;
        glm::uvec3 blockDim;
        glm::uvec3 blockGridDim;
    };

    // Analyze only active blocks (those that passed min-max filtering)
    std::vector<BlockDensity> analyzeActiveBlocks(
        const uint8_t* volumeData,
        const VolumeParams& params,
        const std::vector<uint32_t>& activeBlockIDs,
        float isovalue);

private:
    // Analyze a single block to estimate its complexity
    BlockDensity analyzeBlock(
        const uint8_t* volume,
        const VolumeParams& params,
        uint32_t blockID,
        float isovalue);

    // Classify blocks based on statistical distribution
    void classifyBlocks(std::vector<BlockDensity>& densities);

    // Helper to unpack block ID to 3D coordinates
    glm::uvec3 unpackBlockID(uint32_t blockID, const glm::uvec3& blockGridDim);
};