#include "densityAnalyzer.h"
#include <algorithm>
#include <numeric>

std::vector<DensityAnalyzer::BlockDensity> DensityAnalyzer::analyzeActiveBlocks(
    const uint8_t* volumeData,
    const VolumeParams& params,
    const std::vector<uint32_t>& activeBlockIDs,
    float isovalue) 
{
    std::vector<BlockDensity> densities;
    densities.reserve(activeBlockIDs.size());
    
    // Analyze each active block
    for (uint32_t blockID : activeBlockIDs) {
        BlockDensity density = analyzeBlock(volumeData, params, blockID, isovalue);
        densities.push_back(density);
    }
    
    // Classify based on statistical distribution
    classifyBlocks(densities);
    
    // Print statistics for debugging
    int counts[4] = {0, 0, 0, 0};
    for (const auto& d : densities) {
        counts[d.classification]++;
    }
    printf("Block Classification: Empty=%d, Sparse=%d, Medium=%d, Dense=%d\n",
           counts[0], counts[1], counts[2], counts[3]);
    
    return densities;
}

DensityAnalyzer::BlockDensity DensityAnalyzer::analyzeBlock(
    const uint8_t* volume,
    const VolumeParams& params,
    uint32_t blockID,
    float isovalue) 
{
    BlockDensity result;
    result.blockID = blockID;
    
    // Unpack block coordinates
    glm::uvec3 blockCoord = unpackBlockID(blockID, params.blockGridDim);
    
    // Calculate block bounds in volume space
    glm::uvec3 blockStart = blockCoord * params.blockDim;
    glm::uvec3 blockEnd = glm::min(blockStart + params.blockDim + glm::uvec3(1), params.volumeDim);
    
    // Sample the block at regular intervals (3x3x3 grid)
    // This gives us a quick approximation of block complexity
    uint32_t crossings = 0;
    uint32_t totalSamples = 0;
    uint32_t edgeSamples = 0;
    
    const uint32_t SAMPLE_RATE = 4;  // Increased sampling for better coverage
    
    for (uint32_t sz = 0; sz < SAMPLE_RATE; sz++) {
        for (uint32_t sy = 0; sy < SAMPLE_RATE; sy++) {
            for (uint32_t sx = 0; sx < SAMPLE_RATE; sx++) {
                // Map to actual voxel coordinates
                glm::uvec3 samplePos;
                samplePos.x = blockStart.x + (blockEnd.x - blockStart.x - 1) * sx / (SAMPLE_RATE - 1);
                samplePos.y = blockStart.y + (blockEnd.y - blockStart.y - 1) * sy / (SAMPLE_RATE - 1);
                samplePos.z = blockStart.z + (blockEnd.z - blockStart.z - 1) * sz / (SAMPLE_RATE - 1);
                
                // Ensure we're within bounds
                if (samplePos.x < params.volumeDim.x && 
                    samplePos.y < params.volumeDim.y && 
                    samplePos.z < params.volumeDim.z) {
                    
                    uint32_t idx = samplePos.z * params.volumeDim.x * params.volumeDim.y + 
                                   samplePos.y * params.volumeDim.x + 
                                   samplePos.x;
                    
                    float value = float(volume[idx]);
                    
                    // Check if this sample is near the isovalue
                    float isoDist = std::abs(value - isovalue);
                    if (isoDist < 25.5f) { // Within 10% of 255 range
                        crossings++;
                    }
                    
                    // Also check for edge transitions
                    if (sx > 0 || sy > 0 || sz > 0) {
                        glm::uvec3 prevPos = samplePos;
                        if (sx > 0) prevPos.x--;
                        else if (sy > 0) prevPos.y--;
                        else prevPos.z--;
                        
                        uint32_t prevIdx = prevPos.z * params.volumeDim.x * params.volumeDim.y + 
                                          prevPos.y * params.volumeDim.x + 
                                          prevPos.x;
                        float prevValue = float(volume[prevIdx]);
                        
                        // Check for sign change across isovalue
                        if ((value - isovalue) * (prevValue - isovalue) < 0) {
                            edgeSamples++;
                        }
                    }
                    
                    totalSamples++;
                }
            }
        }
    }
    
    // Calculate complexity based on both crossings and edge transitions
    if (totalSamples > 0) {
        float crossingRatio = float(crossings) / float(totalSamples);
        float edgeRatio = float(edgeSamples) / float(totalSamples);
        
        // Combined complexity metric
        result.complexity = 0.7f * crossingRatio + 0.3f * edgeRatio;
    } else {
        result.complexity = 0.0f;
    }
    
    return result;
}

void DensityAnalyzer::classifyBlocks(std::vector<BlockDensity>& densities) 
{
    if (densities.empty()) return;
    
    // Calculate mean and standard deviation
    float sum = 0.0f;
    for (const auto& d : densities) {
        sum += d.complexity;
    }
    float mean = sum / densities.size();
    
    float variance = 0.0f;
    for (const auto& d : densities) {
        float diff = d.complexity - mean;
        variance += diff * diff;
    }
    float stddev = std::sqrt(variance / densities.size());
    
    // Classify based on statistical distribution
    for (auto& d : densities) {
        // Since this block was marked as active by the filtering pass,
        // it contains the isosurface even if our sampling missed it
        if (d.complexity < 0.001f) {
            // Classify as SPARSE instead of EMPTY since we know it's active
            d.classification = SPARSE;
        } else if (d.complexity < mean - stddev) {
            d.classification = SPARSE;
        } else if (d.complexity < mean + stddev) {
            d.classification = MEDIUM;
        } else {
            d.classification = DENSE;
        }
    }
}

glm::uvec3 DensityAnalyzer::unpackBlockID(uint32_t blockID, const glm::uvec3& blockGridDim) 
{
    uint32_t x = blockID % blockGridDim.x;
    uint32_t y = (blockID / blockGridDim.x) % blockGridDim.y;
    uint32_t z = blockID / (blockGridDim.x * blockGridDim.y);
    return glm::uvec3(x, y, z);
}