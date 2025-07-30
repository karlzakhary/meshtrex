#pragma once

#include "common.h"
#include "buffer.h"
#include <vector>

class VulkanContext;

// Utility functions for density-based dispatch
namespace DensityUtils {
    
    // Read back active block IDs from GPU buffer
    std::vector<uint32_t> readbackActiveBlocks(
        const VulkanContext& context,
        const Buffer& activeBlockIDsBuffer,
        uint32_t activeBlockCount);
    
    // Read volume data from file (for CPU analysis)
    std::vector<uint8_t> readVolumeData(const std::string& filename);
    
};