#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <array>

// Specialization data structure for block dimensions
struct BlockDimensionSpecializationData {
    uint32_t BX;
    uint32_t BY;
    uint32_t BZ;
};

// Helper class to manage shader specialization for dynamic block dimensions
class ShaderSpecialization {
public:
    static VkSpecializationInfo getBlockDimensionSpecialization(BlockDimensionSpecializationData& data) {
        static std::array<VkSpecializationMapEntry, 3> mapEntries = {{
            {0, offsetof(BlockDimensionSpecializationData, BX), sizeof(uint32_t)},
            {1, offsetof(BlockDimensionSpecializationData, BY), sizeof(uint32_t)},
            {2, offsetof(BlockDimensionSpecializationData, BZ), sizeof(uint32_t)}
        }};

        return VkSpecializationInfo{
            .mapEntryCount = static_cast<uint32_t>(mapEntries.size()),
            .pMapEntries = mapEntries.data(),
            .dataSize = sizeof(BlockDimensionSpecializationData),
            .pData = &data
        };
    }
};