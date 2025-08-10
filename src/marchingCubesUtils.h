#pragma once

#include "common.h"
#include "buffer.h"
#include "vulkan_context.h"
#include "mc_tables.h"
#include <cstdint>

struct MarchingCubesTables {
    Buffer numVerticesBuffer;
    Buffer triTableBuffer;
    bool isUnique = false;
};

class MarchingCubesUtils {
public:
    static void createMarchingCubesTables(
        MarchingCubesTables& tables,
        VkDevice device,
        const VulkanContext& context,
        bool useUniqueTables = false);
    
    static void destroyMarchingCubesTables(
        MarchingCubesTables& tables,
        VkDevice device);
    
    // Buffer views no longer needed when using storage buffers
    
private:
    static void uploadTableData(
        VkDevice device,
        const VulkanContext& context,
        VkBuffer dstBuffer,
        const void* data,
        size_t size);
};