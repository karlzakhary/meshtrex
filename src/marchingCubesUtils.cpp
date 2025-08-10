#include "marchingCubesUtils.h"
#include "vulkan_utils.h"
#include <cstring>

void MarchingCubesUtils::createMarchingCubesTables(
    MarchingCubesTables& tables,
    VkDevice device,
    const VulkanContext& context,
    bool useUniqueTables) {
    
    tables.isUnique = useUniqueTables;
    
    // Create buffer for numVertices table (256 entries of uint8)
    size_t numVerticesSize = 256 * sizeof(uint8_t);
    createBuffer(tables.numVerticesBuffer, device, context.getMemoryProperties(),
                 numVerticesSize, 
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Create buffer for triangle table
    size_t triTableSize;
    const void* triTableData;
    const void* numVerticesData;
    
    if (useUniqueTables) {
        // Unique tables: 256 x 16 uint8 values
        triTableSize = 256 * 16 * sizeof(uint8_t);
        triTableData = &MarchingCubes::uniqueTriTable[0][0];
        numVerticesData = &MarchingCubes::numUniqueVertsTable[0];
    } else {
        // Standard tables: 256 x 16 uint8 values  
        triTableSize = 256 * 16 * sizeof(uint8_t);
        triTableData = &MarchingCubes::triTable[0][0];
        numVerticesData = &MarchingCubes::numVerticesTable[0];
    }
    
    createBuffer(tables.triTableBuffer, device, context.getMemoryProperties(),
                 triTableSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Upload data to buffers
    uploadTableData(device, context, tables.numVerticesBuffer.buffer, 
                    numVerticesData, numVerticesSize);
    uploadTableData(device, context, tables.triTableBuffer.buffer,
                    triTableData, triTableSize);
    
    // No longer need buffer views when using storage buffers
}

void MarchingCubesUtils::destroyMarchingCubesTables(
    MarchingCubesTables& tables,
    VkDevice device) {
    
    // Buffer views no longer needed when using storage buffers
    if (tables.numVerticesBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(tables.numVerticesBuffer, device);
    }
    if (tables.triTableBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(tables.triTableBuffer, device);
    }
}

// Buffer views are no longer needed when using storage buffers
// void MarchingCubesUtils::createTableBufferViews(
//     MarchingCubesTables& tables,
//     VkDevice device) {
// }

void MarchingCubesUtils::uploadTableData(
    VkDevice device,
    const VulkanContext& context,
    VkBuffer dstBuffer,
    const void* data,
    size_t size) {
    
    // Create staging buffer
    Buffer stagingBuffer;
    createBuffer(stagingBuffer, device, context.getMemoryProperties(),
                 size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy data to staging buffer
    memcpy(stagingBuffer.data, data, size);
    
    // Transfer to device local buffer
    VkCommandBuffer cmd = beginSingleTimeCommands(device, context.getCommandPool());
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, dstBuffer, 1, &copyRegion);
    endSingleTimeCommands(device, context.getCommandPool(), context.getQueue(), cmd);
    
    // Clean up staging buffer
    destroyBuffer(stagingBuffer, device);
}