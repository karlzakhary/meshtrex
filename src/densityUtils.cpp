#include "densityUtils.h"
#include "vulkan_context.h"
#include "vulkan_utils.h"
#include <fstream>
#include <cstring>

namespace DensityUtils {

std::vector<uint32_t> readbackActiveBlocks(
    const VulkanContext& context,
    const Buffer& activeBlockIDsBuffer,
    uint32_t activeBlockCount) 
{
    if (activeBlockCount == 0) {
        return {};
    }
    
    size_t bufferSize = activeBlockCount * sizeof(uint32_t);
    
    // Create staging buffer for readback
    Buffer stagingBuffer;
    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy from device to staging
    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    
    VkBufferCopy copyRegion{};
    copyRegion.size = bufferSize;
    vkCmdCopyBuffer(cmd, activeBlockIDsBuffer.buffer, stagingBuffer.buffer, 1, &copyRegion);
    
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    
    // Read data from staging buffer
    std::vector<uint32_t> activeBlocks(activeBlockCount);
    memcpy(activeBlocks.data(), stagingBuffer.data, bufferSize);
    
    // Clean up staging buffer
    destroyBuffer(stagingBuffer, context.getDevice());
    
    return activeBlocks;
}

std::vector<uint8_t> readVolumeData(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        printf("Error: Could not open volume file: %s\n", filename.c_str());
        return {};
    }
    
    size_t fileSize = file.tellg();
    file.seekg(0);
    
    std::vector<uint8_t> data(fileSize);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();
    
    return data;
}

} // namespace DensityUtils