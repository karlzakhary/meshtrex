#include "common.h"

#include "buffer.h"

#include <cstring>

#include "resources.h"

void createBuffer(Buffer& result, VkDevice device,
                  const VkPhysicalDeviceMemoryProperties& memoryProperties,
                  size_t size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags memoryFlags)
{
    VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.size = size;
    createInfo.usage = usage;

    VkBuffer buffer = 0;
    VK_CHECK(vkCreateBuffer(device, &createInfo, 0, &buffer));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    uint32_t memoryTypeIndex = selectMemoryType(
        memoryProperties, memoryRequirements.memoryTypeBits, memoryFlags);
    assert(memoryTypeIndex != ~0u);

    VkMemoryAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;

    VkMemoryAllocateFlagsInfo flagInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};

    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocateInfo.pNext = &flagInfo;
        flagInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagInfo.deviceMask = 1;
    }

    VkDeviceMemory memory = 0;
    VK_CHECK(vkAllocateMemory(device, &allocateInfo, 0, &memory));

    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    void *data = 0;
    if (memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(device, memory, 0, size, 0, &data));

    result.buffer = buffer;
    result.memory = memory;
    result.data = data;
    result.size = size;
}

void uploadBuffer(VkDevice device, VkCommandPool commandPool,
                  VkCommandBuffer commandBuffer, VkQueue queue,
                  const Buffer& buffer, const Buffer& scratch, const void *data,
                  size_t size)
{
    // TODO: this function is submitting a command buffer and waiting for device
    // idle for each buffer upload; this is obviously suboptimal and we'd need
    // to batch this later
    assert(size > 0);
    assert(scratch.data);
    assert(scratch.size >= size);
    memcpy(scratch.data, data, size);

    VK_CHECK(vkResetCommandPool(device, commandPool, 0));

    VkCommandBufferBeginInfo beginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkBufferCopy region = {0, 0, VkDeviceSize(size)};
    vkCmdCopyBuffer(commandBuffer, scratch.buffer, buffer.buffer, 1, &region);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    VK_CHECK(vkDeviceWaitIdle(device));
}

void copy1DBufferTo3DImage(Buffer stagingBuffer,
    VkCommandBuffer commandBuffer,VkImage volumeImage,
    uint32_t width, uint32_t height, uint32_t depth)
{
    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;     // tightly packed
    copyRegion.bufferImageHeight = 0;

    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;

    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {width, height, depth};

    vkCmdCopyBufferToImage(
    commandBuffer,
    stagingBuffer.buffer,
    volumeImage,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    1,
    &copyRegion
);
}

void destroyBuffer(const Buffer& buffer, VkDevice device)
{
    vkDestroyBuffer(device, buffer.buffer, 0);
    vkFreeMemory(device, buffer.memory, 0);
}

VkDeviceAddress getBufferAddress(const Buffer& buffer, VkDevice device)
{
    VkBufferDeviceAddressInfo info = {
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer.buffer;

    VkDeviceAddress address = vkGetBufferDeviceAddress(device, &info);
    assert(address != 0);

    return address;
}