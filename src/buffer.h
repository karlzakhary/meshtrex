#pragma once

struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *data;
    size_t size;
};

void createBuffer(Buffer& result, VkDevice device,
                  const VkPhysicalDeviceMemoryProperties& memoryProperties,
                  size_t size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags memoryFlags);
void uploadBuffer(VkDevice device, VkCommandPool commandPool,
                  VkCommandBuffer commandBuffer, VkQueue queue,
                  const Buffer& buffer, const Buffer& scratch, const void *data,
                  size_t size);
void destroyBuffer(const Buffer& buffer, VkDevice device);

void copy1DBufferTo3DImage(Buffer stagingBuffer,
    VkCommandBuffer commandBuffer,VkImage volumeImage,
    uint32_t width, uint32_t height, uint32_t depth);

void copy3DImageTo1DBuffer(Buffer readbackBuffer,
    VkCommandBuffer cmd,VkImage volumeImage,
    VkExtent3D extent);

VkDeviceAddress getBufferAddress(const Buffer& buffer, VkDevice device);
