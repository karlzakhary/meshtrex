#pragma once

inline VkSemaphore createSemaphore(VkDevice device)
{
    VkSemaphoreCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    VkSemaphore semaphore = 0;
    VK_CHECK(vkCreateSemaphore(device, &createInfo, 0, &semaphore));

    return semaphore;
}

inline VkCommandPool createCommandPool(VkDevice device, uint32_t familyIndex)
{
    VkCommandPoolCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    createInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    createInfo.queueFamilyIndex = familyIndex;
    createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool commandPool = 0;
    VK_CHECK(vkCreateCommandPool(device, &createInfo, 0, &commandPool));

    return commandPool;
}

inline VkQueryPool createQueryPool(VkDevice device, uint32_t queryCount,
                                   VkQueryType queryType)
{
    VkQueryPoolCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    createInfo.queryType = queryType;
    createInfo.queryCount = queryCount;

    if (queryType == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
        createInfo.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
    }

    VkQueryPool queryPool = nullptr;
    VK_CHECK(vkCreateQueryPool(device, &createInfo, 0, &queryPool));

    return queryPool;
}

inline VkFence createFence(VkDevice device)
{
    VkFenceCreateInfo createInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    createInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence = nullptr;
    VK_CHECK(vkCreateFence(device, &createInfo, 0, &fence));

    return fence;
}

inline VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    return commandBuffer;
}

// Reverted to original, more complete implementation
inline void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                           VkQueue queue, VkCommandBuffer commandBuffer) {
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer
    };

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static uint32_t selectMemoryType(
    const VkPhysicalDeviceMemoryProperties& memoryProperties,
    uint32_t memoryTypeBits, VkMemoryPropertyFlags flags)
{
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        if ((memoryTypeBits & (1 << i)) != 0 &&
            (memoryProperties.memoryTypes[i].propertyFlags & flags) == flags)
            return i;

    assert(!"No compatible memory type found");
    return ~0u;
}

inline VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,VkImageType viewType, uint32_t mipLevel, uint32_t levelCount)
{
    VkImageAspectFlags aspectMask = (format == VK_FORMAT_D32_SFLOAT)
                                        ? VK_IMAGE_ASPECT_DEPTH_BIT
                                        : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageViewCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    createInfo.image = image;
    createInfo.viewType = viewType == VK_IMAGE_TYPE_2D ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D;
    createInfo.format = format;
    createInfo.subresourceRange.aspectMask = aspectMask;
    createInfo.subresourceRange.baseMipLevel = mipLevel;
    createInfo.subresourceRange.levelCount = levelCount;
    createInfo.subresourceRange.layerCount = 1;

    VkImageView view = 0;
    VK_CHECK(vkCreateImageView(device, &createInfo, 0, &view));

    return view;
}