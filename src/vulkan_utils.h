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

    VkFence fence = nullptr;
    VK_CHECK(vkCreateFence(device, &createInfo, 0, &fence));

    return fence;
}