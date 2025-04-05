#include "common.h"
#include "resources.h"

VkImageMemoryBarrier2 imageBarrier(
    VkImage image, VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask, VkImageLayout oldLayout,
    VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask,
    VkImageLayout newLayout, VkImageAspectFlags aspectMask,
    uint32_t baseMipLevel, uint32_t levelCount)
{
    VkImageMemoryBarrier2 result = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

    result.srcStageMask = srcStageMask;
    result.srcAccessMask = srcAccessMask;
    result.dstStageMask = dstStageMask;
    result.dstAccessMask = dstAccessMask;
    result.oldLayout = oldLayout;
    result.newLayout = newLayout;
    result.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    result.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    result.image = image;
    result.subresourceRange.aspectMask = aspectMask;
    result.subresourceRange.baseMipLevel = baseMipLevel;
    result.subresourceRange.levelCount = levelCount;
    result.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    return result;
}

void pipelineBarrier(VkCommandBuffer commandBuffer,
                     VkDependencyFlags dependencyFlags,
                     size_t bufferBarrierCount,
                     const VkBufferMemoryBarrier2 *bufferBarriers,
                     size_t imageBarrierCount,
                     const VkImageMemoryBarrier2 *imageBarriers)
{
    VkDependencyInfo dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependencyInfo.dependencyFlags = dependencyFlags;
    dependencyInfo.bufferMemoryBarrierCount = unsigned(bufferBarrierCount);
    dependencyInfo.pBufferMemoryBarriers = bufferBarriers;
    dependencyInfo.imageMemoryBarrierCount = unsigned(imageBarrierCount);
    dependencyInfo.pImageMemoryBarriers = imageBarriers;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,VkImageType viewType, uint32_t mipLevel, uint32_t levelCount)
{
    VkImageAspectFlags aspectMask = (format == VK_FORMAT_D32_SFLOAT)
                                        ? VK_IMAGE_ASPECT_DEPTH_BIT
                                        : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageViewCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    createInfo.image = image;
    createInfo.viewType = viewType == VK_IMAGE_VIEW_TYPE_2D ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D;
    createInfo.format = format;
    createInfo.subresourceRange.aspectMask = aspectMask;
    createInfo.subresourceRange.baseMipLevel = mipLevel;
    createInfo.subresourceRange.levelCount = levelCount;
    createInfo.subresourceRange.layerCount = 1;

    VkImageView view = 0;
    VK_CHECK(vkCreateImageView(device, &createInfo, 0, &view));

    return view;
}

void createImage(Image& result, VkDevice device,
                 const VkPhysicalDeviceMemoryProperties& memoryProperties,
                 VkImageType imageType,
                 uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels,
                 VkFormat format, VkImageUsageFlags usage)
{
    VkImageCreateInfo createInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};

    createInfo.imageType = imageType;
    createInfo.format = format;
    createInfo.extent = {width, height, depth};
    createInfo.mipLevels = mipLevels;
    createInfo.arrayLayers = 1;
    createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    createInfo.usage = usage;
    createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = nullptr;
    VK_CHECK(vkCreateImage(device, &createInfo, 0, &image));

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, image, &memoryRequirements);

    uint32_t memoryTypeIndex =
        selectMemoryType(memoryProperties, memoryRequirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    assert(memoryTypeIndex != ~0u);

    VkMemoryAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory memory = nullptr;
    VK_CHECK(vkAllocateMemory(device, &allocateInfo, 0, &memory));

    VK_CHECK(vkBindImageMemory(device, image, memory, 0));

    result.image = image;
    result.imageView = createImageView(device, image, format, imageType, 0, mipLevels);
    result.memory = memory;
}

void destroyImage(const Image& image, VkDevice device)
{
    vkDestroyImageView(device, image.imageView, nullptr);
    vkDestroyImage(device, image.image, nullptr);
    vkFreeMemory(device, image.memory, nullptr);
}