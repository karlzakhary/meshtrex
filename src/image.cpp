#include "common.h"
#include "image.h"
#include "vulkan_utils.h"

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
