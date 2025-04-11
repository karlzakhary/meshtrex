#pragma once

struct Image {
    VkImage image;
    VkImageView imageView;
    VkDeviceMemory memory;
};

void createImage(Image& result, VkDevice device,
                 const VkPhysicalDeviceMemoryProperties& memoryProperties,
                 VkImageType imageType,
                 uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels,
                 VkFormat format, VkImageUsageFlags usage);
void destroyImage(const Image& image, VkDevice device);