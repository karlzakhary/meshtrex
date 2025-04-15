#pragma once

VkBufferMemoryBarrier2 bufferBarrier(
    VkBuffer buffer,
    VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask,
    VkDeviceSize offset, VkDeviceSize size);

VkImageMemoryBarrier2 imageBarrier(
    VkImage image, VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask, VkImageLayout oldLayout,
    VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask,
    VkImageLayout newLayout,
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    uint32_t baseMipLevel = 0, uint32_t levelCount = VK_REMAINING_MIP_LEVELS);

void pipelineBarrier(VkCommandBuffer commandBuffer,
                     VkDependencyFlags dependencyFlags,
                     size_t bufferBarrierCount,
                     const VkBufferMemoryBarrier2 *bufferBarriers,
                     size_t imageBarrierCount,
                     const VkImageMemoryBarrier2 *imageBarriers);

void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
                     VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask);
