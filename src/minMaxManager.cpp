#include "minMaxManager.h"
#include "common.h"
#include "vulkan_context.h"
#include "resources.h"
#include "buffer.h"
#include "image.h"
#include "vulkan_utils.h"
#include "minMaxPass.h"
#include "minMaxOutput.h"
#include "blockFilteringTestUtils.h"
#include "activeBlockFilteringPass.h"
#include "streamingShaderInterface.h"
#include <cstring>
#include <iostream>
#include <string>

void uploadVolumeData(VkCommandBuffer commandBuffer,
    VkImage volumeImage, Buffer stagingBuffer, VkExtent3D extent)
{
    VkImageMemoryBarrier2 preCopyBarrier = imageBarrier(
        volumeImage,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_PIPELINE_STAGE_2_COPY_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0, 1
    );
    pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &preCopyBarrier);

    copy1DBufferTo3DImage(stagingBuffer, commandBuffer, volumeImage, extent.width, extent.height, extent.depth);

    VkImageMemoryBarrier2 postCopyPreComputeBarrier = imageBarrier(
        volumeImage,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
        0, 1
    );
    pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &postCopyPreComputeBarrier);
}

MinMaxOutput computeMinMaxMip(VulkanContext& context, Volume volume, PushConstants& pushConstants)
{
    std::string minMaxLeafShaderPath = "/spirv/minMaxLeaf.comp.spv";
    std::string minMaxOctreeReduceShaderPath = "/spirv/minMaxOctreeReduce.comp.spv";
    MinMaxPass minMaxPass(context, minMaxLeafShaderPath.c_str(), minMaxOctreeReduceShaderPath.c_str());

    MinMaxOutput output{};
    Buffer stagingBuffer = {};
    Buffer countReadbackBuffer = {};

    auto mipExtent = [](VkExtent3D e, uint32_t level) {
        return VkExtent3D{
            std::max(1u, e.width >> level),
            std::max(1u, e.height >> level),
            std::max(1u, e.depth >> level)
        };
    };
    VkExtent3D srcExtent = { volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z };
    VkExtent3D leafExtent = { pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z };
    uint32_t maxDim = std::max({ leafExtent.width, leafExtent.height, leafExtent.depth });
    uint32_t fullMipCount = 1u + static_cast<uint32_t>(std::floor(std::log2(maxDim)));
    output.minMaxMipViews.reserve(fullMipCount);

    createImage(output.minMaxImage, context.getDevice(), context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
        leafExtent.width, leafExtent.height, leafExtent.depth, fullMipCount,
        VK_FORMAT_R32G32_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    for (uint32_t l = 0; l < fullMipCount; ++l) {
        output.minMaxMipViews.push_back(createImageView(context.getDevice(), output.minMaxImage.image, VK_FORMAT_R32G32_UINT, VK_IMAGE_TYPE_3D, l, 1));
        VkImageMemoryBarrier2 minMaxPreComputeBarrier = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, l, 1);
        pipelineBarrier(cmd, 0, 0, nullptr, 1, &minMaxPreComputeBarrier);
    }

    createImage(output.volumeImage, context.getDevice(), context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
        srcExtent.width, srcExtent.height, srcExtent.depth, 1,
        VK_FORMAT_R8_UINT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(), volume.volume_data.size(),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, volume.volume_data.data(), volume.volume_data.size());
    uploadVolumeData(cmd, output.volumeImage.image, stagingBuffer, srcExtent);

    minMaxPass.recordLeafDispatch(cmd, output.volumeImage.imageView, output.minMaxImage.imageView, pushConstants);

    VkExtent3D extent = leafExtent;
    for (uint32_t l = 0; l < fullMipCount - 1; ++l)
    {
        VkExtent3D dstExtent = {
            (extent.width + 1) >> 1,
            (extent.height + 1) >> 1,
            (extent.depth + 1) >> 1
        };

        VkImageMemoryBarrier2 srcRead = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT, l, 1);
        pipelineBarrier(cmd, 0, 0, nullptr, 1, &srcRead);

        VkImageMemoryBarrier2 dstWrite = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT, l + 1, 1);
        pipelineBarrier(cmd, 0, 0, nullptr, 1, &dstWrite);

        minMaxPass.recordOctreeDispatch(cmd, output.minMaxMipViews[l], extent, output.minMaxMipViews[l + 1], dstExtent);
        extent = dstExtent;
    }

    for (uint32_t l = 0; l < fullMipCount; ++l)
    {
        VkImageMemoryBarrier2 minMaxReadBarrier = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, l, 1);
        pipelineBarrier(cmd, 0, 0, nullptr, 1, &minMaxReadBarrier);
    }

    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    VK_CHECK(vkDeviceWaitIdle(context.getDevice()));
    std::cout << "MinMax Compute passes finished." << std::endl;
    VK_CHECK(vkDeviceWaitIdle(context.getDevice()));

    destroyBuffer(countReadbackBuffer, context.getDevice());
    destroyBuffer(stagingBuffer, context.getDevice());

    return output;
}

MinMaxOutput computeStreamingMinMaxMip(VulkanContext& context,
    VkImageView volumeAtlasView,
    VkSampler volumeSampler,
    const Buffer& pageTableBuffer,
    const PageCoord& pageCoord,
    PushConstants& pushConstants)
{
    std::string minMaxLeafShaderPath = "/spirv/minMaxLeaf.comp.spv";
    std::string minMaxOctreeReduceShaderPath = "/spirv/minMaxOctreeReduce.comp.spv";
    std::string streamingMinMaxLeafShaderPath = "/spirv/streamingMinMax.comp.spv";
    std::string streamingMinMaxOctreeReduceShaderPath = "/spirv/streamingOctreeReduce.comp.spv";
    MinMaxPass minMaxPass(context,
        minMaxLeafShaderPath.c_str(),
        minMaxOctreeReduceShaderPath.c_str(),
        streamingMinMaxLeafShaderPath.c_str(),
        streamingMinMaxOctreeReduceShaderPath.c_str());

    MinMaxOutput output{};
    VkExtent3D pageExtent = { pushConstants.volumeDim.x, pushConstants.volumeDim.y, pushConstants.volumeDim.z };
    VkExtent3D leafExtent = { pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z };
    uint32_t maxDim = std::max({ leafExtent.width, leafExtent.height, leafExtent.depth });
    uint32_t fullMipCount = 1u + static_cast<uint32_t>(std::floor(std::log2(maxDim)));
    output.minMaxMipViews.reserve(fullMipCount);

    createImage(output.minMaxImage, context.getDevice(), context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
        leafExtent.width, leafExtent.height, leafExtent.depth, fullMipCount,
        VK_FORMAT_R32G32_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    for (uint32_t l = 0; l < fullMipCount; ++l) {
        output.minMaxMipViews.push_back(createImageView(context.getDevice(), output.minMaxImage.image,
            VK_FORMAT_R32G32_UINT, VK_IMAGE_TYPE_3D, l, 1));
    }

    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());

    for (uint32_t l = 0; l < fullMipCount; ++l) {
        VkImageMemoryBarrier2 barrier = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, l, 1
        );
        pipelineBarrier(cmd, 0, 0, nullptr, 1, &barrier);
    }

    // Create proper streaming push constants
    StreamingMinMaxPushConstants streamingPC = {};
    streamingPC.pageCoord = glm::uvec3(pageCoord.x, pageCoord.y, pageCoord.z);
    streamingPC.mipLevel = 0;
    streamingPC.isoValue = pushConstants.isovalue;
    streamingPC.blockSize = pushConstants.blockDim.x; // Should be 4
    // Calculate page dimensions from blockGridDim - this assumes the caller set it correctly
    streamingPC.pageSizeX = pushConstants.blockGridDim.x * pushConstants.blockDim.x;
    streamingPC.pageSizeY = pushConstants.blockGridDim.y * pushConstants.blockDim.y;
    streamingPC.pageSizeZ = pushConstants.blockGridDim.z * pushConstants.blockDim.z;
    // Volume dimensions are passed correctly in volumeDim
    streamingPC.volumeSizeX = pushConstants.volumeDim.x;
    streamingPC.volumeSizeY = pushConstants.volumeDim.y;
    streamingPC.volumeSizeZ = pushConstants.volumeDim.z;
    // Granularity is the same as page size for now
    streamingPC.granularityX = streamingPC.pageSizeX;
    streamingPC.granularityY = streamingPC.pageSizeY;
    streamingPC.granularityZ = streamingPC.pageSizeZ;
    streamingPC.pageOverlap = 0;
    
    minMaxPass.recordStreamingLeafDispatch(cmd, volumeAtlasView, volumeSampler, pageTableBuffer,
        output.minMaxMipViews[0], streamingPC, pageCoord);

    VkImageMemoryBarrier2 leafToOctreeBarrier = imageBarrier(
        output.minMaxImage.image,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1
    );
    pipelineBarrier(cmd, 0, 0, nullptr, 1, &leafToOctreeBarrier);

    for (uint32_t l = 1; l < fullMipCount; ++l) {
        auto mipExtent = [](VkExtent3D e, uint32_t level) {
            return VkExtent3D{
                std::max(1u, e.width >> level),
                std::max(1u, e.height >> level),
                std::max(1u, e.depth >> level)
            };
        };
        VkExtent3D srcExtent = mipExtent(leafExtent, l - 1);
        VkExtent3D dstExtent = mipExtent(leafExtent, l);
        minMaxPass.recordStreamingOctreeDispatch(cmd,
            output.minMaxMipViews[l - 1], srcExtent,
            output.minMaxMipViews[l], dstExtent,
            pageTableBuffer, pageCoord);

        if (l < fullMipCount - 1) {
            VkImageMemoryBarrier2 levelBarrier = imageBarrier(
                output.minMaxImage.image,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, l, 1
            );
            pipelineBarrier(cmd, 0, 0, nullptr, 1, &levelBarrier);
        }
    }

    VkImageMemoryBarrier2 finalBarrier = imageBarrier(
        output.minMaxImage.image,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, fullMipCount
    );
    pipelineBarrier(cmd, 0, 0, nullptr, 1, &finalBarrier);

    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);

    return output;
}
