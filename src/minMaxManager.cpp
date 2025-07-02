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
#include <cstring>
#include <iostream>
#include <string>

void uploadVolumeData(VkCommandBuffer commandBuffer,
                       VkImage volumeImage, Buffer stagingBuffer, VkExtent3D extent) {
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
    pipelineBarrier(
        commandBuffer,
        {},
        0,
        {},
        1,
        &preCopyBarrier
    );
    // copy1DBufferTo3DImage should copy from stagingBuffer.buffer to volumeImage
    copy1DBufferTo3DImage(stagingBuffer, commandBuffer, volumeImage, extent.width, extent.height, extent.depth);
    // Transition to GENERAL for shader access (read/write)
    VkImageMemoryBarrier2 postCopyPreComputeBarrier = imageBarrier(
        volumeImage,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
        0, 1
    );

    pipelineBarrier(
        commandBuffer,
        {},
        0,
        {},
        1,
        &postCopyPreComputeBarrier
    );
}


// --- Modified Main Orchestrating Function ---
// Returns a struct containing handles to persistent resources
MinMaxOutput computeMinMaxMip(VulkanContext &context, Volume volume, PushConstants& pushConstants)
{
    // --- Create Pass Objects ---
    std::string minMaxLeafShaderPath = "/spirv/minMaxLeaf.comp.spv";
    std::string minMaxOctreeReduceShaderPath = "/spirv/minMaxOctreeReduce.comp.spv";
    
    MinMaxPass minMaxPass(context, minMaxLeafShaderPath.c_str(), minMaxOctreeReduceShaderPath.c_str());

    // --- Prepare Resources (some persistent, some temporary) ---
    MinMaxOutput output{}; // Create the output struct to hold persistent resources
    Buffer stagingBuffer = {}; // Temporary for upload
    Buffer countReadbackBuffer = {}; // Temporary for readback

    // Create persistent resources directly in the output struct
    // (Assumes Image/Buffer default constructors initialize handles to VK_NULL_HANDLE or similar)

    // Create MinMax output image (persistent)
    auto mipExtent = [](VkExtent3D e, uint32_t level) {
        return VkExtent3D {
            std::max(1u, e.width  >> level),
            std::max(1u, e.height >> level),
            std::max(1u, e.depth  >> level)
        };
    };
    VkExtent3D leafExtent = {pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z};
    uint32_t maxDim = std::max({ leafExtent.width, leafExtent.height, leafExtent.depth });
    uint32_t fullMipCount = 1u + static_cast<uint32_t>(std::floor(std::log2(maxDim)));
    
    VkExtent3D srcExtent = {volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z};

    createImage(output.minMaxImage, context.getDevice(), context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
                leafExtent.width, leafExtent.height, leafExtent.depth, fullMipCount,
                VK_FORMAT_R32G32_UINT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // --- Record Command Buffer ---
    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());

    std::vector<VkImageView> mipView(fullMipCount);
    for (uint32_t l = 0; l < fullMipCount; ++l) {
        mipView[l] = createImageView(context.getDevice(), output.minMaxImage.image, VK_FORMAT_R32G32_UINT, VK_IMAGE_TYPE_3D, l, 1);
        // 2. Barrier: Prepare MinMax Image for write
        VkImageMemoryBarrier2 minMaxPreComputeBarrier = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, l, 1);
        pipelineBarrier(cmd, {}, 0, {}, 1, &minMaxPreComputeBarrier);
    }
    output.minMaxMipViews = mipView;

    VkImageView fullMipView = createImageView(context.getDevice(),
                                          output.minMaxImage.image,
                                          VK_FORMAT_R32G32_UINT,
                                          VK_IMAGE_TYPE_3D,
                                          0,                 // baseMip
                                          fullMipCount);     // levelCount = ALL
    output.minMaxFull = fullMipView;
    // 1. Upload Volume Data (creates output.volumeImage and temporary stagingBuffer)
    //    Pass output.volumeImage by reference to be populated.
    createImage(output.volumeImage, context.getDevice(), context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
                    srcExtent.width, srcExtent.height, srcExtent.depth, 1,
                    VK_FORMAT_R8_UINT, // Assuming uint8 input volume data
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT); // Usage for upload and shader read

    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(), volume.volume_data.size(),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, volume.volume_data.data(), volume.volume_data.size());

    uploadVolumeData(cmd, output.volumeImage.image, stagingBuffer, srcExtent);


    // 3. Run MinMax Pass
    minMaxPass.recordLeafDispatch(cmd, output.volumeImage.imageView, output.minMaxImage.imageView, pushConstants);

    /* ---- reduction passes (mip 0 → 1, 1 → 2, …) ------------------- */
    VkExtent3D extent = leafExtent;
    for (uint32_t l = 0; l < fullMipCount - 1; ++l)
    {
        VkExtent3D dstExtent = {
            (extent.width  + 1) >> 1,   //  ceil(src / 2)
            (extent.height + 1) >> 1,
            (extent.depth  + 1) >> 1
        };

        /* 1. make level-l readable ----------------------------------- */
        VkImageMemoryBarrier2 srcRead = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  /* srcStage  */
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,    /* srcAccess */
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  /* dstStage  */
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,     /* dstAccess */
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT, l, 1);
        pipelineBarrier(cmd, {}, 0, {}, 1, &srcRead);

        /* 2. make level-(l+1) writable ------------------------------- */
        VkImageMemoryBarrier2 dstWrite = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_NONE,                        /* was: nothing */
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT, l + 1, 1);
        pipelineBarrier(cmd, {}, 0, {}, 1, &dstWrite);

        minMaxPass.recordOctreeDispatch(cmd, mipView[l], extent, mipView[l+1], dstExtent);
        extent = dstExtent;
    }

    for (uint32_t l = 0; l < fullMipCount; ++l)
    {
        VkImageMemoryBarrier2 minMaxReadBarrier = imageBarrier(
            output.minMaxImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, l, 1);

        pipelineBarrier(cmd, {}, 0, {}, 1, &minMaxReadBarrier);
    }

    // --- End and Submit Command Buffer ---
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    VK_CHECK(vkDeviceWaitIdle(context.getDevice()));
    std::cout << "MinMax Compute passes finished." << std::endl;

    VK_CHECK(vkDeviceWaitIdle(context.getDevice()));

    // --- Cleanup Only Temporary Resources ---
    destroyBuffer(countReadbackBuffer, context.getDevice());
    destroyBuffer(stagingBuffer, context.getDevice());

    return output;
}