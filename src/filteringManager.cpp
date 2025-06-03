#include "filteringManager.h"
#include "common.h"

#include "vulkan_context.h"
#include "resources.h"
#include "buffer.h"
#include "image.h"
#include "vulkan_utils.h"
#include "minMaxPass.h"
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
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
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
FilteringOutput filterActiveBlocks(VulkanContext &context, Volume volume, PushConstants& pushConstants)
{
    // --- Create Pass Objects ---
    std::string minMaxShaderPath = "/spirv/computeMinMax.comp.spv";
    std::string filterShaderPath = "/spirv/occupiedBlockPrefixSum.comp.spv";
    MinMaxPass minMaxPass(context, minMaxShaderPath.c_str());
    ActiveBlockFilteringPass filteringPass(context, filterShaderPath.c_str());

    // --- Prepare Resources (some persistent, some temporary) ---
    FilteringOutput output{}; // Create the output struct to hold persistent resources
    Buffer stagingBuffer = {}; // Temporary for upload
    Buffer countReadbackBuffer = {}; // Temporary for readback

    // Create persistent resources directly in the output struct
    // (Assumes Image/Buffer default constructors initialize handles to VK_NULL_HANDLE or similar)

    // Create MinMax output image (persistent)
    VkExtent3D gridExtent = {pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z};
    VkExtent3D volumeExtent = {volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z};

    createImage(output.minMaxImage, context.getDevice(), context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
                gridExtent.width, gridExtent.height, gridExtent.depth, 1,
                VK_FORMAT_R32G32_UINT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    // Create Filtering output buffers (persistent)
    uint32_t totalBlocks = pushConstants.blockGridDim.x * pushConstants.blockGridDim.y * pushConstants.blockGridDim.z;
    VkDeviceSize compactedBufferSize = totalBlocks * sizeof(uint32_t);
    createBuffer(output.compactedBlockIdBuffer, context.getDevice(), context.getMemoryProperties(),
                 compactedBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDeviceSize countBufferSize = sizeof(uint32_t);
    createBuffer(output.activeBlockCountBuffer, context.getDevice(), context.getMemoryProperties(),
                 countBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // --- Record Command Buffer ---
    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());

    // 1. Upload Volume Data (creates output.volumeImage and temporary stagingBuffer)
    //    Pass output.volumeImage by reference to be populated.
    createImage(output.volumeImage, context.getDevice(), context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
                    volumeExtent.width, volumeExtent.height, volumeExtent.depth, 1,
                    VK_FORMAT_R8_UINT, // Assuming uint8 input volume data
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT); // Usage for upload and shader read

    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(), volume.volume_data.size(),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, volume.volume_data.data(), volume.volume_data.size());

    uploadVolumeData(cmd, output.volumeImage.image, stagingBuffer, volumeExtent);

    // 2. Barrier: Prepare MinMax Image for write
    VkImageMemoryBarrier2 minMaxPreComputeBarrier = imageBarrier(
        output.minMaxImage.image,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    pipelineBarrier(cmd, {}, 0, {}, 1, &minMaxPreComputeBarrier);

    // 3. Run MinMax Pass
    minMaxPass.recordDispatch(cmd, output.volumeImage.imageView, output.minMaxImage.imageView, pushConstants);

    // 4. Barriers: Transition MinMax Image (W->R), Initialize Count Buffer (Fill->RW), Prepare Compact ID Buffer (None->W)
    VkImageMemoryBarrier2 minMaxReadBarrier = imageBarrier(
        output.minMaxImage.image,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    vkCmdFillBuffer(cmd, output.activeBlockCountBuffer.buffer, 0, countBufferSize, 0);

    VkBufferMemoryBarrier2 bufferTransferToComputeBarriers[2] = {};
    bufferTransferToComputeBarriers[0] = bufferBarrier(
       output.activeBlockCountBuffer.buffer,
       VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
       0, VK_WHOLE_SIZE);
    bufferTransferToComputeBarriers[1] = bufferBarrier(
       output.compactedBlockIdBuffer.buffer,
       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
       0, VK_WHOLE_SIZE);

    pipelineBarrier(cmd, {}, 2, bufferTransferToComputeBarriers, 1, &minMaxReadBarrier); // Combined barrier

    // 5. Run Active Block Filtering Pass
    filteringPass.recordDispatch(cmd, output.minMaxImage.imageView, output.compactedBlockIdBuffer, output.activeBlockCountBuffer, pushConstants);

    // 6. Barrier: Ensure compute writes are finished before copy/readback
    VkBufferMemoryBarrier2 bufferComputeToTransferBarriers[2] = {};
    bufferComputeToTransferBarriers[0] = bufferBarrier(
        output.activeBlockCountBuffer.buffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        0, VK_WHOLE_SIZE);
    bufferComputeToTransferBarriers[1] = bufferBarrier(
        output.compactedBlockIdBuffer.buffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, // Prepare for potential use/copy later
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 2, bufferComputeToTransferBarriers, 0, {});

    // --- End and Submit Command Buffer ---
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    VK_CHECK(vkDeviceWaitIdle(context.getDevice()));
    std::cout << "Compute passes finished." << std::endl;

    // --- Read back the active block count (into output struct) ---
    createBuffer(countReadbackBuffer, context.getDevice(), context.getMemoryProperties(), countBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBuffer copyCmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    VkBufferCopy region = {0, 0, countBufferSize};
    vkCmdCopyBuffer(copyCmd, output.activeBlockCountBuffer.buffer, countReadbackBuffer.buffer, 1, &region);
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), copyCmd);
    VK_CHECK(vkDeviceWaitIdle(context.getDevice()));
    memcpy(&output.activeBlockCount, countReadbackBuffer.data, sizeof(uint32_t));
    std::cout << "Occupied Block Filtering finished. Active blocks found: " << output.activeBlockCount << std::endl;

    // (Optional) Test the compacted buffer content using the handle from the output struct
    // testCompactBuffer(context, output.compactedBlockIdBuffer, output.activeBlockCount);

    // --- Cleanup Only Temporary Resources ---
    destroyBuffer(countReadbackBuffer, context.getDevice());
    destroyBuffer(stagingBuffer, context.getDevice());

    // The Pass objects (minMaxPass, filteringPass) will be destroyed automatically
    // when they go out of scope, cleaning up their internal pipelines/layouts.
    // The VulkanContext object will be destroyed when it goes out of scope.
    // --- Return the Output Struct ---
    // The ownership of resources within 'output' is transferred to the caller.
    // Move semantics will be used if FilteringOutput has a move constructor.
    return output;
}