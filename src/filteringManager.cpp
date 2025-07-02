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


// --- Modified Main Orchestrating Function ---
// Returns a struct containing handles to persistent resources
FilteringOutput filterActiveBlocks(VulkanContext &context, MinMaxOutput &minMaxOutput, PushConstants& pushConstants)
{
    // --- Create Pass Objects ---
    std::string filterShaderPath = "/spirv/occupiedBlockPrefixSum.comp.spv";
    ActiveBlockFilteringPass filteringPass(context, filterShaderPath.c_str());

    // --- Prepare Resources (some persistent, some temporary) ---
    FilteringOutput output{}; // Create the output struct to hold persistent resources
    Buffer stagingBuffer = {}; // Temporary for upload
    Buffer countReadbackBuffer = {}; // Temporary for readback

    // Create persistent resources directly in the output struct
    // (Assumes Image/Buffer default constructors initialize handles to VK_NULL_HANDLE or similar)

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

    pipelineBarrier(cmd, {}, 2, bufferTransferToComputeBarriers, 0, {}); // Combined barrier
    VkSampler sampler;
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.minLod = 0;
    sci.maxLod = minMaxOutput.minMaxMipViews.size();
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.anisotropyEnable = VK_FALSE;
    vkCreateSampler(context.getDevice(), &sci, nullptr, &sampler);
    // 5. Run Active Block Filtering Pass
    filteringPass.recordDispatch(cmd, minMaxOutput.minMaxImage.imageView, sampler, output.compactedBlockIdBuffer, output.activeBlockCountBuffer, pushConstants);

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
    testCompactBuffer(context, output.compactedBlockIdBuffer, output.activeBlockCount);

    // --- Cleanup Only Temporary Resources ---
    destroyBuffer(countReadbackBuffer, context.getDevice());
    destroyBuffer(stagingBuffer, context.getDevice());
    vkDestroySampler(context.getDevice(), sampler, nullptr);
    // The Pass objects (minMaxPass, filteringPass) will be destroyed automatically
    // when they go out of scope, cleaning up their internal pipelines/layouts.
    // The VulkanContext object will be destroyed when it goes out of scope.
    // --- Return the Output Struct ---
    // The ownership of resources within 'output' is transferred to the caller.
    // Move semantics will be used if FilteringOutput has a move constructor.
    return output;
}