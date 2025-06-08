#include "common.h"
#include "blockFiltering.h"

#include <cstring>
#include <fstream>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <iostream>
#include <vector>

#include "buffer.h"
#include "mesh.h"
#include "resources.h"
#include "shaders.h"
#include "vulkan_utils.h"
#include "volume.h"
#include "image.h"
#include "vulkan_context.h"
#include "blockFilteringTestUtils.h"

std::tuple<VkPipelineLayout, VkDescriptorSetLayout> createComputeMinMaxPipelineLayout(
    VkDevice device)
{
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants)
    };

    VkDescriptorSetLayoutBinding bindings[2] = {};

    // Binding 0: Input Volume Image (Read-Only in shader)
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, // Shader reads via imageLoad
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    // Binding 1: Output MinMax Image (Write-Only in shader)
    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = 2,
        .pBindings = bindings
    };

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1,
    pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout layout = nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &layout));

    return std::make_tuple(layout, descriptorSetLayout);
}

std::tuple<VkPipelineLayout, VkDescriptorSetLayout> createComputeOccupiedBlockFilteringPipelineLayout(
    VkDevice device)
{
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; // MinMax Input
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Compacted IDs Output
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Atomic Counter Output
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

    VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    return std::make_tuple(pipelineLayout, descriptorSetLayout);
}


void pushMinMaxDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
                        VkImageView inputVolumeView, // Input Volume Image
                        VkImageView outputMinMaxView) // Output MinMax Image
{
    VkDescriptorImageInfo inputImageInfo = {};
    inputImageInfo.sampler = VK_NULL_HANDLE; // Not used for storage image
    inputImageInfo.imageView = inputVolumeView;
    inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // Shader reads via imageLoad

    VkDescriptorImageInfo outputImageInfo = {};
    outputImageInfo.sampler = VK_NULL_HANDLE; // Not used for storage image
    outputImageInfo.imageView = outputMinMaxView;
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // Shader writes via imageStore

    VkWriteDescriptorSet writes[2] = {};

    // Binding 0: Input Volume Image
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &inputImageInfo;

    // Binding 1: Output MinMax Image
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &outputImageInfo;

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout, 0, 2, writes);
}

void pushOccupiedBlockFilteringDescriptorSets(
    VkCommandBuffer cmd,
    VkImageView minMaxImage,
    const PushConstants& pushData,   // Contains dimensions, isovalue etc.
    Buffer& compactedBlockIdBuffer,  // Output: Buffer to store active block IDs
    Buffer& activeBlockCountBuffer,
    VkPipelineLayout pipelineLayout)
{
    VkDescriptorImageInfo minMaxImageInfo = {};
    minMaxImageInfo.imageView = minMaxImage;
    minMaxImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // Must match layout from previous pass

    VkDescriptorBufferInfo compactedIdBufferInfo = {compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo countBufferInfo = {activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &minMaxImageInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &compactedIdBufferInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &countBufferInfo;

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 3, writes);

    // Push Constants (including isovalue)
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushData);
}

// --- Function to set up and run the filtering pass ---
// Returns the total count of active blocks found.
uint32_t runOccupiedBlockFiltering(
    char** argv,
    VulkanContext &context,
    const Image& minMaxImage,        // Input: Result from the previous MinMax pass
    const PushConstants& pushData,   // Contains dimensions, isovalue etc.
    Buffer& compactedBlockIdBuffer,  // Output: Buffer to store active block IDs
    Buffer& activeBlockCountBuffer) // Output: Buffer to store the count
{
    std::cout << "\nSetting up Occupied Block Filtering pass..." << std::endl;

    // --- Create Output Buffers ---
    uint32_t totalBlocks = pushData.blockGridDim.x * pushData.blockGridDim.y * pushData.blockGridDim.z;

    // 1. Compacted Block ID Buffer (worst case size: all blocks active)
    VkDeviceSize compactedBufferSize = totalBlocks * sizeof(uint32_t); // Store uint IDs
    createBuffer(compactedBlockIdBuffer, context.getDevice(), context.getMemoryProperties(),
                 compactedBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // Allow clearing & readback
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // 2. Active Block Count Buffer (single uint)
    VkDeviceSize countBufferSize = sizeof(uint32_t);
    createBuffer(activeBlockCountBuffer, context.getDevice(), context.getMemoryProperties(),
                 countBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // Allow clearing & readback
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); // Can be device local initially

    // --- Initialize Active Block Count Buffer to 0 ---
    VkCommandBuffer initCmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    vkCmdFillBuffer(initCmd, activeBlockCountBuffer.buffer, 0, countBufferSize, 0);
    // Add barrier if needed before compute shader uses it
     VkBufferMemoryBarrier2 countBufBarrier = bufferBarrier(
        activeBlockCountBuffer.buffer, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        0, VK_WHOLE_SIZE
     );

    VkBufferMemoryBarrier2 compactBlockIdBufBarrier = bufferBarrier(
   compactedBlockIdBuffer.buffer, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
   0, VK_WHOLE_SIZE
);

    pipelineBarrier(initCmd, {}, 1, &countBufBarrier, 0, {});
    pipelineBarrier(initCmd, {}, 1, &compactBlockIdBufBarrier, 0, {});
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), initCmd);
    std::cout << "Output buffers created and count initialized." << std::endl;
    auto [pipelineLayout, setLayout] = createComputeOccupiedBlockFilteringPipelineLayout(context.getDevice());

    // --- Load Shader & Create Pipeline ---
    Shader filterCS{};
    assert(loadShader(filterCS, context.getDevice(), "spirv/occupiedBlockPrefixSum.comp.spv"));
    VkPipeline computePipeline = createComputePipeline(context.getDevice(), nullptr, filterCS, pipelineLayout);
    std::cout << "Filtering pipeline created." << std::endl;


    // --- Dispatch ---
    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

    pushOccupiedBlockFilteringDescriptorSets(
        cmd,
        minMaxImage.imageView,
        pushData,
        compactedBlockIdBuffer,
        activeBlockCountBuffer,
        pipelineLayout
    );
    // Calculate dispatch size (1D)
    uint32_t localSizeX = 128; // Must match shader's local_size_x
    uint32_t groupCountX = (totalBlocks + localSizeX - 1) / localSizeX;
    std::cout << "Dispatching " << groupCountX << " workgroups (" << totalBlocks << " total blocks)..." << std::endl;
    vkCmdDispatch(cmd, groupCountX, 1, 1);

    // Barrier: Ensure shader writes (especially to count buffer) are finished before potential readback
    VkBufferMemoryBarrier2 countReadBarrier = {};
    countReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    countReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    countReadBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT; // Shader writes/atomic ops
    countReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; // Prepare for potential copy/readback
    countReadBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    countReadBarrier.buffer = activeBlockCountBuffer.buffer;
    countReadBarrier.offset = 0;
    countReadBarrier.size = VK_WHOLE_SIZE;
    // Add similar barrier for compactedBlockIdBuffer if reading it back immediately

    VkDependencyInfo readDepInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    readDepInfo.bufferMemoryBarrierCount = 1;
    readDepInfo.pBufferMemoryBarriers = &countReadBarrier;
    vkCmdPipelineBarrier2(cmd, &readDepInfo);

    VkBufferMemoryBarrier2 compactBlockReadBarrier = {};
    compactBlockReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    compactBlockReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    compactBlockReadBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT; // Shader writes/atomic ops
    compactBlockReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; // Prepare for potential copy/readback
    compactBlockReadBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    compactBlockReadBarrier.buffer = compactedBlockIdBuffer.buffer;
    compactBlockReadBarrier.offset = 0;
    compactBlockReadBarrier.size = VK_WHOLE_SIZE;
    // Add similar barrier for compactedBlockIdBuffer if reading it back immediately

    VkDependencyInfo readDepCompactInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    readDepCompactInfo.bufferMemoryBarrierCount = 1;
    readDepCompactInfo.pBufferMemoryBarriers = &compactBlockReadBarrier;
    vkCmdPipelineBarrier2(cmd, &readDepCompactInfo);


    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    VK_CHECK(vkDeviceWaitIdle(context.getDevice())); // Ensure completion before reading count

    // --- Read back the count ---
    uint32_t activeCount = 0;
    // Create a host-visible buffer
    Buffer countReadbackBuffer = {};
    createBuffer(countReadbackBuffer, context.getDevice(), context.getMemoryProperties(), countBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Copy device buffer to host buffer
    cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    VkBufferCopy region = {0, 0, countBufferSize};
    vkCmdCopyBuffer(cmd, activeBlockCountBuffer.buffer, countReadbackBuffer.buffer, 1, &region);
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    VK_CHECK(vkDeviceWaitIdle(context.getDevice()));
    // Read from mapped pointer
    memcpy(&activeCount, countReadbackBuffer.data, sizeof(uint32_t));
    destroyBuffer(countReadbackBuffer, context.getDevice()); // Clean up staging buffer

    std::cout << "Occupied Block Filtering finished. Active blocks found: " << activeCount << std::endl;
    testCompactBuffer(context, compactedBlockIdBuffer, activeCount);
    // --- Cleanup ---
    // Destroy pipeline, layout, shader module, descriptor set layout
    vkDestroyPipeline(context.getDevice(), computePipeline, nullptr);
    vkDestroyPipelineLayout(context.getDevice(), pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.getDevice(), setLayout, nullptr);
    vkDestroyShaderModule(context.getDevice(), filterCS.module, nullptr);
    // Buffers (compactedBlockIdBuffer, activeBlockCountBuffer) are kept as output

    return activeCount;
}

void uploadVolumeImage(VkCommandBuffer commandBuffer,
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

uint32_t filterUnoccupiedBlocks(char **argv, const char *path)
{
    VulkanContext context(false);
    VkDevice device = context.getDevice();
    std::cout << "Successfully obtained Vulkan device: " << device << std::endl;

    VkPipelineCache pipelineCache = nullptr;

    // Push constants
    Volume volume = loadVolume(path);
    PushConstants pushConstants = {};
    pushConstants.volumeDim    = glm::uvec4(volume.volume_dims, 1); // Pad W with 1
    pushConstants.blockDim     = glm::uvec4(8, 8, 8, 1);      // Pad W with 1
    pushConstants.blockGridDim = glm::uvec4(glm::uvec3((pushConstants.volumeDim + pushConstants.blockDim - 1u) / pushConstants.blockDim), 1);       // Pad W with 1
    pushConstants.isovalue     = 80;

    // --- Prepare Buffers and Images ---
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, context.getCommandPool());

    Shader minMaxCS{};
    assert(loadShader(minMaxCS, device, "spirv/computeMinMax.comp.spv"));

    auto [pipelineLayout, setLayout] = createComputeMinMaxPipelineLayout(device);

    VkPipeline computeMinMaxPipeline = createComputePipeline(device, pipelineCache, minMaxCS, pipelineLayout);
    // --- End Pipeline Creation ---

    // Push constants
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);

    // Staging buffer for input volume upload
    VkDeviceSize volumeBufferSize = volume.volume_data.size();
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device, context.getMemoryProperties(), volumeBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, volume.volume_data.data(), volume.volume_data.size());

    // Input volume image
    VkExtent3D volumeExtent = {volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z};
    Image volImage = {};
    createImage(volImage, device, context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
                volumeExtent.width, volumeExtent.height, volumeExtent.depth, 1,
                VK_FORMAT_R8_UINT, // Assuming uint8 input volume data
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT); // Usage for upload and shader read
    uploadVolumeImage(commandBuffer, volImage.image, stagingBuffer, volumeExtent);
    // Input volume image is now in VK_IMAGE_LAYOUT_GENERAL

    // --- Create output image for MinMax results ---
    VkExtent3D gridExtent = {pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z};
    VkDeviceSize minMaxTotalBytes = static_cast<size_t>(gridExtent.width) * gridExtent.height * gridExtent.depth * sizeof(MinMaxResult); // Size based on uint32_t struct
    Image minMaxImage = {}; // Assuming Image struct
    createImage(minMaxImage, device, context.getMemoryProperties(), VK_IMAGE_TYPE_3D,
                gridExtent.width, gridExtent.height, gridExtent.depth, 1,
                VK_FORMAT_R32G32_UINT, // Format to store uvec2 (min, max as uint32_t)
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT); // Usage for shader write & readback copy

    // Transition MinMax image layout for shader write
    VkImageMemoryBarrier2 minMaxPreComputeBarrier = imageBarrier(
        minMaxImage.image,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
        0, 1
    );

    pipelineBarrier(
        commandBuffer,
        {},
        0, {},
        1, &minMaxPreComputeBarrier
    );

    // --- Dispatch Compute Shader ---
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeMinMaxPipeline);

    // Push descriptors (using updated function)
    pushMinMaxDescriptorSets(commandBuffer, pipelineLayout,
                       volImage.imageView,    // Input volume image view
                       minMaxImage.imageView); // Output min/max image view

    // Dispatch
    vkCmdDispatch(commandBuffer, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);
    // --- End Dispatch ---

    // Barrier between min-max and occupied block filtering§
    VkImageMemoryBarrier2 minMaxReadBarrier = imageBarrier(
        minMaxImage.image,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1
    );

    pipelineBarrier(
        commandBuffer,
        {},
        0,
        {},
        1,
        &minMaxReadBarrier
    );

    // End and submit commands
    endSingleTimeCommands(device, context.getCommandPool(), context.getQueue(), commandBuffer);
    VK_CHECK(vkDeviceWaitIdle(device)); // Wait for compute to finish
    vkResetCommandPool(device, context.getCommandPool(), 0);

    Buffer compactedBlockIdBuffer = {};
    Buffer activeBlockCountBuffer = {};
    uint32_t blocks = runOccupiedBlockFiltering(
        argv,
        context,
        minMaxImage,
        pushConstants,
        compactedBlockIdBuffer,
        activeBlockCountBuffer
    );

    // --- Cleanup (Simplified - ensure all created resources are destroyed) ---
    VK_CHECK(vkResetCommandPool(device, context.getCommandPool(), 0)); // Reset pool before destroying

    vkDestroyPipeline(device, computeMinMaxPipeline, nullptr);
    vkDestroyShaderModule(device, minMaxCS.module, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);

    destroyImage(minMaxImage, device); // Destroy the new image
    destroyImage(volImage, device);
    destroyBuffer(stagingBuffer, device);
    destroyBuffer(compactedBlockIdBuffer, device);
    destroyBuffer(activeBlockCountBuffer, device);

    return blocks;
}
