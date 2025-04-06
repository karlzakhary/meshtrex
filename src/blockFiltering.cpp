#include "common.h"
#include "blockFiltering.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <iostream>

#include "buffer.h"
#include "device.h"
#include "mesh.h"
#include "resources.h"
#include "shaders.h"
#include "vulkan_utils.h"
#include "volume.h"

struct alignas(16) PushConstants {
    glm::uvec3 volumeDim;
    glm::uvec3 blockDim;
    glm::uvec3 blockGridDim;
};

std::tuple<VkPipelineLayout, VkDescriptorSetLayout> createComputeMinMaxPipelineLayout(
    VkDevice device)
{
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants)
    };

    VkDescriptorSetLayoutBinding bindings[2] = {};

    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);

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

// Helper to insert an image layout transition barrier
void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo depInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier
    };

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

// Helper to upload a 3D volume image using a staging buffer
void uploadVolumeImage(VkCommandBuffer commandBuffer,
                       VkImage volumeImage, Buffer stagingBuffer, VkExtent3D extent) {
    transitionImage(commandBuffer, volumeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    copy1DBufferTo3DImage(stagingBuffer, commandBuffer, volumeImage, extent.width, extent.height, extent.depth);

    transitionImage(commandBuffer, volumeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
}


std::vector<MinMaxResult> mapMinMaxBuffer(VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
                                             VkCommandPool commandPool, VkQueue queue,
                                             const Buffer& minMaxBuffer, VkDeviceSize minMaxBufferSize,
                                             glm::ivec3 blockGridDim)
{
    std::cout << "\nReading back GPU results buffer..." << std::endl;

    // 1. Create a host-visible staging buffer for readback
    Buffer readbackBuffer = {};
    createBuffer(readbackBuffer, device, memoryProperties, minMaxBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, // Destination for copy
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // Host visible & coherent

    // 2. Copy data from GPU buffer to staging buffer
    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);

    VkBufferCopy region = {};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = minMaxBufferSize;
    vkCmdCopyBuffer(cmd, minMaxBuffer.buffer, readbackBuffer.buffer, 1, &region);

    endSingleTimeCommands(device, commandPool, queue, cmd); // Submits and waits

    // 3. Access the *already mapped* pointer
    void* mappedData = readbackBuffer.data; // Use the pointer stored by createBuffer
    if (mappedData == nullptr) {
        // Handle error: buffer wasn't host visible or mapping failed in createBuffer
        destroyBuffer(readbackBuffer, device);
        throw std::runtime_error("Readback buffer is not mapped (or mapped pointer not stored)!");
    }

    // Calculate the number of MinMaxResult elements expected
    // 4. Copy data from mapped buffer to CPU vector
    size_t numElements = minMaxBufferSize / sizeof(MinMaxResult);
    if (minMaxBufferSize % sizeof(MinMaxResult) != 0) {
         std::cerr << "Warning: Buffer size (" << minMaxBufferSize
                   << ") is not a multiple of MinMaxResult size (" << sizeof(MinMaxResult) << ")" << std::endl;
         // Adjust numElements or handle potential partial data if necessary
         numElements = minMaxBufferSize / sizeof(MinMaxResult); // Integer division truncates
    }

    // Verify calculated count matches grid dimensions
    size_t expectedCount = static_cast<size_t>(blockGridDim.x) * blockGridDim.y * blockGridDim.z;
     if (numElements != expectedCount) {
          std::cerr << "Warning: Number of elements in buffer (" << numElements
                   << ") does not match expected block count (" << expectedCount << ")" << std::endl;
          // Decide how to proceed - maybe use the smaller count?
          numElements = std::min(numElements, expectedCount);
     }

    std::vector<MinMaxResult> results(numElements);
    memcpy(results.data(), mappedData, numElements * sizeof(MinMaxResult));

    // Clean
    vkUnmapMemory(device, readbackBuffer.memory);
    destroyBuffer(readbackBuffer, device);
    std::cout << "GPU results readback complete." << std::endl;

    // 6. Print some results (optional)
    uint32_t totalBlocks = numElements; // Use actual number of elements copied
    std::cout << "GPU Results (first few blocks):" << std::endl;
    for (uint32_t i = 0; i < std::min(totalBlocks, 10u); ++i) { // Print fewer results
         // Cast uint8_t to int for correct printing
        std::cout << "Block[" << i << "] Min: " << static_cast<int>(results[i].minVal)
                  << " Max: " << static_cast<int>(results[i].maxVal) << std::endl;
    }
     if (totalBlocks > 10u) {
        std::cout << "... (" << totalBlocks << " total blocks read)" << std::endl;
     }

    // 8. Return the vector by value
    return results;
}

// Helper to read back a 3D volume image to CPU from GPU memory
// Requires the image to be in VK_IMAGE_LAYOUT_GENERAL and of VK_FORMAT_R8_UNORM
std::vector<uint8_t> downloadVolumeImage(
    VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
    VkCommandPool commandPool, VkQueue queue, VkImage volumeImage,
    VkExtent3D extent)
{
    VkDeviceSize volumeSize =
        extent.width * extent.height * extent.depth * sizeof(uint8_t);

    // Create destination buffer (host visible)
    Buffer readbackBuffer;
    createBuffer(readbackBuffer, device, memoryProperties, volumeSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Create command buffer to copy image to buffer
    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);

    // Transition image to TRANSFER_SRC layout
    transitionImage(cmd, volumeImage, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = extent.width;
    copyRegion.bufferImageHeight = extent.height;

    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;

    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = extent;

    vkCmdCopyImageToBuffer(cmd, volumeImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readbackBuffer.buffer, 1, &copyRegion);

    // Transition back to GENERAL for continued use
    transitionImage(cmd, volumeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_GENERAL);

    endSingleTimeCommands(device, commandPool, queue, cmd);

    // Copy to std::vector
    std::vector<uint8_t> result(volumeSize);
    memcpy(result.data(), readbackBuffer.data, volumeSize);

    destroyBuffer(readbackBuffer, device);
    return result;
}

void pushDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VkImageView imageView,
                        VkBuffer buffer)
{
    VkDescriptorImageInfo imageInfo = {.sampler = VK_NULL_HANDLE,
                                       .imageView = imageView,
                                       .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorBufferInfo bufferInfo = {
        .buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[2] = {};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &bufferInfo;

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout, 0, 2, writes);
}

std::vector<MinMaxResult> filterUnoccupiedBlocks(char **argv, const char *path)
{
    VK_CHECK(volkInitialize());
    std::string spath = argv[0];
    std::string::size_type pos = spath.find_last_of("/\\");
    if (pos == std::string::npos)
        spath = "";
    else
        spath = spath.substr(0, pos + 1);
    spath += path;

    Volume volume = loadVolume(spath.c_str());

    VkInstance instance = createInstance();
    assert(instance);

    volkLoadInstanceOnly(instance);

    VkDebugReportCallbackEXT debugCallback = registerDebugCallback(instance);

    VkPhysicalDevice physicalDevices[16];
    uint32_t physicalDeviceCount = std::size(physicalDevices);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                        physicalDevices));

    VkPhysicalDevice physicalDevice =
        pickPhysicalDevice(physicalDevices, physicalDeviceCount);
    assert(physicalDevice);

    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0,
                                                  &extensionCount, 0));

    std::vector<VkExtensionProperties> extensions(extensionCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &extensionCount, extensions.data()));

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    assert(props.limits.timestampComputeAndGraphics);

    uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevice);
    assert(familyIndex != VK_QUEUE_FAMILY_IGNORED);

    VkDevice device =
        createDevice(instance, physicalDevice, familyIndex, false);
    assert(device);

    volkLoadDevice(device);

    vkCmdBeginRendering = vkCmdBeginRenderingKHR;
    vkCmdEndRendering = vkCmdEndRenderingKHR;
    vkCmdPipelineBarrier2 = vkCmdPipelineBarrier2KHR;

    VkQueue queue = nullptr;
    vkGetDeviceQueue(device, familyIndex, 0, &queue);

    Shader minMaxCS{};
    assert(loadShader(minMaxCS, device, argv[0], "spirv/computeMinMax.comp.spv"));


    VkPipelineCache pipelineCache = nullptr;

    auto [pipelineLayout, setLayout] = createComputeMinMaxPipelineLayout(device);

    VkPipeline computeMinMaxPipeline = createComputePipeline(
        device,
        pipelineCache,
        minMaxCS,
        pipelineLayout
    );

    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    VkCommandPool commandPool = createCommandPool(device, familyIndex);
    assert(commandPool);

    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
    assert(commandBuffer);
    VkExtent3D extent = {volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z};
    VkDeviceSize bufferSize = volume.volume_data.size();

    Buffer stagingBuffer = {};
    createBuffer(
        stagingBuffer,
        device,
        memoryProperties,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    memcpy(stagingBuffer.data, volume.volume_data.data(), volume.volume_data.size());

    Image volImage = {};

    createImage(volImage,
        device,
        memoryProperties,
        VK_IMAGE_TYPE_3D,
        volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z, 1,
        VK_FORMAT_R8_UINT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    uploadVolumeImage(commandBuffer, volImage.image, stagingBuffer, extent);

    PushConstants pushConstants = {};
    pushConstants.volumeDim = volume.volume_dims;
    pushConstants.blockDim = glm::uvec3(8, 8, 8);
    pushConstants.blockGridDim = (pushConstants.volumeDim + pushConstants.blockDim - 1u) / pushConstants.blockDim;

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);

    // Create output buffer for compute shader results
    uint totalBlocks = pushConstants.blockGridDim.x * pushConstants.blockGridDim.y * pushConstants.blockGridDim.z;
    VkDeviceSize minMaxSize = totalBlocks * sizeof(MinMaxResult);
    Buffer minMaxBuffer = {};
    createBuffer(minMaxBuffer, device, memoryProperties,
                 minMaxSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeMinMaxPipeline);

    pushDescriptorSets(commandBuffer, pipelineLayout, volImage.imageView, minMaxBuffer.buffer);

    vkCmdDispatch(commandBuffer, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);

    endSingleTimeCommands(device, commandPool, queue, commandBuffer);

    VK_CHECK(vkResetCommandPool(device, commandPool, 0));

    VK_CHECK(vkDeviceWaitIdle(device));

    // vkDestroyCommandPool(device, commandPool, nullptr);
    //
    // vkDestroyPipeline(device, computeMinMaxPipeline, nullptr);
    // vkDestroyShaderModule(device, minMaxCS.module, nullptr);
    //
    // vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    // vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    //
    // destroyImage(volImage, device);
    // destroyBuffer(stagingBuffer, device);
    // destroyBuffer(minMaxBuffer, device);
    //
    // vkDestroyDevice(device, nullptr);
    //
    // if (debugCallback)
    //     vkDestroyDebugReportCallbackEXT(instance, debugCallback, 0);
    //
    // vkDestroyInstance(instance, nullptr);
    //
    // volkFinalize();
    return mapMinMaxBuffer(device, memoryProperties, commandPool, queue, minMaxBuffer, minMaxSize, pushConstants.blockGridDim);
}