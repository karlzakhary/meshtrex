#include "common.h"
#include "blockFiltering.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <iostream>
#include <vector>
#include <stdexcept>

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

void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
                     VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask) {
    VkImageMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
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

void uploadVolumeImage(VkCommandBuffer commandBuffer,
                       VkImage volumeImage, Buffer stagingBuffer, VkExtent3D extent) {
    transitionImage(commandBuffer, volumeImage,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
    );
    // copy1DBufferTo3DImage should copy from stagingBuffer.buffer to volumeImage
    copy1DBufferTo3DImage(stagingBuffer, commandBuffer, volumeImage, extent.width, extent.height, extent.depth);
    // Transition to GENERAL for shader access (read/write)
    transitionImage(commandBuffer, volumeImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );
}


// NEW Function: Reads back the MinMax data from a VkImage via a staging buffer
std::vector<MinMaxResult> mapMinMaxImage(VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
                                         VkCommandPool commandPool, VkQueue queue,
                                         const Image& minMaxImage,
                                         VkExtent3D gridExtent,
                                         VkDeviceSize minMaxTotalBytes)
{
    std::cout << "\nReading back GPU MinMax Image results..." << std::endl;

    // 1. Create a host-visible staging buffer for readback
    Buffer readbackBuffer = {};
    createBuffer(readbackBuffer, device, memoryProperties, minMaxTotalBytes,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, // Destination for image copy
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // 2. Copy data from GPU image to staging buffer
    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);

    // Transition image layout for copying
    transitionImage(cmd, minMaxImage.image,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

    // Setup copy command
    copy3DImageTo1DBuffer(readbackBuffer, cmd, minMaxImage.image, gridExtent);

    // Transition image layout back to GENERAL (or whatever is needed next)
    transitionImage(cmd, minMaxImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

    endSingleTimeCommands(device, commandPool, queue, cmd); // Submits and waits

    // 3. Access the mapped pointer of the readback buffer
    void* mappedData = readbackBuffer.data;
    if (mappedData == nullptr) {
        destroyBuffer(readbackBuffer, device);
        throw std::runtime_error("Readback buffer is not mapped!");
    }

    // 4. Copy data from mapped buffer to CPU vector
    size_t numElements = minMaxTotalBytes / sizeof(MinMaxResult);
    // ... (Add validation checks for size/count as before if desired) ...
    size_t expectedCount = static_cast<size_t>(gridExtent.width) * gridExtent.height * gridExtent.depth;
     if (numElements != expectedCount) {
          std::cerr << "Warning: Number of elements in buffer (" << numElements
                   << ") does not match expected block count (" << expectedCount << ")" << std::endl;
          numElements = std::min(numElements, expectedCount);
     }

    std::vector<MinMaxResult> results(numElements);

    // No vkInvalidateMappedMemoryRanges needed due to HOST_COHERENT
    memcpy(results.data(), mappedData, numElements * sizeof(MinMaxResult));

    destroyBuffer(readbackBuffer, device);
    std::cout << "GPU image results readback complete." << std::endl;

    uint32_t totalBlocks = numElements;
    std::cout << "GPU Results (first few blocks):" << std::endl;
    for (uint32_t i = 0; i < std::min(totalBlocks, 10u); ++i) {
        std::cout << "Block[" << i << "] Min: " << results[i].minVal
                  << " Max: " << results[i].maxVal << std::endl;
    }
     if (totalBlocks > 10u) {
        std::cout << "... (" << totalBlocks << " total blocks read)" << std::endl;
     }

    return results;
}

void pushDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
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

    VkDevice device = createDevice(instance, physicalDevice, familyIndex, false);
    assert(device);

    volkLoadDevice(device);

    vkCmdBeginRendering = vkCmdBeginRenderingKHR;
    vkCmdEndRendering = vkCmdEndRenderingKHR;
    vkCmdPipelineBarrier2 = vkCmdPipelineBarrier2KHR;

    VkQueue queue = nullptr;
    vkGetDeviceQueue(device, familyIndex, 0, &queue);
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    VkCommandPool commandPool = createCommandPool(device, familyIndex);
    // --- End Vulkan Initialization ---


    // --- Load Shader & Create Pipeline ---
    Shader minMaxCS{};
    assert(loadShader(minMaxCS, device, argv[0], "spirv/computeMinMax.comp.spv"));

    VkPipelineCache pipelineCache = nullptr;

    auto [pipelineLayout, setLayout] = createComputeMinMaxPipelineLayout(device);
    VkPipeline computeMinMaxPipeline = createComputePipeline(device, pipelineCache, minMaxCS, pipelineLayout);
    // --- End Pipeline Creation ---

    // --- Prepare Buffers and Images ---
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    // Staging buffer for input volume upload
    VkDeviceSize volumeBufferSize = volume.volume_data.size();
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device, memoryProperties, volumeBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, volume.volume_data.data(), volume.volume_data.size());

    // Input volume image
    VkExtent3D volumeExtent = {volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z};
    Image volImage = {}; // Assuming Image struct { VkImage image; VkDeviceMemory memory; VkImageView imageView; }
    createImage(volImage, device, memoryProperties, VK_IMAGE_TYPE_3D,
                volumeExtent.width, volumeExtent.height, volumeExtent.depth, 1,
                VK_FORMAT_R8_UINT, // Assuming uint8 input volume data
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT); // Usage for upload and shader read
    uploadVolumeImage(commandBuffer, volImage.image, stagingBuffer, volumeExtent);
    // Input volume image is now in VK_IMAGE_LAYOUT_GENERAL

    // Push constants
    PushConstants pushConstants = {};
    pushConstants.volumeDim = volume.volume_dims;
    pushConstants.blockDim = glm::uvec3(8, 8, 8); // Match shader workgroup size
    pushConstants.blockGridDim = (pushConstants.volumeDim + pushConstants.blockDim - 1u) / pushConstants.blockDim;

    // --- Create output image for MinMax results ---
    VkExtent3D gridExtent = {pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z};
    VkDeviceSize minMaxTotalBytes = static_cast<size_t>(gridExtent.width) * gridExtent.height * gridExtent.depth * sizeof(MinMaxResult); // Size based on uint32_t struct
    Image minMaxImage = {}; // Assuming Image struct
    createImage(minMaxImage, device, memoryProperties, VK_IMAGE_TYPE_3D,
                gridExtent.width, gridExtent.height, gridExtent.depth, 1,
                VK_FORMAT_R32G32_UINT, // Format to store uvec2 (min, max as uint32_t)
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT); // Usage for shader write & readback copy

    // Transition MinMax image layout for shader write
    transitionImage(commandBuffer, minMaxImage.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE ,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
    ;

    // --- Dispatch Compute Shader ---
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeMinMaxPipeline);

    // Push constants
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);

    // Push descriptors (using updated function)
    pushDescriptorSets(commandBuffer, pipelineLayout,
                       volImage.imageView,    // Input volume image view
                       minMaxImage.imageView); // Output min/max image view

    // Dispatch
    vkCmdDispatch(commandBuffer, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);
    // --- End Dispatch ---

    // End and submit commands
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
    VK_CHECK(vkDeviceWaitIdle(device)); // Wait for compute to finish

    // --- Readback Results ---
    // Use the new function to read back from the image
    std::vector<MinMaxResult> results = mapMinMaxImage(
        device, memoryProperties, commandPool, queue,
        minMaxImage, gridExtent, minMaxTotalBytes
    );
    // --- End Readback ---


    // --- Cleanup (Simplified - ensure all created resources are destroyed) ---
    VK_CHECK(vkResetCommandPool(device, commandPool, 0)); // Reset pool before destroying
    // vkDestroyCommandPool(device, commandPool, nullptr); // Destroy pool at end

    // vkDestroyPipeline(device, computeMinMaxPipeline, nullptr);
    // vkDestroyShaderModule(device, minMaxCS.module, nullptr);
    // vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    // vkDestroyDescriptorSetLayout(device, setLayout, nullptr);

    destroyImage(minMaxImage, device); // Destroy the new image
    destroyImage(volImage, device);
    destroyBuffer(stagingBuffer, device);
    // destroyBuffer(minMaxBuffer, device); // No longer needed

    // ... rest of cleanup ...
    // vkDestroyDevice(device, nullptr);
    // vkDestroyInstance(instance, nullptr);
    // volkFinalize();
    // --- End Cleanup ---

    return results;
}

