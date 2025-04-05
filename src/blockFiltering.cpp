#include "common.h"
#include "blockFiltering.h"

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
#include "swapchain.h"
#include "vulkan_utils.h"

struct alignas(16) PushConstants {
    glm::ivec3 volumeDim;
    glm::ivec3 blockDim;
    glm::ivec3 blockGridDim;
};

std::tuple<glm::uvec3, std::string, std::vector<uint8_t>> loadVolume(const char *path)
{
    const std::string file = path;
    //const float isovalue = std::stof(argv[2]);
    const std::regex match_filename(R"((\w+)_(\d+)x(\d+)x(\d+)_(.+)\.raw)");
    auto matches = std::sregex_iterator(file.begin(), file.end(), match_filename);
    if (matches == std::sregex_iterator() || matches->size() != 6) {
        std::cerr << "Unrecognized raw volume naming scheme, expected a format like: "
                  << "'<name>_<X>x<Y>x<Z>_<data type>.raw' but '" << file << "' did not match"
                  << std::endl;
        throw std::runtime_error("Invalaid raw file naming scheme");
    }
    const glm::uvec3 volume_dims(
        std::stoi((*matches)[2]), std::stoi((*matches)[3]), std::stoi((*matches)[4]));
    const std::string volume_type = (*matches)[5];

    const size_t volume_bytes =
        static_cast<size_t>(volume_dims.x) * static_cast<size_t>(volume_dims.y) * static_cast<size_t>(volume_dims.z);
    std::vector<uint8_t> volume_data(volume_bytes, 0);
    std::ifstream fin(file.c_str(), std::ios::binary);
    fin.read(reinterpret_cast<char*>(volume_data.data()), volume_data.size());
    return std::make_tuple(volume_dims, volume_type, volume_data);
}

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

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    return commandBuffer;
}

// Helper to end and submit a single-use command buffer
void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                           VkQueue queue, VkCommandBuffer commandBuffer) {
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer
    };

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// Helper to insert an image layout transition barrier
void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

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


// Added function to validate the min/max buffer after compute shader execution.
void validateMinMaxBuffer(VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
                          VkCommandPool commandPool, VkQueue queue,
                          const Buffer& minMaxBuffer, size_t minMaxSize, glm::ivec3 blockGridDim) {
    Buffer readbackBuffer;
    createBuffer(readbackBuffer, device, memoryProperties, minMaxSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);

    VkBufferCopy region = {0, 0, minMaxSize};
    vkCmdCopyBuffer(cmd, minMaxBuffer.buffer, readbackBuffer.buffer, 1, &region);

    endSingleTimeCommands(device, commandPool, queue, cmd);

    auto* data = reinterpret_cast<glm::vec2*>(readbackBuffer.data);
    uint32_t totalBlocks = blockGridDim.x * blockGridDim.y * blockGridDim.z;

    for (uint32_t i = 0; i < std::min(totalBlocks, 32u); ++i) {
        std::cout << "Block[" << i << "] Min: " << data[i].x << " Max: " << data[i].y << std::endl;
    }

    destroyBuffer(readbackBuffer, device);
}

void filterUnoccupiedBlocks(char **argv, const char *path)
{
    VK_CHECK(volkInitialize());
    std::string spath = argv[0];
    std::string::size_type pos = spath.find_last_of("/\\");
    if (pos == std::string::npos)
        spath = "";
    else
        spath = spath.substr(0, pos + 1);
    spath += path;
    auto [volume_dims, volume_type, volume_data] = loadVolume(spath.c_str());

    int rc = glfwInit();
    assert(rc);

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

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window =
        glfwCreateWindow(1024, 768, "meshtrex", nullptr, nullptr);
    assert(window);

    VkSurfaceKHR surface = createSurface(instance, window);
    assert(surface);

    VkBool32 presentSupported = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex,
                                                  surface, &presentSupported));
    assert(presentSupported);

    VkFormat swapchainFormat = getSwapchainFormat(physicalDevice, surface);
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkSemaphore acquireSemaphore = createSemaphore(device);
    assert(acquireSemaphore);

    VkSemaphore releaseSemaphore = createSemaphore(device);
    assert(releaseSemaphore);

    VkFence frameFence = createFence(device);
    assert(frameFence);

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

    Swapchain swapchain;
    createSwapchain(swapchain, physicalDevice, device, surface, familyIndex,
                    window, swapchainFormat);

    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    VkCommandPool commandPool = createCommandPool(device, familyIndex);
    assert(commandPool);

    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
    assert(commandBuffer);
    VkExtent3D extent = {volume_dims.x, volume_dims.y, volume_dims.z};
    VkDeviceSize bufferSize = volume_data.size(); // For uint8 volume: 256*256*225

    Buffer stagingBuffer = {};
    createBuffer(
        stagingBuffer,
        device,
        memoryProperties,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    memcpy(stagingBuffer.data, volume_data.data(), volume_data.size());

    Image volImage = {};

    createImage(volImage,
        device,
        memoryProperties,
        VK_IMAGE_TYPE_3D,
        volume_dims.x, volume_dims.y, volume_dims.z, 1,
        VK_FORMAT_R8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    uploadVolumeImage(commandBuffer, volImage.image, stagingBuffer, extent);

    // Create output buffer for compute shader results
    VkDeviceSize minMaxSize = (volume_dims.x / 16) * (volume_dims.y / 16) * (volume_dims.z / 16) * sizeof(glm::vec2);
    Buffer minMaxBuffer;
    createBuffer(minMaxBuffer, device, memoryProperties,
                 minMaxSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);


    endSingleTimeCommands(device, commandPool, queue, commandBuffer);

    VK_CHECK(vkResetCommandPool(device, commandPool, 0));

    commandBuffer = beginSingleTimeCommands(device, commandPool);

    // Bind the compute pipeline and dispatch
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeMinMaxPipeline);

    PushConstants pushConstants = {};
    pushConstants.volumeDim = glm::ivec3(volume_dims.x, volume_dims.y, volume_dims.z);
    pushConstants.blockDim = glm::ivec3(16, 16, 16); // Example block size
    pushConstants.blockGridDim = glm::ivec3((volume_dims.x + 15) / 16, (volume_dims.y + 15) / 16, (volume_dims.z + 15) / 16);

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);

    VkDescriptorImageInfo imageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = volImage.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };

    VkDescriptorBufferInfo bufferInfo = {
        .buffer = minMaxBuffer.buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    VkWriteDescriptorSet writes[2] = {};

    writes[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .pImageInfo = &imageInfo
    };

    writes[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &bufferInfo
    };

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 2, writes);

    vkCmdDispatch(commandBuffer, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        // .waitSemaphoreCount = 1,
        // .pWaitSemaphores = &acquireSemaphore,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
        // .signalSemaphoreCount = 1,
        // .pSignalSemaphores = &releaseSemaphore
    };

    // VK_CHECK(vkWaitForFences(device, 1, &frameFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device, 1, &frameFence));
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    VK_CHECK(vkDeviceWaitIdle(device));
    validateMinMaxBuffer(device, memoryProperties, commandPool, queue, minMaxBuffer, minMaxSize, pushConstants.blockGridDim);
    // Add to filterUnoccupiedBlocks function.
    std::cout << "Block Dim: " << pushConstants.blockDim.x << ", " << pushConstants.blockDim.y << ", " << pushConstants.blockDim.z << std::endl;
    std::cout << "Block Grid Dim: " << pushConstants.blockGridDim.x << ", " << pushConstants.blockGridDim.y << ", " << pushConstants.blockGridDim.z << std::endl;
    glfwWaitEvents();
}