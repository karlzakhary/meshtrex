#include "common.h"
#include "objViewer.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <cstring>
#include <string>

#include "buffer.h"
#include "device.h"
#include "mesh.h"
#include "resources.h"
#include "shaders.h"
#include "swapchain.h"
#include "vulkan_utils.h"

std::tuple<VkPipelineLayout, VkDescriptorSetLayout> createObjectPipelineLayout(
    VkDevice device)
{
    VkDescriptorSetLayoutBinding setBindings[1] = {};
    setBindings[0].binding = 0;
    setBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    setBindings[0].descriptorCount = 1;
    setBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo setCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setCreateInfo.flags =
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    setCreateInfo.bindingCount = std::size(setBindings);
    setCreateInfo.pBindings = setBindings;

    VkDescriptorSetLayout setLayout = nullptr;
    VK_CHECK(
        vkCreateDescriptorSetLayout(device, &setCreateInfo, nullptr, &setLayout));

    VkPipelineLayoutCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    createInfo.setLayoutCount = 1;
    createInfo.pSetLayouts = &setLayout;

    VkPipelineLayout layout = nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &createInfo, nullptr, &layout));

    return std::make_tuple(layout, setLayout);
}

void drawObject(char **argv, const char *path)
{
    VK_CHECK(volkInitialize());
    std::string spath = argv[0];
    std::string::size_type pos = spath.find_last_of("/\\");
    if (pos == std::string::npos)
        spath = "";
    else
        spath = spath.substr(0, pos + 1);
    spath += path;
    Mesh mesh = {};
    loadMesh(mesh, spath.c_str());

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

    Shader triangleVS{};
    assert(loadShader(triangleVS, device, argv[0], "spirv/objViewer.vert.spv"));

    Shader triangleFS{};
    assert(loadShader(triangleFS, device, argv[0], "spirv/objViewer.frag.spv"));

    VkPipelineCache pipelineCache = nullptr;
    VkPipelineRenderingCreateInfo renderingInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainFormat,
        .depthAttachmentFormat = depthFormat};

    auto [pipelineLayout, setLayout] = createObjectPipelineLayout(device);
    VkPipeline trianglePipeline =
        createGraphicsPipeline(device, pipelineCache, renderingInfo,
                               {&triangleVS, &triangleFS}, pipelineLayout, {});

    Swapchain swapchain;
    createSwapchain(swapchain, physicalDevice, device, surface, familyIndex,
                    window, swapchainFormat);

    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    VkCommandPool commandPool = createCommandPool(device, familyIndex);
    assert(commandPool);

    VkCommandBufferAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};

    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    std::vector<VkImageView> swapchainImageViews(swapchain.imageCount);
    VkCommandBuffer commandBuffer = nullptr;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

    VkClearColorValue colorClear = {48.f / 255.f, 10.f / 255.f, 36.f / 255.f,
                                    1};

    Buffer vb = {};
    createBuffer(vb, device, memoryProperties, 128 * 1024 * 1024,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer ib = {};
    createBuffer(ib, device, memoryProperties, 128 * 1024 * 1024,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    assert(vb.size >= mesh.vertices.size() * sizeof(Vertex));
    memcpy(vb.data, mesh.vertices.data(),
           mesh.vertices.size() * sizeof(Vertex));

    assert(ib.size >= mesh.indices.size() * sizeof(uint32_t));
    memcpy(ib.data, mesh.indices.data(),
           mesh.indices.size() * sizeof(uint32_t));

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        SwapchainStatus swapchainStatus =
            updateSwapchain(swapchain, physicalDevice, device, surface,
                            familyIndex, window, swapchainFormat);

        if (swapchainStatus == Swapchain_NotReady) continue;

        if (swapchainStatus == Swapchain_Resized ||
            !swapchainImageViews.front()) {
            for (uint32_t i = 0; i < swapchain.imageCount; ++i) {
                if (swapchainImageViews[i])
                    vkDestroyImageView(device, swapchainImageViews[i], 0);

                swapchainImageViews[i] = createImageView(
                    device, swapchain.images[i], swapchainFormat, 0, 1);
            }
        }

        uint32_t imageIndex = 0;
        VkResult acquireResult = vkAcquireNextImageKHR(
            device, swapchain.swapchain, ~0ull, acquireSemaphore,
            VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) continue;
        VK_CHECK_SWAPCHAIN(acquireResult);

        VK_CHECK(vkResetCommandPool(device, commandPool, 0));

        VkCommandBufferBeginInfo beginInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

        VkImageMemoryBarrier2 renderBeginBarrier =
            imageBarrier(swapchain.images[imageIndex],
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR, 0,
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &renderBeginBarrier);

        VkRenderingAttachmentInfo colorAttachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = swapchainImageViews[imageIndex],
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = colorClear};

        VkRenderingInfo passInfo = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        passInfo.renderArea.extent.width = swapchain.width;
        passInfo.renderArea.extent.height = swapchain.height;
        passInfo.layerCount = 1;
        passInfo.colorAttachmentCount = 1;
        passInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(commandBuffer, &passInfo);

        VkViewport viewport = {0.0f,
                               static_cast<float>(swapchain.height),
                               static_cast<float>(swapchain.width),
                               -static_cast<float>(swapchain.height),
                               0.0f,
                               1.0f};
        VkRect2D scissor = {{0, 0}, {(swapchain.width), (swapchain.height)}};

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          trianglePipeline);

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = vb.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = vb.size;

        VkWriteDescriptorSet descriptors[1] = {};
        descriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptors[0].dstBinding = 0;
        descriptors[0].descriptorCount = 1;
        descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptors[0].pBufferInfo = &bufferInfo;

        vkCmdPushDescriptorSetKHR(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
            std::size(descriptors), descriptors);

        vkCmdBindIndexBuffer(commandBuffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, uint32_t(mesh.indices.size()), 1, 0, 0,
                         0);

        vkCmdEndRendering(commandBuffer);

        VkImageMemoryBarrier2 presentBarrier =
            imageBarrier(swapchain.images[imageIndex],
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR, 0,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &presentBarrier);

        VK_CHECK(vkEndCommandBuffer(commandBuffer));

        VkPipelineStageFlags submitStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &acquireSemaphore;
        submitInfo.pWaitDstStageMask = &submitStageMask;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &releaseSemaphore;

        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

        VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &releaseSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain.swapchain;
        presentInfo.pImageIndices = &imageIndex;

        VK_CHECK(vkQueuePresentKHR(queue, &presentInfo));

        VK_CHECK(vkDeviceWaitIdle(device));

        glfwWaitEvents();
    }

    VK_CHECK(vkDeviceWaitIdle(device));

    for (uint32_t i = 0; i < swapchain.imageCount; ++i)
        if (swapchainImageViews[i])
            vkDestroyImageView(device, swapchainImageViews[i], nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyPipeline(device, trianglePipeline, nullptr);
    vkDestroyShaderModule(device, triangleVS.module, nullptr);
    vkDestroyShaderModule(device, triangleFS.module, nullptr);
    destroySwapchain(device, swapchain);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyFence(device, frameFence, nullptr);
    vkDestroySemaphore(device, releaseSemaphore, nullptr);
    vkDestroySemaphore(device, acquireSemaphore, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);

    glfwDestroyWindow(window);
    destroyBuffer(vb, device);
    destroyBuffer(ib, device);
    vkDestroyDevice(device, nullptr);

    if (debugCallback)
        vkDestroyDebugReportCallbackEXT(instance, debugCallback, 0);

    vkDestroyInstance(instance, nullptr);

    volkFinalize();
}