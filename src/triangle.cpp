#include "common.h"
#include "triangle.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "device.h"
#include "glm/vec4.hpp"
#include "resources.h"
#include "shaders.h"
#include "swapchain.h"
#include "vulkan_utils.h"

VkPipelineLayout createBasicPipelineLayout(VkDevice device)
{
    VkPipelineLayoutCreateInfo layoutInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pSetLayouts = nullptr;
    layoutInfo.pPushConstantRanges = nullptr;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));
    return layout;
}

void drawTriangle(char **argv)
{
    VK_CHECK(volkInitialize());

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

    GLFWwindow *window =
        glfwCreateWindow(1024, 768, "meshtrex", nullptr, nullptr);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
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
    assert(loadShader(triangleVS, device, argv[0], "spirv/triangle.vert.spv"));

    Shader triangleFS{};
    assert(loadShader(triangleFS, device, argv[0], "spirv/triangle.frag.spv"));

    VkPipelineCache pipelineCache = nullptr;
    VkPipelineRenderingCreateInfo renderingInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainFormat,
        .depthAttachmentFormat = depthFormat};

    VkPipelineLayout pipelineLayout = createBasicPipelineLayout(device);
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
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

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
    vkDestroyFence(device, frameFence, nullptr);
    vkDestroySemaphore(device, releaseSemaphore, nullptr);
    vkDestroySemaphore(device, acquireSemaphore, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);

    glfwDestroyWindow(window);

    vkDestroyDevice(device, nullptr);

    if (debugCallback)
        vkDestroyDebugReportCallbackEXT(instance, debugCallback, 0);

    vkDestroyInstance(instance, 0);

    volkFinalize();
}