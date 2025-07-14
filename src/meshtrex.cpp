#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#ifndef __APPLE__
#include <cstdint>
#endif

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include "renderdoc_app.h"

#include "common.h"
#include "vulkan_context.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "minMaxManager.h"
#include "filteringManager.h"
#include "extractionManager.h"
#include "extractionTestUtils.h"
#include "renderingPipeline.h"
#include "renderingManager.h"
#include "image.h"
#include "resources.h"


// --- Camera and Input State ---
glm::vec3 cameraPos = glm::vec3(128.0f, 128.0f, 350.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw = -90.0f;
float pitch = 0.0f;
double lastX = 512.0, lastY = 384.0;
bool firstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = (float)(xpos - lastX);
    float yoffset = (float)(lastY - ypos);
    lastX = xpos;
    lastY = ypos;
    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    yaw += xoffset;
    pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void processInput(GLFWwindow* window, float deltaTime) {
    float cameraSpeed = 150.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

// --- Main Application ---
int main(int argc, char** argv) {
    try {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(1024, 768, "MeshTrex", nullptr, nullptr);
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        VulkanContext context(true);
        VkDevice device = context.getDevice();
        VkQueue queue = context.getQueue();
        VkCommandPool commandPool = context.getCommandPool();

        VkSurfaceKHR surface = createSurface(context.getInstance(), window);
        VkFormat swapchainFormat = getSwapchainFormat(context.getPhysicalDevice(), surface);
        VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
        Swapchain swapchain;
        createSwapchain(swapchain, context.getPhysicalDevice(), device, surface, context.getGraphicsQueueFamilyIndex(), window, swapchainFormat);

        std::vector<VkImageView> swapchainImageViews(swapchain.imageCount);
        Image depthImage;
        
        auto createSwapchainResources = [&]() {
            for (uint32_t i = 0; i < swapchain.imageCount; ++i) {
                swapchainImageViews[i] = createImageView(device, swapchain.images[i], swapchainFormat, VK_IMAGE_TYPE_2D, 0, 1);
            }
            createImage(depthImage, device, context.getMemoryProperties(), VK_IMAGE_TYPE_2D, swapchain.width, swapchain.height, 1, 1, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);
            VkImageMemoryBarrier2 depthBarrier = imageBarrier(depthImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            pipelineBarrier(cmd, {}, 0, nullptr, 1, &depthBarrier);
            endSingleTimeCommands(device, commandPool, queue, cmd);
        };
        
        auto destroySwapchainResources = [&]() {
            destroyImage(depthImage, device);
            for(auto& view : swapchainImageViews) { if(view) vkDestroyImageView(device, view, nullptr); }
        };

        createSwapchainResources();
        
        std::string volumePath = getFullPath(ROOT_BUILD_PATH, "/raw_volumes/bonsai_256x256x256_uint8.raw");
        Volume volume = loadVolume(volumePath.c_str());
        
        PushConstants pushConstants = {};
        pushConstants.volumeDim = glm::uvec4(volume.volume_dims, 1);
        pushConstants.blockDim = glm::uvec4(4, 4, 4, 1);
        pushConstants.blockGridDim = glm::uvec4(
            (volume.volume_dims.x + pushConstants.blockDim.x - 1) / pushConstants.blockDim.x,
            (volume.volume_dims.y + pushConstants.blockDim.y - 1) / pushConstants.blockDim.y,
            (volume.volume_dims.z + pushConstants.blockDim.z - 1) / pushConstants.blockDim.z,
            1);
        pushConstants.isovalue = 80;

        std::cout << "Loaded volume dims: ("
                  << pushConstants.volumeDim.x << "x" << pushConstants.volumeDim.y << "x" << pushConstants.volumeDim.z << ")" << std::endl;
        std::cout << "Block grid: " << pushConstants.blockGridDim.x << "x" << pushConstants.blockGridDim.y << "x" << pushConstants.blockGridDim.z << std::endl;

        MinMaxOutput minMaxOutput = computeMinMaxMip(context, volume, pushConstants);
        FilteringOutput filteringResult = filterActiveBlocks(context, minMaxOutput, pushConstants);
        ExtractionOutput extractionResult = extractMeshletDescriptors(context, minMaxOutput, filteringResult, pushConstants);
        writeGPUExtractionToOBJ(context, extractionResult, "/tmp/meshtrex.obj");
        extractionResult.meshletCount = mapCounterBuffer(context, extractionResult.meshletDescriptorCountBuffer);
        std::cout << "Extraction complete. Found " << extractionResult.meshletCount << " meshlets." << std::endl;
        vkDeviceWaitIdle(context.getDevice());
        {
            VkCommandBuffer setupCmd = beginSingleTimeCommands(device, commandPool);

            std::vector<VkBufferMemoryBarrier2> bufferBarriers;
            bufferBarriers.reserve(4);

            VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
            VkAccessFlags2 srcAccess = VK_ACCESS_2_SHADER_WRITE_BIT;
            VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT; // First stage that reads the buffers
            VkAccessFlags2 dstAccess = VK_ACCESS_2_SHADER_READ_BIT;

            bufferBarriers.push_back(bufferBarrier(extractionResult.vertexBuffer.buffer, srcStage, srcAccess, dstStage, dstAccess, 0, VK_WHOLE_SIZE));
            bufferBarriers.push_back(bufferBarrier(extractionResult.indexBuffer.buffer, srcStage, srcAccess, dstStage, dstAccess, 0, VK_WHOLE_SIZE));
            bufferBarriers.push_back(bufferBarrier(extractionResult.meshletDescriptorBuffer.buffer, srcStage, srcAccess, dstStage, dstAccess, 0, VK_WHOLE_SIZE));
            bufferBarriers.push_back(bufferBarrier(extractionResult.meshletDescriptorCountBuffer.buffer, srcStage, srcAccess, dstStage, dstAccess, 0, VK_WHOLE_SIZE));
            
            pipelineBarrier(setupCmd, {}, bufferBarriers.size(), bufferBarriers.data(), 0, nullptr);
            
            endSingleTimeCommands(device, commandPool, queue, setupCmd);
        }

        const int MAX_FRAMES_IN_FLIGHT = 2;
        RenderingPipeline renderingPipeline;
        renderingPipeline.setup(device, swapchainFormat, depthFormat, VK_SAMPLE_COUNT_1_BIT, MAX_FRAMES_IN_FLIGHT);

        Buffer sceneUbo;
        createBuffer(sceneUbo, device, context.getMemoryProperties(), sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Create sync objects for frames in flight
        std::vector<VkSemaphore> imageAvailableSemaphores(MAX_FRAMES_IN_FLIGHT);
        std::vector<VkSemaphore> renderFinishedSemaphores(MAX_FRAMES_IN_FLIGHT);
        std::vector<VkFence> inFlightFences(MAX_FRAMES_IN_FLIGHT);
        std::vector<VkCommandBuffer> commandBuffers(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
        VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, (uint32_t)commandBuffers.size()};
        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]));
            VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]));
            VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]));
        }

        uint32_t currentFrame = 0;
        float lastFrameTime = 0.0f;
        // return 0;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            float currentFrameTime = (float)glfwGetTime();
            float deltaTime = currentFrameTime - lastFrameTime;
            lastFrameTime = currentFrameTime;
            processInput(window, deltaTime);

            // **FIX: Use the sync objects for the current frame in flight**
            VK_CHECK(vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX));

            uint32_t imageIndex;
            VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

            if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR || acquireResult == VK_SUBOPTIMAL_KHR) {
                vkDeviceWaitIdle(device);
                destroySwapchainResources();
                createSwapchain(swapchain, context.getPhysicalDevice(), device, surface, context.getGraphicsQueueFamilyIndex(), window, swapchainFormat);
                createSwapchainResources();
                continue;
            }
            VK_CHECK(acquireResult);
            
            VK_CHECK(vkResetFences(device, 1, &inFlightFences[currentFrame]));
            VkCommandBuffer commandBuffer = commandBuffers[currentFrame];
            VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
            
            VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
            VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
            
            SceneUniforms uniforms;
            glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)swapchain.width / (float)swapchain.height, 0.1f, 1000.0f);
            projection[1][1] *= -1; // Invert Y axis
            uniforms.viewProjectionMatrix = projection * glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            uniforms.cameraPos = cameraPos;
            uniforms.lightPos = glm::vec4(400.0f, 400.0f, 400.0f, 1.0f);
            memcpy(sceneUbo.data, &uniforms, sizeof(SceneUniforms));
            VkMemoryBarrier2 uboBarrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, // The first stage that reads the UBO
                .dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT
            };
            VkDependencyInfo dependencyInfo = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &uboBarrier };
            vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
            updateRenderingDescriptors(device, renderingPipeline, currentFrame, sceneUbo, extractionResult);

            VkImageMemoryBarrier2 renderBeginBarrier = imageBarrier(swapchain.images[imageIndex], VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &renderBeginBarrier);

            recordRenderingCommands(commandBuffer, renderingPipeline, currentFrame, extractionResult, {swapchain.width, swapchain.height}, swapchainImageViews[imageIndex], depthImage.imageView);
            
            VkImageMemoryBarrier2 presentBarrier = imageBarrier(swapchain.images[imageIndex], VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
            pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &presentBarrier);
            
            VK_CHECK(vkEndCommandBuffer(commandBuffer));

            VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
            VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
            VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, waitSemaphores, waitStages, 1, &commandBuffer, 1, signalSemaphores};
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, inFlightFences[currentFrame]));

            VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr, 1, signalSemaphores, 1, &swapchain.swapchain, &imageIndex, nullptr};
            acquireResult = vkQueuePresentKHR(queue, &presentInfo);

            if (acquireResult != VK_ERROR_OUT_OF_DATE_KHR && acquireResult != VK_SUBOPTIMAL_KHR) {
                 VK_CHECK_SWAPCHAIN(acquireResult);
            }

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        }
        vkDestroyCommandPool(device, commandPool, nullptr);
        destroyBuffer(sceneUbo, device);
        destroySwapchainResources();
        destroySwapchain(device, swapchain);
        vkDestroySurfaceKHR(context.getInstance(), surface, nullptr);
        extractionResult.cleanup();
        filteringResult.cleanup(device);
        minMaxOutput.cleanup(device);
        renderingPipeline.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}