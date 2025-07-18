#include "renderingManager.h"
#include "vulkan_utils.h"
#include "resources.h"
#include "buffer.h"
#include "image.h"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cstring>

struct RenderPushConstants {
    glm::mat4 viewProj;
};

void RenderingManager::handleCameraInput(float deltaTime) {
    const float moveSpeed = 1.0f * deltaTime; // units per second
    glm::vec3 forward = glm::normalize(cameraTarget_ - cameraPos_);
    glm::vec3 right = glm::normalize(glm::cross(forward, cameraUp_));

    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPos_ += forward * moveSpeed;
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPos_ -= forward * moveSpeed;
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
        cameraPos_ -= right * moveSpeed;
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
        cameraPos_ += right * moveSpeed;
    }
    if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS) {
        cameraPos_ += cameraUp_ * moveSpeed;
    }
    if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS) {
        cameraPos_ -= cameraUp_ * moveSpeed;
    }
    
    // Update target to maintain view direction
    cameraTarget_ = cameraPos_ + forward;
}

RenderingManager::RenderingManager(VulkanContext& context, uint32_t width, uint32_t height, const char* title)
    : context_(context) {
    initWindow(width, height, title);

    surface_ = createSurface(context_.getInstance(), window_);
    assert(surface_);

    swapchainFormat_ = getSwapchainFormat(context_.getPhysicalDevice(), surface_);
    depthFormat_ = VK_FORMAT_D32_SFLOAT;

    // Create Descriptor Set Layout
    VkDescriptorSetLayoutBinding setBindings[3] = {};
    setBindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr}; // Vertex Buffer
    setBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT |VK_SHADER_STAGE_MESH_BIT_EXT, nullptr}; // Index Buffer
    setBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr}; // Meshlet Descriptors

    VkDescriptorSetLayoutCreateInfo setLayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    setLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    setLayoutInfo.bindingCount = 3;
    setLayoutInfo.pBindings = setBindings;

    VK_CHECK(vkCreateDescriptorSetLayout(context_.getDevice(), &setLayoutInfo, nullptr, &descriptorSetLayout_));

    // Create Pipeline Layout
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    pcRange.offset = 0;
    pcRange.size = sizeof(RenderPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;

    VK_CHECK(vkCreatePipelineLayout(context_.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    createSwapchainResources();
    
    renderingPipeline_.setup(context.getDevice(), swapchainFormat_, depthFormat_, pipelineLayout_);
}

RenderingManager::~RenderingManager() {
    vkDeviceWaitIdle(context_.getDevice());
    
    renderingPipeline_.cleanup();
    cleanupSwapchainResources();
    
    // Destroy layouts and surface
    vkDestroyPipelineLayout(context_.getDevice(), pipelineLayout_, nullptr);
    vkDestroyDescriptorSetLayout(context_.getDevice(), descriptorSetLayout_, nullptr); // Now destroyed correctly
    vkDestroySurfaceKHR(context_.getInstance(), surface_, nullptr);
    
    glfwDestroyWindow(window_);
}

void RenderingManager::initWindow(uint32_t width, uint32_t height, const char* title) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    assert(window_);

    glfwGetCursorPos(window_, &lastMouseX_, &lastMouseY_);
}

void RenderingManager::createSwapchainResources() {
    createSwapchain(swapchain_, context_.getPhysicalDevice(), context_.getDevice(), surface_, context_.getGraphicsQueueFamilyIndex(), window_, swapchainFormat_);

    createImage(depthImage_, context_.getDevice(), context_.getMemoryProperties(), VK_IMAGE_TYPE_2D, swapchain_.width, swapchain_.height, 1, 1, depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    
    swapchainImageViews_.resize(swapchain_.imageCount);
    for (uint32_t i = 0; i < swapchain_.imageCount; ++i) {
        swapchainImageViews_[i] = createImageView(context_.getDevice(), swapchain_.images[i], swapchainFormat_, VK_IMAGE_TYPE_2D, 0, 1);
    }
}

void RenderingManager::cleanupSwapchainResources() {
    destroyImage(depthImage_, context_.getDevice());

    for (auto imageView : swapchainImageViews_) {
        vkDestroyImageView(context_.getDevice(), imageView, nullptr);
    }
    destroySwapchain(context_.getDevice(), swapchain_);
}

uint32_t RenderingManager::readCounterFromBuffer(const Buffer& counterBuffer) {
    uint32_t count = 0;
    VkDeviceSize bufferSize = sizeof(uint32_t);

    if (counterBuffer.buffer == VK_NULL_HANDLE || counterBuffer.size < bufferSize) return 0;

    Buffer readbackBuffer{};
    createBuffer(readbackBuffer, context_.getDevice(), context_.getMemoryProperties(), bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkCommandBuffer cmd = beginSingleTimeCommands(context_.getDevice(), context_.getCommandPool());
    
    VkBufferCopy region = {0, 0, bufferSize};
    vkCmdCopyBuffer(cmd, counterBuffer.buffer, readbackBuffer.buffer, 1, &region);

    endSingleTimeCommands(context_.getDevice(), context_.getCommandPool(), context_.getQueue(), cmd);
    
    memcpy(&count, readbackBuffer.data, bufferSize);
    
    destroyBuffer(readbackBuffer, context_.getDevice());
    
    return count;
}


void RenderingManager::render(const ExtractionOutput& extractionOutput) {
    VkDevice device = context_.getDevice();
    uint32_t actualMeshletCount = readCounterFromBuffer(extractionOutput.meshletDescriptorCountBuffer);
    
    std::cout << "Rendering " << actualMeshletCount << " meshlets." << std::endl;
    if (actualMeshletCount == 0) return;

    VkSemaphore acquireSemaphore = createSemaphore(device);
    VkSemaphore releaseSemaphore = createSemaphore(device);

    VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence frameFence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frameFence));

    // Allocate one command buffer to be reused for the render loop
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = context_.getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));


    while (!glfwWindowShouldClose(window_)) {
        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - lastFrameTime_;
        lastFrameTime_ = currentTime;
        glfwPollEvents();
        handleCameraInput(deltaTime);
        // Update camera based on mouse input
        double mouseX, mouseY;
        glfwGetCursorPos(window_, &mouseX, &mouseY);
        if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            float dx = (float)(mouseX - lastMouseX_);
            float dy = (float)(mouseY - lastMouseY_);

            glm::mat4 rotationY = glm::rotate(glm::mat4(1.0f), -dx * 0.005f, glm::vec3(0,1,0));
            cameraPos_ = glm::vec3(rotationY * glm::vec4(cameraPos_ - cameraTarget_, 1.0f)) + cameraTarget_;

            glm::vec3 right = glm::normalize(glm::cross(cameraTarget_ - cameraPos_, glm::vec3(0,1,0)));
            glm::mat4 rotationX = glm::rotate(glm::mat4(1.0f), -dy * 0.005f, right);
            cameraPos_ = glm::vec3(rotationX * glm::vec4(cameraPos_ - cameraTarget_, 1.0f)) + cameraTarget_;
        }
        lastMouseX_ = mouseX;
        lastMouseY_ = mouseY;

        VK_CHECK(vkWaitForFences(device, 1, &frameFence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &frameFence));

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapchain_.swapchain, UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE, &imageIndex);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            vkDeviceWaitIdle(device);
            cleanupSwapchainResources();
            createSwapchainResources();
            continue;
        }
        assert(result == VK_SUCCESS);

        // Reset the command buffer for new recording
        VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
        
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));


        // Transition color image
        VkImageMemoryBarrier2 colorBarrier = imageBarrier(swapchain_.images[imageIndex], VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        
        // Transition depth image
        VkImageMemoryBarrier2 depthBarrier = imageBarrier(depthImage_.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        
        VkImageMemoryBarrier2 barriers[] = {colorBarrier, depthBarrier};
        pipelineBarrier(commandBuffer, 0, 0, nullptr, 2, barriers);

        VkClearColorValue clearColor = {0.1f, 0.2f, 0.3f, 1.0f};
        VkClearDepthStencilValue clearDepth = {0.0f, 0}; // Clear to far plane for reversed-Z

        VkRenderingAttachmentInfo colorAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView = swapchainImageViews_[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = clearColor;

        VkRenderingAttachmentInfo depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView = depthImage_.imageView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = clearDepth;

        VkRenderingInfo renderingInfo = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderingInfo.renderArea = {{0, 0}, {swapchain_.width, swapchain_.height}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        VkViewport viewport = {0.0f, (float)swapchain_.height, (float)swapchain_.width, -(float)swapchain_.height, 0.0f, 1.0f};
        VkRect2D scissor = {{0, 0}, {swapchain_.width, swapchain_.height}};
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderingPipeline_.pipeline_);

        RenderPushConstants pushConsts;
        // Create reversed-Z projection matrix for better depth precision
        float fov = glm::radians(60.0f);
        float aspect = (float)swapchain_.width / (float)swapchain_.height;
        float nearPlane = 0.1f;
        float farPlane = 1000.0f;
        
        glm::mat4 proj = glm::mat4(0.0f);
        float tanHalfFov = tan(fov / 2.0f);
        proj[0][0] = 1.0f / (aspect * tanHalfFov);
        proj[1][1] = 1.0f / tanHalfFov;
        proj[2][2] = nearPlane / (farPlane - nearPlane);
        proj[2][3] = -1.0f;
        proj[3][2] = (farPlane * nearPlane) / (farPlane - nearPlane);
        glm::mat4 view = glm::lookAt(cameraPos_, cameraTarget_, glm::vec3(0, 1, 0));
        pushConsts.viewProj = proj * view;

        vkCmdPushConstants(commandBuffer, pipelineLayout_, VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(RenderPushConstants), &pushConsts);

        VkDescriptorBufferInfo vbInfo = {extractionOutput.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo ibInfo = {extractionOutput.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo mbInfo = {extractionOutput.meshletDescriptorBuffer.buffer, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet writes[3] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, 0, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vbInfo, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, 0, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ibInfo, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, 0, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mbInfo, nullptr};

        vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 3, writes);

        vkCmdDrawMeshTasksEXT(commandBuffer, actualMeshletCount, 1, 1);

        vkCmdEndRendering(commandBuffer);
        
        VkImageMemoryBarrier2 presentBarrier = imageBarrier(swapchain_.images[imageIndex], VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &presentBarrier);

        VK_CHECK(vkEndCommandBuffer(commandBuffer));

        VkPipelineStageFlags submitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &acquireSemaphore;
        submitInfo.pWaitDstStageMask = &submitStageMask;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &releaseSemaphore;
        
        VK_CHECK(vkQueueSubmit(context_.getQueue(), 1, &submitInfo, frameFence));

        VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &releaseSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain_.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        
        result = vkQueuePresentKHR(context_.getQueue(), &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            // Handled at top of loop
        } else {
            assert(result == VK_SUCCESS);
        }
    }

    vkDeviceWaitIdle(device);

    vkFreeCommandBuffers(device, context_.getCommandPool(), 1, &commandBuffer);
    vkDestroyFence(device, frameFence, nullptr);
    vkDestroySemaphore(device, releaseSemaphore, nullptr);
    vkDestroySemaphore(device, acquireSemaphore, nullptr);
}
