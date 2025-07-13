#include "renderingFramework.h"
#include "renderingManager.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>

// Helper function to find memory type
static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, 
                               const VkPhysicalDeviceMemoryProperties& memProperties) {
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

// Camera implementation
Camera::Camera(float fov, float aspect, float near, float far)
    : position_(0.0f, 0.0f, 3.0f)
    , front_(0.0f, 0.0f, -1.0f)
    , up_(0.0f, 1.0f, 0.0f)
    , yaw_(-90.0f)
    , pitch_(0.0f)
    , fov_(fov)
    , aspect_(aspect)
    , nearPlane_(near)
    , farPlane_(far) {
    updateVectors();
}

void Camera::updateAspect(float aspect) {
    aspect_ = aspect;
}

void Camera::setPosition(const glm::vec3& pos) {
    position_ = pos;
}

void Camera::lookAt(const glm::vec3& target, const glm::vec3& up) {
    front_ = glm::normalize(target - position_);
    right_ = glm::normalize(glm::cross(front_, up));
    up_ = glm::cross(right_, front_);
    
    // Calculate yaw and pitch from front vector
    yaw_ = glm::degrees(atan2(front_.z, front_.x));
    pitch_ = glm::degrees(asin(front_.y));
}

void Camera::rotate(float yaw, float pitch) {
    yaw_ += yaw;
    pitch_ += pitch;
    
    // Constrain pitch
    if (pitch_ > 89.0f) pitch_ = 89.0f;
    if (pitch_ < -89.0f) pitch_ = -89.0f;
    
    updateVectors();
}

void Camera::move(const glm::vec3& direction) {
    position_ += direction;
}

void Camera::zoom(float amount) {
    fov_ -= amount;
    if (fov_ < 1.0f) fov_ = 1.0f;
    if (fov_ > 90.0f) fov_ = 90.0f;
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position_, position_ + front_, up_);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(fov_), aspect_, nearPlane_, farPlane_);
}

void Camera::updateVectors() {
    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front_ = glm::normalize(front);
    
    right_ = glm::normalize(glm::cross(front_, glm::vec3(0.0f, 1.0f, 0.0f)));
    up_ = glm::normalize(glm::cross(right_, front_));
}

// RenderingFramework implementation
RenderingFramework::RenderingFramework(VulkanContext& context, uint32_t width, uint32_t height)
    : context_(context)
    , camera_(45.0f, static_cast<float>(width) / height, 0.1f, 1000.0f) {
    
    createWindow(width, height);
    createSurface();
    createSwapchain();
    createImageViews();
    createDepthResources();
    createRenderingPipeline();
    createSceneUBO();
    createCommandBuffers();
    createSyncObjects();
    
    // Set initial camera position
    camera_.setPosition(glm::vec3(5.0f, 5.0f, 5.0f));
    camera_.lookAt(glm::vec3(0.0f, 0.0f, 0.0f));
}

RenderingFramework::~RenderingFramework() {
    cleanup();
}

void RenderingFramework::createWindow(uint32_t width, uint32_t height) {
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwInit();
    // GLFWwindow *window =
    //     glfwCreateWindow(1024, 768, "meshtrex", nullptr, nullptr);
    // assert(window);
    
    window_ = glfwCreateWindow(width, height, "MeshTrex - Marching Cubes Renderer", nullptr, nullptr);
    if (!window_) {
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    glfwSetCursorPosCallback(window_, mouseCallback);
    glfwSetScrollCallback(window_, scrollCallback);
    
    // Capture mouse
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void RenderingFramework::createSurface() {
    VK_CHECK(glfwCreateWindowSurface(context_.getInstance(), window_, nullptr, &surface_));
}

void RenderingFramework::createSwapchain() {
    SwapchainSupportDetails swapchainSupport = querySwapchainSupport();
    
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapchainSupport.capabilities);
    
    uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
    if (swapchainSupport.capabilities.maxImageCount > 0 && 
        imageCount > swapchainSupport.capabilities.maxImageCount) {
        imageCount = swapchainSupport.capabilities.maxImageCount;
    }
    
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    
    VK_CHECK(vkCreateSwapchainKHR(context_.getDevice(), &createInfo, nullptr, &swapchain_));
    
    vkGetSwapchainImagesKHR(context_.getDevice(), swapchain_, &imageCount, nullptr);
    swapchainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(context_.getDevice(), swapchain_, &imageCount, swapchainImages_.data());
    
    swapchainImageFormat_ = surfaceFormat.format;
    swapchainExtent_ = extent;
}

void RenderingFramework::createImageViews() {
    swapchainImageViews_.resize(swapchainImages_.size());
    
    for (size_t i = 0; i < swapchainImages_.size(); i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages_[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainImageFormat_;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        
        VK_CHECK(vkCreateImageView(context_.getDevice(), &viewInfo, nullptr, &swapchainImageViews_[i]));
    }
}

void RenderingFramework::createDepthResources() {
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    
    // Create depth image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = swapchainExtent_.width;
    imageInfo.extent.height = swapchainExtent_.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VK_CHECK(vkCreateImage(context_.getDevice(), &imageInfo, nullptr, &depthImage_));
    
    // Allocate memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(context_.getDevice(), depthImage_, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, 
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                context_.getMemoryProperties());
    
    VK_CHECK(vkAllocateMemory(context_.getDevice(), &allocInfo, nullptr, &depthImageMemory_));
    VK_CHECK(vkBindImageMemory(context_.getDevice(), depthImage_, depthImageMemory_, 0));
    
    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    VK_CHECK(vkCreateImageView(context_.getDevice(), &viewInfo, nullptr, &depthImageView_));
}

void RenderingFramework::createRenderingPipeline() {
    if (!renderingPipeline_.setup(context_.getDevice(), swapchainImageFormat_, 
                                  VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT)) {
        throw std::runtime_error("Failed to create rendering pipeline");
    }
}

void RenderingFramework::createSceneUBO() {
    VkDeviceSize bufferSize = sizeof(SceneData);
    
    createBuffer(sceneUBO_, context_.getDevice(), context_.getMemoryProperties(),
                 bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Map permanently for easy updates
    if (sceneUBO_.data == nullptr) {
        throw std::runtime_error("Failed to map scene UBO");
    }
}

void RenderingFramework::createCommandBuffers() {
    commandBuffers_.resize(MAX_FRAMES_IN_FLIGHT);
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = context_.getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
    
    VK_CHECK(vkAllocateCommandBuffers(context_.getDevice(), &allocInfo, commandBuffers_.data()));
}

void RenderingFramework::createSyncObjects() {
    imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateSemaphore(context_.getDevice(), &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]));
        VK_CHECK(vkCreateSemaphore(context_.getDevice(), &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]));
        VK_CHECK(vkCreateFence(context_.getDevice(), &fenceInfo, nullptr, &inFlightFences_[i]));
    }
}

SwapchainSupportDetails RenderingFramework::querySwapchainSupport() {
    SwapchainSupportDetails details;
    
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context_.getPhysicalDevice(), surface_, &details.capabilities);
    
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(context_.getPhysicalDevice(), surface_, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(context_.getPhysicalDevice(), surface_, &formatCount, details.formats.data());
    }
    
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(context_.getPhysicalDevice(), surface_, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(context_.getPhysicalDevice(), surface_, &presentModeCount, details.presentModes.data());
    }
    
    return details;
}

VkSurfaceFormatKHR RenderingFramework::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR RenderingFramework::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D RenderingFramework::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);
        
        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        
        actualExtent.width = std::clamp(actualExtent.width, 
                                       capabilities.minImageExtent.width, 
                                       capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, 
                                        capabilities.minImageExtent.height, 
                                        capabilities.maxImageExtent.height);
        
        return actualExtent;
    }
}

void RenderingFramework::updateSceneUBO() {
    sceneData_.viewProjectionMatrix = camera_.getProjectionMatrix() * camera_.getViewMatrix();
    sceneData_.modelMatrix = glm::mat4(1.0f);
    sceneData_.cameraPos_world = camera_.getPosition();
    sceneData_.pad = 0.0f;
    
    memcpy(sceneUBO_.data, &sceneData_, sizeof(SceneData));
}

void RenderingFramework::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, 
                                            const ExtractionOutput& extractionOutput) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    
    // Transition swapchain image to color attachment
    VkImageMemoryBarrier2 colorBarrier{};
    colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    colorBarrier.srcAccessMask = 0;
    colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.image = swapchainImages_[imageIndex];
    colorBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    
    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &colorBarrier;
    
    vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    
    // Update rendering descriptors
    updateRenderingDescriptors(context_.getDevice(), renderingPipeline_, sceneUBO_, extractionOutput);
    
    // Record rendering commands
    recordRenderingCommands(commandBuffer, context_.getDevice(), renderingPipeline_, extractionOutput,
                           swapchainExtent_, swapchainImageViews_[imageIndex], depthImageView_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    
    // Transition swapchain image to present
    colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    colorBarrier.dstAccessMask = 0;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    
    VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void RenderingFramework::renderFrame(const ExtractionOutput& extractionOutput) {
    vkWaitForFences(context_.getDevice(), 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(context_.getDevice(), swapchain_, UINT64_MAX,
                                           imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }
    
    vkResetFences(context_.getDevice(), 1, &inFlightFences_[currentFrame_]);
    
    vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
    updateSceneUBO();
    recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex, extractionOutput);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[currentFrame_];
    
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[currentFrame_]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    
    VK_CHECK(vkQueueSubmit(context_.getQueue(), 1, &submitInfo, inFlightFences_[currentFrame_]));
    
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    
    VkSwapchainKHR swapChains[] = {swapchain_};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    
    result = vkQueuePresentKHR(context_.getQueue(), &presentInfo);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized_) {
        framebufferResized_ = false;
        recreateSwapchain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image");
    }
    
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void RenderingFramework::processInput() {
    float cameraSpeed = 0.05f;
    
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) {
        glm::vec3 forward = -glm::vec3(camera_.getViewMatrix()[0][2], camera_.getViewMatrix()[1][2], camera_.getViewMatrix()[2][2]);
        camera_.move(cameraSpeed * forward);
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
        glm::vec3 forward = -glm::vec3(camera_.getViewMatrix()[0][2], camera_.getViewMatrix()[1][2], camera_.getViewMatrix()[2][2]);
        camera_.move(-cameraSpeed * forward);
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
        glm::vec3 right = glm::vec3(camera_.getViewMatrix()[0][0], camera_.getViewMatrix()[1][0], camera_.getViewMatrix()[2][0]);
        camera_.move(-cameraSpeed * right);
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
        glm::vec3 right = glm::vec3(camera_.getViewMatrix()[0][0], camera_.getViewMatrix()[1][0], camera_.getViewMatrix()[2][0]);
        camera_.move(cameraSpeed * right);
    }
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, true);
}

void RenderingFramework::onMouseMove(double xpos, double ypos) {
    if (firstMouse_) {
        lastX_ = static_cast<float>(xpos);
        lastY_ = static_cast<float>(ypos);
        firstMouse_ = false;
    }
    
    float xoffset = static_cast<float>(xpos) - lastX_;
    float yoffset = lastY_ - static_cast<float>(ypos);
    lastX_ = static_cast<float>(xpos);
    lastY_ = static_cast<float>(ypos);
    
    float sensitivity = 0.1f;
    camera_.rotate(xoffset * sensitivity, yoffset * sensitivity);
}

void RenderingFramework::onMouseScroll(double xoffset, double yoffset) {
    camera_.zoom(static_cast<float>(yoffset));
}

void RenderingFramework::onFramebufferResize(int width, int height) {
    framebufferResized_ = true;
}

bool RenderingFramework::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

void RenderingFramework::waitIdle() {
    vkDeviceWaitIdle(context_.getDevice());
}

void RenderingFramework::recreateSwapchain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
    }
    
    vkDeviceWaitIdle(context_.getDevice());
    
    cleanupSwapchain();
    
    createSwapchain();
    createImageViews();
    createDepthResources();
    
    camera_.updateAspect(static_cast<float>(swapchainExtent_.width) / swapchainExtent_.height);
}

void RenderingFramework::cleanupSwapchain() {
    if (depthImageView_ != VK_NULL_HANDLE) {
        vkDestroyImageView(context_.getDevice(), depthImageView_, nullptr);
        depthImageView_ = VK_NULL_HANDLE;
    }
    if (depthImage_ != VK_NULL_HANDLE) {
        vkDestroyImage(context_.getDevice(), depthImage_, nullptr);
        depthImage_ = VK_NULL_HANDLE;
    }
    if (depthImageMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(context_.getDevice(), depthImageMemory_, nullptr);
        depthImageMemory_ = VK_NULL_HANDLE;
    }
    
    for (auto imageView : swapchainImageViews_) {
        vkDestroyImageView(context_.getDevice(), imageView, nullptr);
    }
    
    vkDestroySwapchainKHR(context_.getDevice(), swapchain_, nullptr);
}

void RenderingFramework::cleanup() {
    waitIdle();
    
    cleanupSwapchain();
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(context_.getDevice(), renderFinishedSemaphores_[i], nullptr);
        vkDestroySemaphore(context_.getDevice(), imageAvailableSemaphores_[i], nullptr);
        vkDestroyFence(context_.getDevice(), inFlightFences_[i], nullptr);
    }
    
    destroyBuffer(sceneUBO_, context_.getDevice());
    renderingPipeline_.cleanup();
    
    vkDestroySurfaceKHR(context_.getInstance(), surface_, nullptr);
    glfwDestroyWindow(window_);
    glfwTerminate();
}

// GLFW callbacks
void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<RenderingFramework*>(glfwGetWindowUserPointer(window));
    app->onFramebufferResize(width, height);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    auto app = reinterpret_cast<RenderingFramework*>(glfwGetWindowUserPointer(window));
    app->onMouseMove(xpos, ypos);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto app = reinterpret_cast<RenderingFramework*>(glfwGetWindowUserPointer(window));
    app->onMouseScroll(xoffset, yoffset);
}