#pragma once

#include "common.h"
#include "vulkan_context.h"
#include "extractionOutput.h"
#include "renderingPipeline.h"
#include "swapchain.h"
#include "image.h"
#include <vector>

// Forward declare GLFWwindow
struct GLFWwindow;

// Manages the window, swapchain, and main render loop.
class RenderingManager {
public:
    RenderingManager(VulkanContext& context, uint32_t width = 1280, uint32_t height = 720, const char* title = "MeshTrex Renderer");
    ~RenderingManager();

    // No copy/move
    RenderingManager(const RenderingManager&) = delete;
    RenderingManager& operator=(const RenderingManager&) = delete;

    void render(const ExtractionOutput& extractionOutput);
    void handleCameraInput(float deltaTime);

private:
    void initWindow(uint32_t width, uint32_t height, const char* title);
    void createSwapchainResources();
    void cleanupSwapchainResources();
    uint32_t readCounterFromBuffer(const Buffer& counterBuffer);

    VulkanContext& context_;
    GLFWwindow* window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    
    Swapchain swapchain_;
    Image depthImage_;
    VkFormat swapchainFormat_;
    VkFormat depthFormat_;
    
    RenderingPipeline renderingPipeline_;
    VkPipelineLayout pipelineLayout_;

    std::vector<VkImageView> swapchainImageViews_;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;

    // Simple camera state
glm::vec3 cameraPos_ = glm::vec3(-1.f, -1.f, -2.f); // Closer initial position
    glm::vec3 cameraTarget_ = glm::vec3(128.f, 128.f, 128.f);
    glm::vec3 cameraUp_ = glm::vec3(0.0f, 1.0f, 0.0f);
    double lastMouseX_ = 0, lastMouseY_ = 0;
    float lastFrameTime_ = 0.0f;
};