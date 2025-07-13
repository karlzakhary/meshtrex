#pragma once

#include "common.h"
#include "vulkan_context.h"
#include "extractionOutput.h"
#include "renderingPipeline.h"
#include <GLFW/glfw3.h>
#include <vector>

// Scene data for UBO
struct SceneData {
    glm::mat4 viewProjectionMatrix;
    glm::mat4 modelMatrix;
    glm::vec3 cameraPos_world;
    float pad;
};

// Camera for navigation
class Camera {
public:
    Camera(float fov, float aspect, float near, float far);
    
    void updateAspect(float aspect);
    void setPosition(const glm::vec3& pos);
    void lookAt(const glm::vec3& target, const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f));
    
    // Mouse/keyboard controls
    void rotate(float yaw, float pitch);
    void move(const glm::vec3& direction);
    void zoom(float amount);
    
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::vec3 getPosition() const { return position_; }
    
private:
    glm::vec3 position_;
    glm::vec3 front_;
    glm::vec3 up_;
    glm::vec3 right_;
    
    float yaw_;
    float pitch_;
    float fov_;
    float aspect_;
    float nearPlane_;
    float farPlane_;
    
    void updateVectors();
};

// Swapchain management
struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class RenderingFramework {
public:
    RenderingFramework(VulkanContext& context, uint32_t width, uint32_t height);
    ~RenderingFramework();
    
    // Disable copy
    RenderingFramework(const RenderingFramework&) = delete;
    RenderingFramework& operator=(const RenderingFramework&) = delete;
    
    // Main rendering
    void renderFrame(const ExtractionOutput& extractionOutput);
    bool shouldClose() const;
    void waitIdle();
    
    // Input handling
    void processInput();
    void onMouseMove(double xpos, double ypos);
    void onMouseScroll(double xoffset, double yoffset);
    void onFramebufferResize(int width, int height);
    
    // Getters
    GLFWwindow* getWindow() const { return window_; }
    Camera& getCamera() { return camera_; }
    
private:
    // Core objects
    VulkanContext& context_;
    GLFWwindow* window_;
    VkSurfaceKHR surface_;
    Camera camera_;
    
    // Swapchain
    VkSwapchainKHR swapchain_;
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    VkFormat swapchainImageFormat_;
    VkExtent2D swapchainExtent_;
    
    // Depth buffer
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    
    // Rendering pipeline
    RenderingPipeline renderingPipeline_;
    
    // Scene data
    Buffer sceneUBO_;
    SceneData sceneData_;
    
    // Synchronization
    static const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    
    // Command buffers
    std::vector<VkCommandBuffer> commandBuffers_;
    
    // State
    bool framebufferResized_ = false;
    float lastX_ = 0.0f;
    float lastY_ = 0.0f;
    bool firstMouse_ = true;
    
    // Initialization
    void createWindow(uint32_t width, uint32_t height);
    void createSurface();
    void createSwapchain();
    void createImageViews();
    void createDepthResources();
    void createRenderingPipeline();
    void createSceneUBO();
    void createSyncObjects();
    void createCommandBuffers();
    
    // Swapchain management
    SwapchainSupportDetails querySwapchainSupport();
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void recreateSwapchain();
    void cleanupSwapchain();
    
    // Rendering
    void updateSceneUBO();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, const ExtractionOutput& extractionOutput);
    
    // Cleanup
    void cleanup();
};

// GLFW callbacks
void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);