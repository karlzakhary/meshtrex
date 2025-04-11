#pragma once

#include <vulkan/vulkan.h> // Include Vulkan types used in the interface
#include <cstdint>         // For uint32_t

// --- Class Definition ---
class VulkanContext {
public:
    // --- Constructor and Destructor ---
    // Constructor takes optional flag for mesh shading request
    explicit VulkanContext(bool requestMeshShading = false);
    // Destructor cleans up all managed resources
    ~VulkanContext();

    // --- Resource Accessors (Getters) ---
    // Defined inline for efficiency as they are simple returns
    VkInstance getInstance() const { return instance_; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }
    VkDevice getDevice() const { return device_; }
    VkQueue getQueue() const { return queue_; }
    uint32_t getGraphicsQueueFamilyIndex() const { return graphicsFamilyIndex_; }
    VkCommandPool getCommandPool() const { return commandPool_; }
    const VkPhysicalDeviceMemoryProperties& getMemoryProperties() const { return memoryProperties_; }

    // --- Disable Copying and Assignment ---
    // Prevents accidental copying of the context and its managed resources
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // --- Optional: Move Semantics (Declare if needed) ---
    // VulkanContext(VulkanContext&& other) noexcept;
    // VulkanContext& operator=(VulkanContext&& other) noexcept;

private:
    // --- Private Vulkan Handles ---
    // These are managed by this class instance
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugReportCallbackEXT debugCallback_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t graphicsFamilyIndex_ = VK_QUEUE_FAMILY_IGNORED;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryProperties_ = {};
};