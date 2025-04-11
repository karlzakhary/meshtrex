#include "common.h"
#include "vulkan_utils.h"
#include "vulkan_context.h" // Include the class declaration

// --- Required Includes for Implementation ---
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For logging (optional)
#include <cassert>   // For assert()
#include <cstring>   // For C-style string functions if needed by helpers

// Include Volk header (needed for loading functions)
#include <volk.h>

// Include your existing device function declarations
#include "device.h" // Assumes functions like createInstance are declared here


// --- Constructor Implementation ---
VulkanContext::VulkanContext(bool requestMeshShading)
 // Initialize members to default values (good practice, even if set later)
 : instance_(VK_NULL_HANDLE),
   debugCallback_(VK_NULL_HANDLE),
   physicalDevice_(VK_NULL_HANDLE),
   device_(VK_NULL_HANDLE),
   queue_(VK_NULL_HANDLE),
   graphicsFamilyIndex_(VK_QUEUE_FAMILY_IGNORED),
   commandPool_(VK_NULL_HANDLE),
   memoryProperties_{}
{
    // --- 1. Initialize Volk ---
    VkResult volkInitResult = volkInitialize();
    if (volkInitResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to initialize Volk");
    }

    // --- 2. Create Instance (using external function) ---
    instance_ = createInstance(); // Calls function from device.cpp
    if (!instance_) {
        throw std::runtime_error("Failed to create Vulkan Instance (check device.cpp logs)");
    }
    volkLoadInstanceOnly(instance_);

    // --- 3. Register Debug Callback (using external function) ---
    debugCallback_ = registerDebugCallback(instance_); // Calls function from device.cpp
    // No explicit error check needed here if registerDebugCallback handles it

    // --- 4. Pick Physical Device (using external function) ---
    VkPhysicalDevice physicalDevices[16]; // Max 16 physical devices
    uint32_t physicalDeviceCount = std::size(physicalDevices);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &physicalDeviceCount, physicalDevices));
    if (physicalDeviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    physicalDevice_ = pickPhysicalDevice(physicalDevices, physicalDeviceCount); // Calls function from device.cpp
    if (!physicalDevice_) {
        throw std::runtime_error("Failed to find a suitable GPU (check device.cpp logs)");
    }

    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr,
                                                  &extensionCount, nullptr));

    std::vector<VkExtensionProperties> extensions(extensionCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
        physicalDevice_, nullptr, &extensionCount, extensions.data()));

    // --- Optional: Check timestamp support ---
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physicalDevice_, &props);
    assert(props.limits.timestampComputeAndGraphics); // Or handle failure gracefully

    // --- 5. Get Graphics Queue Family Index (using external function) ---
    graphicsFamilyIndex_ = getGraphicsFamilyIndex(physicalDevice_); // Calls function from device.cpp
    if (graphicsFamilyIndex_ == VK_QUEUE_FAMILY_IGNORED) {
         throw std::runtime_error("Failed to find a graphics queue family!");
    }

    // --- 6. Create Logical Device (using external function) ---
    // Note: Add logic here to properly check if mesh shading is actually supported
    // by the chosen physicalDevice_ before passing `true` if needed.
    device_ = createDevice(instance_, physicalDevice_, graphicsFamilyIndex_, requestMeshShading); // Calls function from device.cpp
    if (!device_) {
        throw std::runtime_error("Failed to create logical device (check device.cpp logs)");
    }
    volkLoadDevice(device_);

    vkCmdBeginRendering = vkCmdBeginRenderingKHR;
    vkCmdEndRendering = vkCmdEndRenderingKHR;
    vkCmdPipelineBarrier2 = vkCmdPipelineBarrier2KHR;

    // --- 7. Get Device Queue ---
    vkGetDeviceQueue(device_, graphicsFamilyIndex_, 0, &queue_);
    if (!queue_) {
        // Should not happen if device creation succeeded for this queue family
        throw std::runtime_error("Failed to get device queue!");
    }

    // --- 8. Get Memory Properties ---
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);

    // --- 9. Create Command Pool (using internal helper) ---
    commandPool_ = createCommandPool(device_, graphicsFamilyIndex_);
     if (!commandPool_) {
        // createCommandPoolInternal uses VK_CHECK, so this might be redundant
        // depending on VK_CHECK's behavior (e.g., if it throws)
        throw std::runtime_error("Failed to create command pool!");
    }

    // Optional: Log success
    std::cout << "Vulkan Context Initialized Successfully." << std::endl;
}

// --- Destructor Implementation ---
VulkanContext::~VulkanContext() {
    // Destroy resources in reverse order of creation

    // Destroy device-owned resources first
    if (device_ != VK_NULL_HANDLE) {
        // Wait for the device to be idle before destroying resources
        // It's crucial to ensure no GPU operations are pending.
        // This should ideally be done *before* calling the destructor,
        // but adding it here provides some safety. Consider where vkDeviceWaitIdle
        // best fits in your application's shutdown sequence.
        // vkDeviceWaitIdle(device_); // Uncomment if appropriate here

        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
            commandPool_ = VK_NULL_HANDLE; // Good practice to null handles after destruction
        }
        // Queue is implicitly destroyed with the device
        queue_ = VK_NULL_HANDLE;

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    // Destroy instance-owned resources
    if (instance_ != VK_NULL_HANDLE) {
        // Check if the debug callback exists AND the destruction function is loaded
        if (debugCallback_ != VK_NULL_HANDLE) {
            // Make sure vkDestroyDebugReportCallbackEXT is loaded (Volk should handle this)
            PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallback =
               reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                    vkGetInstanceProcAddr(instance_,
                                          "vkDestroyDebugReportCallbackEXT"));
            if (vkDestroyDebugReportCallback) {
               vkDestroyDebugReportCallback(instance_, debugCallback_, nullptr);
            } else {
                // Log a warning if the function couldn't be loaded but the handle exists
                 std::cerr << "Warning: Could not load vkDestroyDebugReportCallbackEXT to destroy debug callback." << std::endl;
            }
             debugCallback_ = VK_NULL_HANDLE;
        }

        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    // Physical device handle doesn't need destruction
    physicalDevice_ = VK_NULL_HANDLE;
    volkFinalize();
    // Optional: Log destruction
    std::cout << "Vulkan Context Destroyed." << std::endl;
}