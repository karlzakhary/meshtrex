#include "common.h"

#include "device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "config.h"

// Validation is enabled by default in Debug
#ifndef NDEBUG
#define KHR_VALIDATION 1
#endif

// Synchronization validation is disabled by default in Debug since it's rather
// slow
#define SYNC_VALIDATION CONFIG_SYNCVAL

static bool isLayerSupported(const char *name)
{
    uint32_t propertyCount = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&propertyCount, 0));

    std::vector<VkLayerProperties> properties(propertyCount);
    VK_CHECK(
        vkEnumerateInstanceLayerProperties(&propertyCount, properties.data()));

    for (uint32_t i = 0; i < propertyCount; ++i)
        if (strcmp(name, properties[i].layerName) == 0) return true;

    return false;
}

DGCSupport queryDGCSupport(VkPhysicalDevice device) {
    // Check if the extension is available
    uint32_t extensionCount = 0;
    DGCSupport dgcSupport {};
    
    
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
    
    for (const auto& extension : extensions) {
        if (strcmp(extension.extensionName, VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME) == 0) {
            dgcSupport.dgcSupported = true;
        }
        if (strcmp(extension.extensionName, VK_NV_DEVICE_GENERATED_COMMANDS_COMPUTE_EXTENSION_NAME) == 0) {
            dgcSupport.dgcComputeSupported = true;
        }
    }
    
    if (!dgcSupport.dgcSupported) {
        return dgcSupport;
    }
    
    // Query features and properties
    dgcSupport.dgcFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV;
    
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &dgcSupport.dgcFeatures;
    
    vkGetPhysicalDeviceFeatures2(device, &features2);
    
    dgcSupport.dgcProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_NV;
    
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &dgcSupport.dgcProperties;
    
    vkGetPhysicalDeviceProperties2(device, &properties2);

    return dgcSupport;
}


VkInstance createInstance()
{
    assert(volkGetInstanceVersion() >= VK_API_VERSION_1_3);

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;

#if KHR_VALIDATION || SYNC_VALIDATION
    const char *debugLayers[] = {
        "VK_LAYER_KHRONOS_validation",
    };

    if (isLayerSupported("VK_LAYER_KHRONOS_validation")) {
        createInfo.ppEnabledLayerNames = debugLayers;
        createInfo.enabledLayerCount = std::size(debugLayers);
        printf("Enabled Vulkan validation layers (sync validation %s)\n",
               SYNC_VALIDATION ? "enabled" : "disabled");
    } else {
        printf("Warning: Vulkan debug layers are not available\n");
    }
    VkValidationFeaturesEXT validationFeatures = {};
    validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;

    VkValidationFeatureEnableEXT enabledFeatures[] = {
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
    };

    validationFeatures.enabledValidationFeatureCount = 1;
    validationFeatures.pEnabledValidationFeatures = enabledFeatures;
#if SYNC_VALIDATION
    VkValidationFeatureEnableEXT enabledValidationFeatures[] = {
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
    };

    VkValidationFeaturesEXT validationFeatures = {
        VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    validationFeatures.enabledValidationFeatureCount =
        sizeof(enabledValidationFeatures) /
        sizeof(enabledValidationFeatures[0]);
    validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;

    createInfo.pNext = &validationFeatures;
#endif
#endif

    const char *extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_XLIB_KHR
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
        VK_EXT_METAL_SURFACE_EXTENSION_NAME,
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
#ifndef NDEBUG
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
#endif
    };

    createInfo.ppEnabledExtensionNames = extensions;
    createInfo.enabledExtensionCount = std::size(extensions);
    createInfo.pNext = &validationFeatures;
#ifdef VK_USE_PLATFORM_METAL_EXT
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstance instance = nullptr;
    VK_CHECK(vkCreateInstance(&createInfo, 0, &instance));

    return instance;
}

static VkBool32 VKAPI_CALL debugReportCallback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location, int32_t messageCode,
    const char *pLayerPrefix, const char *pMessage, void *pUserData)
{
    if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) return VK_FALSE;

    const char *type = (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)     ? "ERROR"
                       : (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) ? "WARNING"
                                                                   : "INFO";

    char message[4096];
    snprintf(message, COUNTOF(message), "%s: %s\n", type, pMessage);

    printf("%s", message);

    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
        assert(!"Validation error encountered!");

    return VK_FALSE;
}

VkDebugReportCallbackEXT registerDebugCallback(VkInstance instance)
{
    if (!vkCreateDebugReportCallbackEXT) return nullptr;

    VkDebugReportCallbackCreateInfoEXT createInfo = {
        VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT};
    createInfo.flags = VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_INFORMATION_BIT_EXT | VK_DEBUG_REPORT_DEBUG_BIT_EXT |
                       VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
                       VK_DEBUG_REPORT_ERROR_BIT_EXT;
    createInfo.pfnCallback = debugReportCallback;

    VkDebugReportCallbackEXT callback = 0;
    VK_CHECK(
        vkCreateDebugReportCallbackEXT(instance, &createInfo, 0, &callback));

    return callback;
}

uint32_t getGraphicsFamilyIndex(VkPhysicalDevice physicalDevice)
{
    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, 0);

    std::vector<VkQueueFamilyProperties> queues(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount,
                                             queues.data());

    for (uint32_t i = 0; i < queueCount; ++i)
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;

    return VK_QUEUE_FAMILY_IGNORED;
}

VkPhysicalDevice pickPhysicalDevice(VkPhysicalDevice *physicalDevices,
                                    uint32_t physicalDeviceCount)
{
    VkPhysicalDevice preferred = nullptr;
    VkPhysicalDevice fallback = nullptr;

    const char *ngpu = getenv("NGPU");

    for (uint32_t i = 0; i < physicalDeviceCount; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevices[i], &props);

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

        printf("GPU%d: %s\n", i, props.deviceName);

        uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevices[i]);
        if (familyIndex == VK_QUEUE_FAMILY_IGNORED) continue;

        if (props.apiVersion < VK_API_VERSION_1_2) continue;

        if (ngpu && atoi(ngpu) == i) {
            preferred = physicalDevices[i];
        }

        if (!preferred &&
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            preferred = physicalDevices[i];
        }

        if (!fallback) {
            fallback = physicalDevices[i];
        }
    }

    VkPhysicalDevice result = preferred ? preferred : fallback;

    if (result) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(result, &props);

        printf("Selected GPU %s\n", props.deviceName);
    } else {
        printf("ERROR: No GPUs found\n");
    }

    return result;
}

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice,
                      uint32_t familyIndex, bool meshShadingSupported)
{
    // First, enumerate supported device extensions
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());
    
    // Helper to check if extension is available
    auto isExtensionAvailable = [&](const char* extensionName) {
        for (const auto& ext : availableExtensions) {
            if (strcmp(ext.extensionName, extensionName) == 0) {
                return true;
            }
        }
        return false;
    };

    float queuePriorities[] = {1.0f};

    VkDeviceQueueCreateInfo queueInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = familyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = queuePriorities;

    std::vector<const char *> extensions = {
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
#ifdef __APPLE__
        "VK_KHR_portability_subset",
#endif
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
        VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME,
    };
    
    // Only add compute DGC extension if available
    if (isExtensionAvailable(VK_NV_DEVICE_GENERATED_COMMANDS_COMPUTE_EXTENSION_NAME)) {
        extensions.push_back(VK_NV_DEVICE_GENERATED_COMMANDS_COMPUTE_EXTENSION_NAME);
        printf("VK_NV_device_generated_commands_compute extension is available\n");
    } else {
        printf("Warning: VK_NV_device_generated_commands_compute extension is not available\n");
    }

    if (meshShadingSupported)
        extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

    VkPhysicalDeviceFeatures2 features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.features.multiDrawIndirect = true;
#ifndef __APPLE__
    features.features.pipelineStatisticsQuery = true;
#endif
    features.features.shaderInt16 = true;
    features.features.shaderInt64 = true;
    features.features.samplerAnisotropy = true;
    features.features.shaderStorageImageWriteWithoutFormat = true;
    features.features.sparseBinding = true;
    features.features.sparseResidencyImage3D = true;

    VkPhysicalDeviceVulkan11Features features11 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    features11.storageBuffer16BitAccess = true;
    features11.shaderDrawParameters = true;

    VkPhysicalDeviceVulkan12Features features12 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
#ifndef __APPLE__
    features12.drawIndirectCount = true;
#endif
    features12.storageBuffer8BitAccess = true;
    features12.uniformAndStorageBuffer8BitAccess = true;
    features12.shaderFloat16 = true;
    features12.shaderInt8 = true;
#ifndef __APPLE__
    features12.samplerFilterMinmax = true;
#endif

    features12.scalarBlockLayout = true;

    features12.descriptorIndexing = true;
    features12.shaderSampledImageArrayNonUniformIndexing = true;
    features12.descriptorBindingSampledImageUpdateAfterBind = true;
    features12.descriptorBindingUpdateUnusedWhilePending = true;
    features12.descriptorBindingPartiallyBound = true;
    features12.descriptorBindingVariableDescriptorCount = true;
    features12.runtimeDescriptorArray = true;

    VkPhysicalDeviceVulkan13Features features13 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = true;
    features13.synchronization2 = true;
    features13.maintenance4 = true;

    // This will only be used if meshShadingSupported=true (see below)
    VkPhysicalDeviceMeshShaderFeaturesEXT featuresMesh = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    featuresMesh.taskShader = true;
    featuresMesh.meshShader = true;

    VkPhysicalDeviceDynamicRenderingFeatures featuresDynamicRendering = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    featuresDynamicRendering.dynamicRendering = true;

    VkPhysicalDeviceSynchronization2Features featuresSynchronization2 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    featuresSynchronization2.synchronization2 = true;

    VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV featuresDGC = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV};
    featuresDGC.deviceGeneratedCommands = true;

    VkDeviceCreateInfo createInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;

    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());

    createInfo.pNext = &features;
    features.pNext = &features11;
    features11.pNext = &features12;
#ifndef __APPLE__
    features12.pNext = &features13;

    void **ppNext = &features13.pNext;
#else
    features12.pNext = &featuresDynamicRendering;
    featuresDynamicRendering.pNext = &featuresSynchronization2;

    void **ppNext = &featuresSynchronization2.pNext;
#endif
    if (meshShadingSupported) {
        *ppNext = &featuresMesh;
        ppNext = &featuresMesh.pNext;
    }

    // Add DGC features to the chain
    *ppNext = &featuresDGC;
    ppNext = &featuresDGC.pNext;

    VkDevice device = nullptr;
    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));

    return device;
}
