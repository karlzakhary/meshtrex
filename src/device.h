#pragma once

struct DGCSupport {
    bool dgcSupported = false;
    bool dgcComputeSupported = false;
    VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV dgcFeatures = {};
    VkPhysicalDeviceDeviceGeneratedCommandsPropertiesNV dgcProperties = {};
};

DGCSupport queryDGCSupport(VkPhysicalDevice device);
VkInstance createInstance();
VkDebugReportCallbackEXT registerDebugCallback(VkInstance instance);

uint32_t getGraphicsFamilyIndex(VkPhysicalDevice physicalDevice);
VkPhysicalDevice pickPhysicalDevice(VkPhysicalDevice* physicalDevices, uint32_t physicalDeviceCount);

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t familyIndex, bool meshShadingSupported);
