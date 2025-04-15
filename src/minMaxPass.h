#pragma once

#include "common.h"
#include "shaders.h"

#include <tuple>

class VulkanContext; // Forward declaration

class MinMaxPass {
public:
    MinMaxPass(const VulkanContext& context, const char* shaderPath);
    ~MinMaxPass();

    MinMaxPass(const MinMaxPass&) = delete;
    MinMaxPass& operator=(const MinMaxPass&) = delete;
    MinMaxPass(MinMaxPass&&) = delete;
    MinMaxPass& operator=(MinMaxPass&&) = delete;

    void recordDispatch(VkCommandBuffer cmd,
                        VkImageView inputVolumeView,
                        VkImageView outputMinMaxView,
                        const PushConstants& pushConstants);

    VkPipelineLayout getPipelineLayout() const { return pipelineLayout_; }

private:
    void createPipelineLayout();
    void createPipeline(const char* shaderPath);

    const VulkanContext& context_; // Reference to the main Vulkan context
    VkDevice device_; // Cache device handle for convenience

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    Shader computeShader_{};
};