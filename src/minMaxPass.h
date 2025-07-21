#pragma once

#include "common.h"
#include "shaders.h"

#include <tuple>

class VulkanContext; // Forward declaration

class MinMaxPass {
public:
    MinMaxPass(const VulkanContext& context, const char* leafShaderPath, const char* octreeShaderPath);
    ~MinMaxPass();

    MinMaxPass(const MinMaxPass&) = delete;
    MinMaxPass& operator=(const MinMaxPass&) = delete;
    MinMaxPass(MinMaxPass&&) = delete;
    MinMaxPass& operator=(MinMaxPass&&) = delete;

    void recordLeafDispatch(VkCommandBuffer cmd,
                        VkImageView inputVolumeView,
                        VkImageView minMaxView,
                        const PushConstants& pushConstants);

    void recordOctreeDispatch(VkCommandBuffer cmd,
                             VkImageView srcView,
                             VkExtent3D srcExtent,
                             VkImageView dstView,
                             VkExtent3D dstExtent);

    VkPipelineLayout getLeafPipelineLayout() const { return leafPipelineLayout_; }
    VkPipelineLayout getOctreePipelineLayout() const { return octreePipelineLayout_; }
    VkPipeline getLeafPipeline() const { return leafPipeline_; }
    VkPipeline getOctreePipeline() const { return octreePipeline_; }
    VkDescriptorSetLayout getLeafDescriptorSetLayout() const { return leafDescriptorSetLayout_; }
    VkDescriptorSetLayout getOctreeDescriptorSetLayout() const { return octreeDescriptorSetLayout_; }
    VkShaderModule getLeafShaderModule() const { return leafMinMaxComputeShader_.module; }
    VkShaderModule getOctreeShaderModule() const { return octreeMinMaxComputeShader_.module; }
    
    // Transfer ownership of resources to prevent destruction
    void transferResourceOwnership() {
        leafPipeline_ = VK_NULL_HANDLE;
        octreePipeline_ = VK_NULL_HANDLE;
        leafPipelineLayout_ = VK_NULL_HANDLE;
        octreePipelineLayout_ = VK_NULL_HANDLE;
        leafDescriptorSetLayout_ = VK_NULL_HANDLE;
        octreeDescriptorSetLayout_ = VK_NULL_HANDLE;
        leafMinMaxComputeShader_.module = VK_NULL_HANDLE;
        octreeMinMaxComputeShader_.module = VK_NULL_HANDLE;
    }

private:
    void createLeafPipelineLayout();
    void createLeafPipeline(const char* leafShaderPath);
    void createOctreePipelineLayout();
    void createOctreePipeline(const char* octreeShaderPath);

    const VulkanContext& context_; // Reference to the main Vulkan context
    VkDevice device_; // Cache device handle for convenience

    VkDescriptorSetLayout leafDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout leafPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout octreeDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout octreePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline leafPipeline_ = VK_NULL_HANDLE;
    VkPipeline octreePipeline_ = VK_NULL_HANDLE;
    Shader leafMinMaxComputeShader_{};
    Shader octreeMinMaxComputeShader_{};
};