#pragma once

#include "common.h"
#include "shaders.h"
#include "streamingSystem.h" // For PageCoord
#include "streamingShaderInterface.h"

#include <tuple>

class VulkanContext; // Forward declaration

class MinMaxPass {
public:
    MinMaxPass(const VulkanContext& context, const char* leafShaderPath, const char* octreeShaderPath);
    
    // Constructor with streaming shader support
    MinMaxPass(const VulkanContext& context, 
               const char* leafShaderPath, 
               const char* octreeShaderPath,
               const char* streamingLeafShaderPath,
               const char* streamingOctreeShaderPath);
               
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

    // Streaming versions that work with sparse atlas and page table
    void recordStreamingLeafDispatch(VkCommandBuffer cmd,
                                    VkImageView volumeAtlasView,
                                    VkSampler volumeSampler,
                                    const Buffer& pageTableBuffer,
                                    VkImageView minMaxView,
                                    const StreamingMinMaxPushConstants& pushConstants,
                                    const PageCoord& pageCoord);

    void recordStreamingOctreeDispatch(VkCommandBuffer cmd,
                                      VkImageView srcView,
                                      VkExtent3D srcExtent,
                                      VkImageView dstView,
                                      VkExtent3D dstExtent,
                                      const Buffer& pageTableBuffer,
                                      const PageCoord& pageCoord);

    VkPipelineLayout getLeafPipelineLayout() const { return leafPipelineLayout_; }
    VkPipelineLayout getOctreePipelineLayout() const { return octreePipelineLayout_; }
    VkPipelineLayout getStreamingLeafPipelineLayout() const { return streamingLeafPipelineLayout_; }
    VkPipelineLayout getStreamingOctreePipelineLayout() const { return streamingOctreePipelineLayout_; }

private:
    void createLeafPipelineLayout();
    void createLeafPipeline(const char* leafShaderPath);
    void createOctreePipelineLayout();
    void createOctreePipeline(const char* octreeShaderPath);
    void createStreamingPipelineLayouts();
    void createStreamingPipelines();

    const VulkanContext& context_; // Reference to the main Vulkan context
    VkDevice device_; // Cache device handle for convenience

    // Regular pipelines
    VkDescriptorSetLayout leafDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout leafPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout octreeDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout octreePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline leafPipeline_ = VK_NULL_HANDLE;
    VkPipeline octreePipeline_ = VK_NULL_HANDLE;
    Shader leafMinMaxComputeShader_{};
    Shader octreeMinMaxComputeShader_{};
    
    // Streaming pipelines
    VkDescriptorSetLayout streamingLeafDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout streamingLeafPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout streamingOctreeDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout streamingOctreePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline streamingLeafPipeline_ = VK_NULL_HANDLE;
    VkPipeline streamingOctreePipeline_ = VK_NULL_HANDLE;
    Shader streamingLeafMinMaxComputeShader_{};
    Shader streamingOctreeMinMaxComputeShader_{};
};