#pragma once

#include "common.h"

#include "buffer.h"
#include "shaders.h"
#include "streamingSystem.h" // For PageCoord


class VulkanContext; // Forward declaration

class ActiveBlockFilteringPass {
public:
    ActiveBlockFilteringPass(const VulkanContext& context, const char* shaderPath);
    
    // Constructor with streaming support
    ActiveBlockFilteringPass(const VulkanContext& context, 
                           const char* regularShaderPath,
                           const char* streamingShaderPath);
    ~ActiveBlockFilteringPass();

    // Non-copyable/movable
    ActiveBlockFilteringPass(const ActiveBlockFilteringPass&) = delete;
    ActiveBlockFilteringPass& operator=(const ActiveBlockFilteringPass&) = delete;
    ActiveBlockFilteringPass(ActiveBlockFilteringPass&&) = delete;
    ActiveBlockFilteringPass& operator=(ActiveBlockFilteringPass&&) = delete;

    void recordDispatch(VkCommandBuffer cmd,
                        VkImageView minMaxImageView,
                        VkSampler sampler,
                        const Buffer& compactedBlockIdBuffer, // Output
                        const Buffer& activeBlockCountBuffer, // Output (atomic counter)
                        const PushConstants& pushConstants) const;
    
    // Streaming version with page table support
    void recordStreamingDispatch(VkCommandBuffer cmd,
                                VkImageView minMaxImageView,
                                VkSampler sampler,
                                const Buffer& pageTableBuffer,
                                const Buffer& compactedBlockIdBuffer,
                                const Buffer& activeBlockCountBuffer,
                                const PushConstants& pushConstants,
                                const PageCoord& pageCoord) const;

    [[nodiscard]] VkPipelineLayout getPipelineLayout() const { return pipelineLayout_; }
    [[nodiscard]] VkPipelineLayout getStreamingPipelineLayout() const { return streamingPipelineLayout_; }

private:
    void createPipelineLayout();
    void createPipeline(const char* shaderPath);
    void createStreamingPipelineLayout();
    void createStreamingPipeline(const char* shaderPath);

    const VulkanContext& context_;
    VkDevice device_;

    // Regular pipeline resources
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    Shader computeShader_{};
    
    // Streaming pipeline resources
    VkDescriptorSetLayout streamingDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout streamingPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline streamingPipeline_ = VK_NULL_HANDLE;
    Shader streamingComputeShader_{};
};