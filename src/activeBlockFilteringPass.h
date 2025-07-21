#pragma once

#include "common.h"

#include "buffer.h"
#include "shaders.h"


class VulkanContext; // Forward declaration

class ActiveBlockFilteringPass {
public:
    ActiveBlockFilteringPass(const VulkanContext& context, const char* shaderPath);
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

    [[nodiscard]] VkPipelineLayout getPipelineLayout() const { return pipelineLayout_; }
    [[nodiscard]] VkPipeline getPipeline() const { return pipeline_; }
    [[nodiscard]] VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout_; }
    [[nodiscard]] VkShaderModule getShaderModule() const { return computeShader_.module; }
    
    // Transfer ownership of resources to prevent destruction
    void transferResourceOwnership() {
        pipeline_ = VK_NULL_HANDLE;
        pipelineLayout_ = VK_NULL_HANDLE;
        descriptorSetLayout_ = VK_NULL_HANDLE;
        computeShader_.module = VK_NULL_HANDLE;
    }

private:
    void createPipelineLayout();
    void createPipeline(const char* shaderPath);

    const VulkanContext& context_;
    VkDevice device_;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    Shader computeShader_{};
};