#pragma once

#include "common.h"
#include "vulkan_context.h"
#include "buffer.h"

class DGCGenerationPipeline {
public:
    DGCGenerationPipeline() = default;
    ~DGCGenerationPipeline();

    bool setup(VkDevice device);
    void cleanup();

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    
    bool createDescriptorSetLayout();
    bool createPipelineLayout();
    bool createComputePipeline();
    bool createDescriptorPool();
    bool allocateDescriptorSet();
};

// Function to generate indirect draw commands on GPU
void generateIndirectDrawCommands(VkCommandBuffer cmd,
                                 VkDevice device,
                                 const Buffer& activeBlockCountBuffer,
                                 const Buffer& indirectDrawBuffer);