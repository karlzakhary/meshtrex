#pragma once

#include "common.h"
#include "buffer.h"
#include "resources.h"
#include "shaders.h"
#include <vector>

// Push constants for DGC rendering generation
struct DGCRenderingPushConstants {
    uint32_t taskWorkgroupSize;
};

class DGCRenderingPipeline {
public:
    DGCRenderingPipeline() = default;
    ~DGCRenderingPipeline();

    // Disable copy
    DGCRenderingPipeline(const DGCRenderingPipeline&) = delete;
    DGCRenderingPipeline& operator=(const DGCRenderingPipeline&) = delete;

    // Enable move
    DGCRenderingPipeline(DGCRenderingPipeline&& other) noexcept;
    DGCRenderingPipeline& operator=(DGCRenderingPipeline&& other) noexcept;

    bool setup(VkDevice device);
    void cleanup();

    // Public members
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

private:
    Shader computeShader_;
    
    void releaseResources();
    void createDescriptorSetLayout();
    void createPipelineLayout();
    void createDescriptorPool();
    void allocateDescriptorSet();
    void createComputePipeline();
};

// Helper function to generate indirect draw commands for rendering
void generateRenderingIndirectCommands(
    VkCommandBuffer cmd,
    VkDevice device,
    const Buffer& meshletCountBuffer,
    const Buffer& indirectDrawBuffer,
    uint32_t taskWorkgroupSize = 32
);