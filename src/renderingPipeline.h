#pragma once

#include "common.h"
#include "resources.h"
#include "shaders.h"
#include <string>
#include <vector>

class RenderingPipeline {
public:
    RenderingPipeline() = default;
    ~RenderingPipeline();

    RenderingPipeline(const RenderingPipeline&) = delete;
    RenderingPipeline& operator=(const RenderingPipeline&) = delete;
    RenderingPipeline(RenderingPipeline&& other) noexcept;
    RenderingPipeline& operator=(RenderingPipeline&& other) noexcept;

    // Setup: Takes VS/FS shaders and vertex input layout
    bool setup(
        VkDevice device,
        VkFormat colorFormat,
        VkFormat depthFormat,
        VkSampleCountFlagBits msaaSamples
    );

    void cleanup();

    // --- Public Members ---
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

private:
    // Vertex and Fragment shaders
    Shader taskShader_{};
    Shader meshShader_{};
    Shader fragmentShader_{};

    void releaseResources();
    void createPipelineLayout();
};