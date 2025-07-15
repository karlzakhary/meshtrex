#pragma once

#include "common.h"
#include "shaders.h"

// Encapsulates the graphics pipeline for rendering meshlets.
class RenderingPipeline {
public:
    RenderingPipeline() = default;
    ~RenderingPipeline();

    // No copy/move
    RenderingPipeline(const RenderingPipeline&) = delete;
    RenderingPipeline& operator=(const RenderingPipeline&) = delete;

    void setup(VkDevice device, VkFormat colorFormat, VkFormat depthFormat, VkPipelineLayout layout);
    void cleanup();

    VkPipeline pipeline_ = VK_NULL_HANDLE;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    Shader meshShader_{};
    Shader fragShader_{};
    Shader taskShader_{};
};