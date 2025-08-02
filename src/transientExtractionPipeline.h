#pragma once

#include "common.h"
#include <string>
#include <vector>
#include "shaders.h"

class TransientExtractionPipeline {
public:
    TransientExtractionPipeline() = default;
    ~TransientExtractionPipeline();

    TransientExtractionPipeline(const TransientExtractionPipeline&) = delete;
    TransientExtractionPipeline& operator=(const TransientExtractionPipeline&) = delete;
    TransientExtractionPipeline(TransientExtractionPipeline&& other) noexcept;
    TransientExtractionPipeline& operator=(TransientExtractionPipeline&& other) noexcept;

    bool setup(
        VkDevice device,
        VkFormat colorFormat,
        VkFormat depthFormat,
        uint32_t blockX,
        uint32_t blockY,
        uint32_t blockZ,
        bool pmb = true
    );

    bool setupWithShaders(
        VkDevice device,
        VkFormat colorFormat,
        VkFormat depthFormat,
        uint32_t blockX,
        uint32_t blockY,
        uint32_t blockZ,
        bool pmb,
        const char* taskShaderPath,
        const char* meshShaderPath,
        const char* fragShaderPath
    );

    void cleanup();

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

    VkShaderModule getTaskShaderModule() const { return taskShader_.module; }
    VkShaderModule getMeshShaderModule() const { return meshShader_.module; }
    VkDescriptorPool getDescriptorPool() const { return descriptorPool_; }

    void transferResourceOwnership() {
        pipeline_ = VK_NULL_HANDLE;
        pipelineLayout_ = VK_NULL_HANDLE;
        descriptorSetLayout_ = VK_NULL_HANDLE;
        descriptorPool_ = VK_NULL_HANDLE;
        descriptorSet_ = VK_NULL_HANDLE;
        taskShader_.module = VK_NULL_HANDLE;
        meshShader_.module = VK_NULL_HANDLE;
        fragShader_.module = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

private:
    Shader taskShader_{};
    Shader meshShader_{};
    Shader fragShader_{};

    void releaseResources();
    void createPipelineLayout();
    void createTransientExtractionGraphicsPipeline(
        VkFormat colorFormat,
        VkFormat depthFormat,
        uint32_t blockX,
        uint32_t blockY,
        uint32_t blockZ,
        bool pmb
    );
    void createDescriptorPool();
    void allocateDescriptorSets();
};