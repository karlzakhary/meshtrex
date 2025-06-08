#pragma once

#include "common.h"

#include <string>
#include <vector>

#include "shaders.h"

class ExtractionPipeline {
public:
    // Constructor/Destructor for RAII
    ExtractionPipeline() = default; // Default constructor
    ~ExtractionPipeline();          // Destructor for cleanup

    // Prevent copying
    ExtractionPipeline(const ExtractionPipeline&) = delete;
    ExtractionPipeline& operator=(const ExtractionPipeline&) = delete;
    // Allow moving (optional but good practice)
    ExtractionPipeline(ExtractionPipeline&& other) noexcept;
    ExtractionPipeline& operator=(ExtractionPipeline&& other) noexcept;


    // Setup method to create all Vulkan objects
    bool setup(
        VkDevice device,
        VkFormat colorFormat,       // Target color attachment format
        VkFormat depthFormat        // Target depth attachment format
    );

    // Explicit cleanup method (alternative or supplement to destructor)
    void cleanup();

    // --- Public Members (Handles needed by external functions) ---
    VkDevice device_ = VK_NULL_HANDLE; // Store the device for cleanup
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE; // Assuming one set

private:
    // Shader modules managed by Shader struct (RAII)
    Shader taskShader_{};
    Shader meshShader_{};

    // Internal helper to release resources safely
    void releaseResources();
    void createPipelineLayout();
    void createExtractionGraphicsPipeline(VkFormat colorFormat,
                                          VkFormat depthFormat);
    void createDescriptorPool();
    void allocateDescriptorSets();
};