#pragma once

#include "common.h"
#include "extractionPipeline.h"
#include <string>
#include <vector>

class DensityExtractionPipeline {
public:
    // Constructor/Destructor for RAII
    DensityExtractionPipeline() = default;
    ~DensityExtractionPipeline();

    // Prevent copying
    DensityExtractionPipeline(const DensityExtractionPipeline&) = delete;
    DensityExtractionPipeline& operator=(const DensityExtractionPipeline&) = delete;
    
    // Allow moving
    DensityExtractionPipeline(DensityExtractionPipeline&& other) noexcept;
    DensityExtractionPipeline& operator=(DensityExtractionPipeline&& other) noexcept;

    // Setup method to create all three density-specific pipelines
    bool setup(
        VkDevice device,
        VkFormat colorFormat,       // Target color attachment format
        VkFormat depthFormat,       // Target depth attachment format
        uint32_t blockX,            // Block Dimension in X
        uint32_t blockY,            // Block Dimension in Y
        uint32_t blockZ             // Block Dimension in Z
    );

    // Explicit cleanup method
    void cleanup();

    // Get pipelines for each density class
    VkPipeline getSparsePipeline() const { return sparsePipeline_.pipeline_; }
    VkPipeline getMediumPipeline() const { return mediumPipeline_.pipeline_; }
    VkPipeline getDensePipeline() const { return densePipeline_.pipeline_; }
    
    // Get shared pipeline layout (all variants use the same layout)
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout_; }
    
    // Get descriptor set layout
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout_; }
    
    // Transfer ownership of resources to prevent destruction
    void transferResourceOwnership();

private:
    VkDevice device_ = VK_NULL_HANDLE;
    
    // Shared resources
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    
    // Individual pipelines for each density class
    ExtractionPipeline sparsePipeline_;
    ExtractionPipeline mediumPipeline_;
    ExtractionPipeline densePipeline_;
    
    // Helper to release resources safely
    void releaseResources();
};