#pragma once

#include "common.h"
#include "buffer.h"
#include "densityAnalyzer.h"
#include <vector>

class VulkanContext;
class GPUProfiler;

class DensityDispatcher {
public:
    struct ClassifiedBlocks {
        // Sorted block IDs by classification
        std::vector<uint32_t> emptyBlocks;
        std::vector<uint32_t> sparseBlocks;
        std::vector<uint32_t> mediumBlocks;
        std::vector<uint32_t> denseBlocks;
        
        // Counts for dispatch
        uint32_t sparseCount = 0;
        uint32_t mediumCount = 0;
        uint32_t denseCount = 0;
        
        // Consolidated buffer with all blocks sorted by density
        Buffer consolidatedBuffer;
        uint32_t sparseOffset = 0;  // Offset in consolidatedBuffer
        uint32_t mediumOffset = 0;
        uint32_t denseOffset = 0;
        
        void cleanup(VkDevice device);
    };
    
    // Classify blocks and upload to GPU buffers
    ClassifiedBlocks classifyAndUpload(
        const std::vector<DensityAnalyzer::BlockDensity>& densities,
        const VulkanContext& context);
    
    // Record density-based extraction commands
    void recordDensityBasedExtraction(
        VkCommandBuffer cmd,
        const ClassifiedBlocks& blocks,
        const VulkanContext& context,
        VkPipeline sparsePipeline,
        VkPipeline mediumPipeline,
        VkPipeline densePipeline,
        VkPipelineLayout pipelineLayout,
        const DensityPushConstants& pushConstants,
        VkDescriptorSet descriptorSet,
        GPUProfiler* profiler = nullptr,
        VkDescriptorSet sparseDescriptorSet = VK_NULL_HANDLE,
        VkDescriptorSet denseDescriptorSet = VK_NULL_HANDLE);

private:
    // Upload block IDs to GPU buffer
    void uploadBlocks(
        const std::vector<uint32_t>& blocks,
        Buffer& buffer,
        const VulkanContext& context);
};