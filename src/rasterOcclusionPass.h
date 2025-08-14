#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "buffer.h"

class VulkanContext;
class SharedResources;
struct MinMaxOutput;
struct FilteringOutput;
struct PushConstants;

class RasterOcclusionPass {
public:
    RasterOcclusionPass(const VulkanContext& context);
    ~RasterOcclusionPass();

    struct Output {
        // Visibility buffer containing flags for each block
        Buffer visibilityBuffer{};
        
        // Bitfield buffers for current and previous frame
        Buffer currentBitfieldBuffer{};
        Buffer previousBitfieldBuffer{};
        
        // Compacted PVS data for current frame
        Buffer pvsCurrentBuffer{};
        uint32_t pvsCurrentCount = 0;
        
        // PVS difference (curr - prev)
        Buffer pvsDifferenceBuffer{};  // PVScurr-prev
        uint32_t pvsDifferenceCount = 0;
        
        // Temporal coherence: Previous frame's PVS (for next frame)
        Buffer pvsPreviousBuffer{};    // PVSprev
        uint32_t pvsPreviousCount = 0;
        
        // Frame counter for temporal coherence
        uint32_t frameIndex = 0;
        bool isFirstFrame = true;
        
        // Temporary resources that need to be kept alive until command buffer submission
        struct TempResources {
            VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
            Buffer uniformBuffer{};
            Buffer stagingBuffer{};
            Buffer stagingBuffer2{};  // Second staging buffer for else branch
            Buffer readbackBuffer{};  // Readback buffer for PVS counts
            
            void destroy(VkDevice device) {
                if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                if (uniformBuffer.buffer) destroyBuffer(uniformBuffer, device);
                if (stagingBuffer.buffer) destroyBuffer(stagingBuffer, device);
                if (stagingBuffer2.buffer) destroyBuffer(stagingBuffer2, device);
                if (readbackBuffer.buffer) destroyBuffer(readbackBuffer, device);
            }
        } tempResources;
        
        // Swap current PVS to become previous PVS for next frame
        void swapTemporalBuffers();
        
        // Copy current frame data to previous frame buffers
        void copyCurrentToPrevious(VkDevice device, VkCommandBuffer cmd);
        
        // Read back PVS counts from GPU (call after fence wait)
        void readbackPVSCounts(VkDevice device);
        
        // Clean up temporary resources (call after command buffer submission)
        void cleanupTempResources(VkDevice device) {
            tempResources.destroy(device);
            tempResources = {}; // Reset to null handles
        }
        
        // Cleanup all resources
        void destroy(VkDevice device);
    };

    // Main occlusion culling function (single-frame, deprecated)
    Output performOcclusionCulling(
        VkCommandBuffer cmd,
        const MinMaxOutput& minMaxOutput,
        const FilteringOutput& previousPVS,  // PVS from previous frame
        const PushConstants& pushConstants,
        const glm::mat4& viewProjMatrix,    // View-projection matrix
        VkImageView depthImageView,
        VkExtent2D renderExtent,
        bool ownCommandBuffer = false
    );
    
    // Perform occlusion culling with temporal coherence
    void performTemporalOcclusionCulling(
        VkCommandBuffer cmd,
        Output& output,  // In/out: contains previous frame data, updated with current
        const MinMaxOutput& minMaxOutput,
        const PushConstants& pushConstants,
        const glm::mat4& viewProjMatrix,
        VkImageView depthImageView,
        VkExtent2D renderExtent
    );
    
    // Temporary descriptor pool for indirect update (cleaned up after command submission)
    struct IndirectTempResources {
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        
        void destroy(VkDevice device) {
            if (descriptorPool) {
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                descriptorPool = VK_NULL_HANDLE;
            }
        }
    };
    
    // Get indirect temp resources for deferred destruction
    IndirectTempResources getIndirectTempResources() {
        IndirectTempResources resources = indirectTempResources_;
        indirectTempResources_ = {}; // Clear the original
        return resources;
    }
    
    // Clean up temporary resources for indirect draw (call after command buffer submission)
    void cleanupIndirectTempResources() {
        indirectTempResources_.destroy(device_);
    }

private:
    const VulkanContext& context_;
    VkDevice device_;
    
    // Temporary descriptor pool for indirect update (cleaned up after command submission)
    IndirectTempResources indirectTempResources_;
    
    // Pipeline for task/mesh shader occlusion culling
    VkPipeline occlusionPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout occlusionPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout occlusionDescriptorSetLayout_ = VK_NULL_HANDLE;
    
    // Pipeline for visibility compaction (sparse to bitfield)
    VkPipeline visibilityCompactionPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout visibilityCompactionPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout visibilityCompactionDescriptorSetLayout_ = VK_NULL_HANDLE;
    
    // Pipeline for building PVS output
    VkPipeline buildOutputPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout buildOutputPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout buildOutputDescriptorSetLayout_ = VK_NULL_HANDLE;
    
    // Shader modules
    VkShaderModule taskShader_ = VK_NULL_HANDLE;
    VkShaderModule meshShader_ = VK_NULL_HANDLE;
    VkShaderModule fragmentShader_ = VK_NULL_HANDLE;
    VkShaderModule visibilityCompactionShader_ = VK_NULL_HANDLE;
    VkShaderModule buildOutputShader_ = VK_NULL_HANDLE;
    
    // Indirect draw support
    Buffer indirectDrawBuffer_{};
    VkPipeline indirectUpdatePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout indirectUpdatePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout indirectUpdateDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkShaderModule indirectUpdateComputeShader_ = VK_NULL_HANDLE;
    bool useIndirectDraw_ = true;
    
    // Persistent resources - created once, reused across frames
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
    std::array<VkDescriptorPool, MAX_FRAMES_IN_FLIGHT> persistentDescriptorPools_{};
    uint32_t currentFrameIndex_ = 0;
    
    void createPersistentResources();
    void destroyPersistentResources();
    
public:
    // Call this at the beginning of each frame to reset the descriptor pool and advance frame index
    void beginFrame() {
        currentFrameIndex_ = (currentFrameIndex_ + 1) % MAX_FRAMES_IN_FLIGHT;
        vkResetDescriptorPool(device_, persistentDescriptorPools_[currentFrameIndex_], 0);
    }
    
private:
    void createPipelineLayout();
    void createOcclusionPipeline();
    void createVisibilityCompactionPipeline();
    void createBuildOutputPipeline();
    void loadShaders();
    
    // Helper to create visibility buffer
    void createVisibilityBuffer(Output& output, uint32_t numBlocks);
    void createBitfieldBuffers(Output& output, uint32_t numBlocks);
    void createPVSBuffers(Output& output, uint32_t maxBlocks);
    
    // Initialize output for first frame
    void initializeOutput(Output& output, uint32_t numBlocks);
    
    // Indirect draw helpers
    void createIndirectDrawBuffer();
    void createIndirectUpdatePipeline();
    void updateIndirectDrawBufferGPU(VkCommandBuffer cmd, uint32_t totalBlocks);
};