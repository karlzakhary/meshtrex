#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <glm/glm.hpp>

class VulkanContext;
struct MinMaxOutput;
struct FilteringOutput;
struct PushConstants;

class RasterOcclusionPass {
public:
    RasterOcclusionPass(const VulkanContext& context);
    ~RasterOcclusionPass();

    struct Output {
        // Visibility buffer containing flags for each block
        VkBuffer visibilityBuffer = VK_NULL_HANDLE;
        VkDeviceMemory visibilityMemory = VK_NULL_HANDLE;
        VkDeviceSize visibilityBufferSize = 0;
        
        // Bitfield buffers for current and previous frame
        VkBuffer currentBitfieldBuffer = VK_NULL_HANDLE;
        VkDeviceMemory currentBitfieldMemory = VK_NULL_HANDLE;
        VkDeviceSize bitfieldBufferSize = 0;
        
        VkBuffer previousBitfieldBuffer = VK_NULL_HANDLE;
        VkDeviceMemory previousBitfieldMemory = VK_NULL_HANDLE;
        
        // Compacted PVS data for current frame
        VkBuffer pvsCurrentBuffer = VK_NULL_HANDLE;
        VkDeviceMemory pvsCurrentMemory = VK_NULL_HANDLE;
        uint32_t pvsCurrentCount = 0;
        
        // PVS difference (curr - prev)
        VkBuffer pvsDifferenceBuffer = VK_NULL_HANDLE;  // PVScurr-prev
        VkDeviceMemory pvsDifferenceMemory = VK_NULL_HANDLE;
        uint32_t pvsDifferenceCount = 0;
        
        // Temporal coherence: Previous frame's PVS (for next frame)
        VkBuffer pvsPreviousBuffer = VK_NULL_HANDLE;    // PVSprev
        VkDeviceMemory pvsPreviousMemory = VK_NULL_HANDLE;
        uint32_t pvsPreviousCount = 0;
        
        // PVS buffer size (same for all PVS buffers)
        VkDeviceSize pvsBufferSize = 0;
        
        // Frame counter for temporal coherence
        uint32_t frameIndex = 0;
        bool isFirstFrame = true;
        
        // Temporary resources that need to be kept alive until command buffer submission
        struct TempResources {
            VkSampler minMaxSampler = VK_NULL_HANDLE;
            VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
            VkBuffer uniformBuffer = VK_NULL_HANDLE;
            VkDeviceMemory uniformMemory = VK_NULL_HANDLE;
            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
            VkBuffer stagingBuffer2 = VK_NULL_HANDLE;  // Second staging buffer for else branch
            VkDeviceMemory stagingMemory2 = VK_NULL_HANDLE;
            VkBuffer readbackBuffer = VK_NULL_HANDLE;  // Readback buffer for PVS counts
            VkDeviceMemory readbackMemory = VK_NULL_HANDLE;
            
            void destroy(VkDevice device) {
                if (minMaxSampler) vkDestroySampler(device, minMaxSampler, nullptr);
                if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                if (uniformBuffer) vkDestroyBuffer(device, uniformBuffer, nullptr);
                if (uniformMemory) vkFreeMemory(device, uniformMemory, nullptr);
                if (stagingBuffer) vkDestroyBuffer(device, stagingBuffer, nullptr);
                if (stagingMemory) vkFreeMemory(device, stagingMemory, nullptr);
                if (stagingBuffer2) vkDestroyBuffer(device, stagingBuffer2, nullptr);
                if (stagingMemory2) vkFreeMemory(device, stagingMemory2, nullptr);
                if (readbackBuffer) vkDestroyBuffer(device, readbackBuffer, nullptr);
                if (readbackMemory) vkFreeMemory(device, readbackMemory, nullptr);
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

private:
    const VulkanContext& context_;
    VkDevice device_;
    
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
};