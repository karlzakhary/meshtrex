#pragma once

#include <vulkan/vulkan.h>
#include "rasterOcclusionPass.h"
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "buffer.h"
#include "marchingCubesUtils.h"

class VulkanContext;
struct MinMaxOutput;
struct PushConstants;

class TransientExtractionPass {
public:
    TransientExtractionPass(const VulkanContext& context, VkFormat swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM);
    ~TransientExtractionPass();

    struct ShadingParameters {
        alignas(16) glm::vec3 lightDir;
        alignas(16) glm::vec3 viewPos;
        alignas(16) glm::vec3 baseColor;
        alignas(16) glm::vec3 prevPassColor;    // Color for previous frame geometry
        alignas(16) glm::vec3 newPassColor;      // Color for newly visible geometry
        alignas(4) float ambientStrength;
        alignas(4) float diffuseStrength;
        alignas(4) float specularStrength;
        alignas(4) float shininess;
        alignas(4) uint32_t enableDebugColors;  // 1 = show different colors for passes
    };

    // Render both transient extraction passes (legacy - calls Pass1 and Pass2 in sequence)
    void renderTransientPasses(
        VkCommandBuffer cmd,
        const RasterOcclusionPass::Output& occlusionOutput,
        const MinMaxOutput& minMaxOutput,
        const PushConstants& pushConstants,
        const glm::mat4& viewProjMatrix,
        const glm::vec3& cameraPos,
        VkImageView colorImageView,
        VkImageView depthImageView,
        VkExtent2D renderExtent,
        const ShadingParameters& shadingParams = getDefaultShadingParams()
    );
    
    // Render Pass 1: Previous frame's visible blocks (establishes depth for occlusion)
    void renderPass1_PreviousVisible(
        VkCommandBuffer cmd,
        const RasterOcclusionPass::Output& occlusionOutput,
        const MinMaxOutput& minMaxOutput,
        const PushConstants& pushConstants,
        const glm::mat4& viewProjMatrix,
        const glm::vec3& cameraPos,
        VkImageView colorImageView,
        VkImageView depthImageView,
        VkExtent2D renderExtent,
        const ShadingParameters& shadingParams = getDefaultShadingParams()
    );
    
    // Render Pass 2: Newly visible blocks (difference between current and previous)
    void renderPass2_NewlyVisible(
        VkCommandBuffer cmd,
        const RasterOcclusionPass::Output& occlusionOutput,
        const MinMaxOutput& minMaxOutput,
        const PushConstants& pushConstants,
        const glm::mat4& viewProjMatrix,
        const glm::vec3& cameraPos,
        VkImageView colorImageView,
        VkImageView depthImageView,
        VkExtent2D renderExtent,
        const ShadingParameters& shadingParams = getDefaultShadingParams()
    );

    // Get default shading parameters
    static ShadingParameters getDefaultShadingParams() {
        ShadingParameters params;
        params.lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
        params.viewPos = glm::vec3(0.0f);
        params.baseColor = glm::vec3(0.8f, 0.8f, 0.8f);
        params.prevPassColor = glm::vec3(0.3f, 0.5f, 0.8f);  // Blueish for old geometry
        params.newPassColor = glm::vec3(0.8f, 0.3f, 0.3f);   // Reddish for new geometry
        params.ambientStrength = 0.2f;
        params.diffuseStrength = 0.7f;
        params.specularStrength = 0.3f;
        params.shininess = 32.0f;
        params.enableDebugColors = 0;
        return params;
    }
    
    // Temporary resources that need to be kept alive until command buffer submission
    struct TempResources {
        // Resources from Pass 1
        // Note: minMaxSampler is owned by MinMaxOutput, not tracked here
        // Note: viewUniformBuffer is now persistent, not tracked here
        VkDescriptorPool descriptorPool_pass1 = VK_NULL_HANDLE;
        
        // Resources from Pass 2
        // Note: minMaxSampler is owned by MinMaxOutput, not tracked here
        // Note: viewUniformBuffer is now persistent, not tracked here
        VkDescriptorPool descriptorPool_pass2 = VK_NULL_HANDLE;
        
        // Resources from indirect update compute
        VkDescriptorPool descriptorPool_indirectUpdate = VK_NULL_HANDLE;
        
        void destroy(VkDevice device) {
            // Clean up Pass 1 resources
            if (descriptorPool_pass1) {
                vkDestroyDescriptorPool(device, descriptorPool_pass1, nullptr);
                descriptorPool_pass1 = VK_NULL_HANDLE;
            }
            
            // Clean up Pass 2 resources
            if (descriptorPool_pass2) {
                vkDestroyDescriptorPool(device, descriptorPool_pass2, nullptr);
                descriptorPool_pass2 = VK_NULL_HANDLE;
            }
            
            // Clean up indirect update resources
            if (descriptorPool_indirectUpdate) {
                vkDestroyDescriptorPool(device, descriptorPool_indirectUpdate, nullptr);
                descriptorPool_indirectUpdate = VK_NULL_HANDLE;
            }
        }
    } tempResources_;
    
    // Clean up temporary resources (call after command buffer submission)
    void cleanupTempResources() {
        tempResources_.destroy(device_);
        tempResources_ = {}; // Reset to null handles
    }
    
    // Get temp resources for deferred destruction
    TempResources getTempResources() {
        TempResources resources = tempResources_;
        tempResources_ = {}; // Clear the original
        return resources;
    }

private:
    const VulkanContext& context_;
    VkDevice device_;
    
    // Persistent resources - created once, reused across frames
    // Note: We use the minMaxSampler from MinMaxOutput instead of creating our own
    Buffer persistentViewUniformBuffer_pass1_{};
    Buffer persistentViewUniformBuffer_pass2_{};
    
    void createPersistentResources();
    void destroyPersistentResources();
    
    // Pipeline for pass 1 (PVS_prev)
    VkPipeline pass1Pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pass1PipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout pass1DescriptorSetLayout_ = VK_NULL_HANDLE;
    
    // Pipeline for pass 2 (PVS_curr-prev)
    VkPipeline pass2Pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pass2PipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout pass2DescriptorSetLayout_ = VK_NULL_HANDLE;
    
    // Compute pipeline for indirect draw updates
    VkPipeline indirectUpdatePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout indirectUpdatePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout indirectUpdateDescriptorSetLayout_ = VK_NULL_HANDLE;
    
    // Shader modules
    VkShaderModule pass1TaskShader_ = VK_NULL_HANDLE;
    VkShaderModule pass2TaskShader_ = VK_NULL_HANDLE;
    VkShaderModule meshShader_ = VK_NULL_HANDLE;      // Shared by both passes
    VkShaderModule fragmentShader_ = VK_NULL_HANDLE;  // Shared by both passes
    VkShaderModule indirectUpdateComputeShader_ = VK_NULL_HANDLE;  // Compute shader for indirect draw updates
    
    // Shading parameters buffer
    Buffer shadingParamsBuffer_{};
    
    // Swapchain format
    VkFormat swapchainFormat_;
    
    // PVS bypass control for testing
    bool bypassPVS_ = false;  // Set to true to bypass occlusion culling
    
    // Marching cubes lookup tables as TBOs
    MarchingCubesTables mcTables_;  // Using shared struct from marchingCubesUtils.h
    
    // Indirect draw buffers for GPU-driven rendering
    Buffer indirectDrawBuffer_ = {};  // Contains VkDrawMeshTasksIndirectCommandEXT structs
    Buffer pvsCountBuffer_ = {};      // Contains PVS counts for compute shader (2 uint32_t)
    bool useIndirectDraw_ = false;    // Whether to use indirect drawing
    
    void createPipelineLayouts();
    void createPipelines();
    void loadShaders();
    void createShadingParametersBuffer();
    void setupMarchingCubesTables(bool useUniqueTables = false);
    void createIndirectDrawBuffer();
    void createIndirectUpdatePipeline();
    void updateIndirectDrawBuffer(VkCommandBuffer cmd, 
                                 const RasterOcclusionPass::Output& occlusionOutput);
    void updateIndirectDrawBufferGPU(VkCommandBuffer cmd,
                                    const RasterOcclusionPass::Output& occlusionOutput,
                                    const PushConstants& pushConstants);
    
    // Helper to create a descriptor set for a pass
    VkDescriptorSet createPassDescriptorSet(
        VkDescriptorPool pool,
        VkDescriptorSetLayout layout,
        const Buffer& viewUniformBuffer,
        VkImageView minMaxImageView,
        VkSampler minMaxSampler,
        VkImageView volumeImageView,
        const Buffer& pvsBuffer,  // Either pvsPrev or pvsDifference
        uint32_t bindingIndex  // 14 for pvsPrev, 15 for pvsDifference
    );
};