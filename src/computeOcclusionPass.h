#pragma once

#include "common.h"
#include "minMaxOutput.h"
#include "rasterOcclusionPass.h" // Reuse the Output structure
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

class VulkanContext;

// Forward declaration
struct PushConstants;

class ComputeOcclusionPass {
public:
    // Reuse the same output structure as RasterOcclusionPass for compatibility
    using Output = RasterOcclusionPass::Output;
    
    struct HiZPyramid {
        std::vector<VkImage> images;
        std::vector<VkImageView> imageViews;
        std::vector<VkDeviceMemory> memories;
        VkImageView fullPyramidView = VK_NULL_HANDLE; // TODO: Create a view spanning all levels
        VkSampler sampler = VK_NULL_HANDLE;
        uint32_t levels = 0;
        VkExtent2D baseExtent = {0, 0};
        
        void destroy(VkDevice device);
    };
    
    struct UniformData {
        glm::mat4 viewProj;
        glm::mat4 prevViewProj;
        glm::vec4 frustumPlanes[6];
        glm::vec3 volumeMin;
        float blockSize;
        glm::vec3 volumeMax;
        float isovalue;
        glm::ivec3 volumeDimensions;
        uint32_t totalBlocks;
        glm::vec2 screenSize;
        uint32_t hiZLevels;
        uint32_t previousPVSCount;
    };
    
    ComputeOcclusionPass(const VulkanContext& context);
    ~ComputeOcclusionPass();
    
    // Call at the beginning of each frame to reset descriptor pools
    void beginFrame() {
        currentFrameIndex_ = (currentFrameIndex_ + 1) % MAX_FRAMES_IN_FLIGHT;
        vkResetDescriptorPool(device_, descriptorPools_[currentFrameIndex_], 0);
    }
    
    // Initialize resources
    void initializeOutput(Output& output, uint32_t totalBlocks);
    
    // Main occlusion culling function using compute-based Hi-Z
    // This prepares buffers and checks if update is needed
    bool performComputeOcclusionCulling(
        VkCommandBuffer cmd,
        Output& output,
        const MinMaxOutput& minMaxOutput,
        const PushConstants& pushConstants,
        const glm::mat4& viewProjMatrix,
        VkImage depthImage,
        VkImageView depthImageView,
        VkExtent2D renderExtent,
        bool forceUpdate = false
    );
    
    // Run occlusion culling using the previously generated Hi-Z pyramid
    void runOcclusionCulling(
        VkCommandBuffer cmd,
        Output& output,
        const MinMaxOutput& minMaxOutput,
        const PushConstants& pushConstants,
        const glm::mat4& viewProjMatrix,
        VkExtent2D renderExtent
    );
    
    // Generate Hi-Z pyramid from depth buffer
    void generateHiZPyramid(
        VkCommandBuffer cmd,
        VkImage depthImage,
        VkImageView depthImageView,
        VkExtent2D depthExtent
    );
    
    // Extract frustum planes from view-projection matrix
    void extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4 planes[6]);
    
private:
    const VulkanContext& context_;
    VkDevice device_;
    
    // Frame management
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
    uint32_t currentFrameIndex_ = 0;
    VkDescriptorPool descriptorPools_[MAX_FRAMES_IN_FLIGHT] = {};
    
    // Hi-Z pyramid resources
    HiZPyramid hiZPyramid_;
    
    // Pipeline for Hi-Z generation
    VkPipeline hiZGeneratePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout hiZGenerateLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout hiZGenerateDescLayout_ = VK_NULL_HANDLE;
    
    // Pipeline for initializing Hi-Z from depth buffer
    VkPipeline hiZInitPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout hiZInitLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout hiZInitDescLayout_ = VK_NULL_HANDLE;
    
    // Pipeline for compute occlusion culling
    VkPipeline occlusionPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout occlusionLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout occlusionDescLayout_ = VK_NULL_HANDLE;
    
    // Pipeline for building PVS output from bitfields
    VkPipeline buildOutputPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout buildOutputLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout buildOutputDescLayout_ = VK_NULL_HANDLE;
    
    // Uniform buffer
    Buffer uniformBuffer_;
    
    // Readback buffer for PVS counts
    Buffer readbackBuffer_;
    
    // Shaders
    VkShaderModule hiZGenerateShader_ = VK_NULL_HANDLE;
    VkShaderModule hiZInitShader_ = VK_NULL_HANDLE;
    VkShaderModule occlusionShader_ = VK_NULL_HANDLE;
    VkShaderModule buildOutputShader_ = VK_NULL_HANDLE;
    
    // Depth sampler for Hi-Z initialization
    VkSampler depthSampler_ = VK_NULL_HANDLE;
    
    // Temporal coherence tracking
    // Note: Camera tracking now uses output.previousViewProj for consistency with raster path
    uint32_t frameIndex_ = 0;
    
    // Store parameters for deferred occlusion culling
    const MinMaxOutput* storedMinMaxOutput_ = nullptr;
    const PushConstants* storedPushConstants_ = nullptr;
    glm::mat4 storedViewProjMatrix_;
    VkExtent2D storedRenderExtent_;
    
    void createHiZPyramid(VkExtent2D baseExtent);
    void createPipelines();
    void createDescriptorPools();
    void loadShaders();
    
    // Helper to calculate number of mip levels needed
    uint32_t calculateMipLevels(VkExtent2D extent);
};