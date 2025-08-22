#include "computeOcclusionPass.h"
#include "vulkan_context.h"
#include "vulkan_utils.h"
#include "shaders.h"
#include "common.h"  // For PushConstants
#include <iostream>
#include <cmath>
#include <cstring>
#include <cassert>

ComputeOcclusionPass::ComputeOcclusionPass(const VulkanContext& context) 
    : context_(context), device_(context.getDevice()) {
    loadShaders();
    createPipelines();
    createDescriptorPools();
}

ComputeOcclusionPass::~ComputeOcclusionPass() {
    hiZPyramid_.destroy(device_);
    
    // Destroy descriptor pools
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (descriptorPools_[i] != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptorPools_[i], nullptr);
        }
    }
    
    if (hiZGeneratePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, hiZGeneratePipeline_, nullptr);
    }
    if (hiZGenerateLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, hiZGenerateLayout_, nullptr);
    }
    if (hiZGenerateDescLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, hiZGenerateDescLayout_, nullptr);
    }
    
    if (hiZInitPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, hiZInitPipeline_, nullptr);
    }
    if (hiZInitLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, hiZInitLayout_, nullptr);
    }
    if (hiZInitDescLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, hiZInitDescLayout_, nullptr);
    }
    
    if (occlusionPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, occlusionPipeline_, nullptr);
    }
    if (occlusionLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, occlusionLayout_, nullptr);
    }
    if (occlusionDescLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, occlusionDescLayout_, nullptr);
    }
    
    if (buildOutputPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, buildOutputPipeline_, nullptr);
    }
    if (buildOutputLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, buildOutputLayout_, nullptr);
    }
    if (buildOutputDescLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, buildOutputDescLayout_, nullptr);
    }
    
    if (uniformBuffer_.buffer != VK_NULL_HANDLE) {
        destroyBuffer(uniformBuffer_, device_);
    }
    
    if (hiZGenerateShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, hiZGenerateShader_, nullptr);
    }
    if (hiZInitShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, hiZInitShader_, nullptr);
    }
    if (occlusionShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, occlusionShader_, nullptr);
    }
    if (buildOutputShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, buildOutputShader_, nullptr);
    }
    
    if (depthSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, depthSampler_, nullptr);
    }
    
    if (readbackBuffer_.buffer != VK_NULL_HANDLE) {
        destroyBuffer(readbackBuffer_, device_);
    }
}

void ComputeOcclusionPass::HiZPyramid::destroy(VkDevice device) {
    if (sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }
    
    // Destroy full pyramid view
    if (fullPyramidView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, fullPyramidView, nullptr);
        fullPyramidView = VK_NULL_HANDLE;
    }
    
    // Destroy individual level views
    for (auto& imageView : imageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, imageView, nullptr);
        }
    }
    imageViews.clear();
    
    // Destroy image and memory (now just one of each)
    if (!images.empty() && images[0] != VK_NULL_HANDLE) {
        vkDestroyImage(device, images[0], nullptr);
    }
    if (!memories.empty() && memories[0] != VK_NULL_HANDLE) {
        vkFreeMemory(device, memories[0], nullptr);
    }
    images.clear();
    memories.clear();
    
    levels = 0;
    baseExtent = {0, 0};
}

uint32_t ComputeOcclusionPass::calculateMipLevels(VkExtent2D extent) {
    uint32_t maxDim = std::max(extent.width, extent.height);
    return static_cast<uint32_t>(std::floor(std::log2(maxDim))) + 1;
}

void ComputeOcclusionPass::createHiZPyramid(VkExtent2D baseExtent) {
    // Destroy existing pyramid if any
    hiZPyramid_.destroy(device_);
    
    hiZPyramid_.baseExtent = baseExtent;
    hiZPyramid_.levels = calculateMipLevels(baseExtent);
    
    // Create a SINGLE image with multiple mip levels
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R32_SFLOAT;
    imageInfo.extent = {baseExtent.width, baseExtent.height, 1};
    imageInfo.mipLevels = hiZPyramid_.levels;  // Multiple mip levels in one image!
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | 
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    // Store in first element of vectors for compatibility
    hiZPyramid_.images.resize(1);
    hiZPyramid_.memories.resize(1);
    
    VK_CHECK(vkCreateImage(device_, &imageInfo, nullptr, &hiZPyramid_.images[0]));
    
    // Allocate memory for the single image
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device_, hiZPyramid_.images[0], &memReqs);
    
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = selectMemoryType(
        context_.getMemoryProperties(),
        memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    
    VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &hiZPyramid_.memories[0]));
    VK_CHECK(vkBindImageMemory(device_, hiZPyramid_.images[0], hiZPyramid_.memories[0], 0));
    
    // Create a full pyramid view spanning all mip levels for sampling
    VkImageViewCreateInfo fullViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    fullViewInfo.image = hiZPyramid_.images[0];
    fullViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    fullViewInfo.format = VK_FORMAT_R32_SFLOAT;
    fullViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    fullViewInfo.subresourceRange.baseMipLevel = 0;
    fullViewInfo.subresourceRange.levelCount = hiZPyramid_.levels;  // All mip levels!
    fullViewInfo.subresourceRange.baseArrayLayer = 0;
    fullViewInfo.subresourceRange.layerCount = 1;
    
    VK_CHECK(vkCreateImageView(device_, &fullViewInfo, nullptr, &hiZPyramid_.fullPyramidView));
    
    // Create individual views for each mip level (for compute shader writes)
    hiZPyramid_.imageViews.resize(hiZPyramid_.levels);
    for (uint32_t level = 0; level < hiZPyramid_.levels; level++) {
        VkImageViewCreateInfo levelViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        levelViewInfo.image = hiZPyramid_.images[0];
        levelViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        levelViewInfo.format = VK_FORMAT_R32_SFLOAT;
        levelViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        levelViewInfo.subresourceRange.baseMipLevel = level;
        levelViewInfo.subresourceRange.levelCount = 1;
        levelViewInfo.subresourceRange.baseArrayLayer = 0;
        levelViewInfo.subresourceRange.layerCount = 1;
        
        VK_CHECK(vkCreateImageView(device_, &levelViewInfo, nullptr, &hiZPyramid_.imageViews[level]));
    }
    
    // Create sampler for Hi-Z pyramid
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(hiZPyramid_.levels);
    samplerInfo.anisotropyEnable = VK_FALSE;
    
    VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &hiZPyramid_.sampler));
}

void ComputeOcclusionPass::loadShaders() {
    // Load Hi-Z generation shader
    Shader hiZShader{};
    assert(loadShader(hiZShader, device_, "/spirv/hiz_pyramid_generate.comp.spv"));
    hiZGenerateShader_ = hiZShader.module;
    
    // Load Hi-Z init from depth shader
    Shader hiZInitShader{};
    assert(loadShader(hiZInitShader, device_, "/spirv/hiz_init_from_depth.comp.spv"));
    hiZInitShader_ = hiZInitShader.module;
    
    // Load compute occlusion culling shader  
    Shader occlusionShader{};
    assert(loadShader(occlusionShader, device_, "/spirv/compute_occlusion_culling.comp.spv"));
    occlusionShader_ = occlusionShader.module;
    
    // Load build PVS output shader
    Shader buildOutputShader{};
    assert(loadShader(buildOutputShader, device_, "/spirv/build_pvs_output.comp.spv"));
    buildOutputShader_ = buildOutputShader.module;
}

void ComputeOcclusionPass::createDescriptorPools() {
    // Create descriptor pools for each frame in flight
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 20},         // For Hi-Z pyramid levels
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},        // For PVS buffers
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5},         // For uniform buffers
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5}  // For Hi-Z sampler
        };
        
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 20;  // Maximum number of descriptor sets that can be allocated
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        
        VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPools_[i]));
    }
}

void ComputeOcclusionPass::createPipelines() {
    // Create Hi-Z init from depth pipeline
    {
        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> bindings(2);
        
        // Binding 0: Depth texture sampler
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // Binding 1: Output storage image
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &hiZInitDescLayout_));
        
        // Pipeline layout (no push constants needed)
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &hiZInitDescLayout_;
        
        VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &hiZInitLayout_));
        
        // Compute pipeline
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = hiZInitShader_;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = hiZInitLayout_;
        
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &hiZInitPipeline_));
    }
    
    // Create Hi-Z generation pipeline
    {
        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> bindings(2);
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &hiZGenerateDescLayout_));
        
        // Push constants for mip level info
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t) * 3; // mipLevel, inputWidth, inputHeight
        
        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &hiZGenerateDescLayout_;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        
        VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &hiZGenerateLayout_));
        
        // Compute pipeline
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = hiZGenerateShader_;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = hiZGenerateLayout_;
        
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &hiZGeneratePipeline_));
    }
    
    // Create compute occlusion culling pipeline
    {
        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> bindings(6);
        
        // Min-max texture sampler
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // Hi-Z pyramid sampler
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // Previous PVS buffer
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // Current PVS buffer
        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // PVS counter
        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // Uniform buffer
        bindings[5].binding = 5;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &occlusionDescLayout_));
        
        // Pipeline layout - no push constants needed anymore
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &occlusionDescLayout_;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
        
        VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &occlusionLayout_));
        
        // Compute pipeline
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = occlusionShader_;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = occlusionLayout_;
        
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &occlusionPipeline_));
    }
    
    // Create uniform buffer
    createBuffer(uniformBuffer_, device_, context_.getMemoryProperties(),
        sizeof(UniformData),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Create depth sampler for Hi-Z initialization
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &depthSampler_));
    
    // Create readback buffer for PVS counts (current and difference)
    // Need space for two uint32_t values
    createBuffer(readbackBuffer_, device_, context_.getMemoryProperties(),
        sizeof(uint32_t) * 2,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Create build PVS output pipeline (converts bitfields to PVS lists)
    {
        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> bindings(4);
        
        // Current frame bitfield (input)
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // Previous frame bitfield (input)
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // PVS current buffer (output)
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // PVS difference buffer (output)
        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &buildOutputDescLayout_));
        
        // Push constants for numBitfieldEntries
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t); // numBitfieldEntries
        
        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &buildOutputDescLayout_;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        
        VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &buildOutputLayout_));
        
        // Compute pipeline
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = buildOutputShader_;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = buildOutputLayout_;
        
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &buildOutputPipeline_));
    }
}

void ComputeOcclusionPass::initializeOutput(Output& output, uint32_t totalBlocks) {
    // Reuse the same initialization as RasterOcclusionPass for compatibility
    if (output.pvsCurrentBuffer.buffer == VK_NULL_HANDLE) {
        // Create PVS buffers - need space for count + all block indices
        VkDeviceSize bufferSize = (totalBlocks + 1) * sizeof(uint32_t);
        
        createBuffer(output.pvsCurrentBuffer, device_, context_.getMemoryProperties(),
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            
        createBuffer(output.pvsPreviousBuffer, device_, context_.getMemoryProperties(),
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            
        createBuffer(output.pvsDifferenceBuffer, device_, context_.getMemoryProperties(),
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    
    // Create bitfield buffers if they don't exist (needed for temporal coherence)
    if (output.currentBitfieldBuffer.buffer == VK_NULL_HANDLE) {
        // Bitfield needs 1 bit per block: (totalBlocks + 31) / 32 uints
        VkDeviceSize bitfieldSize = ((totalBlocks + 31) / 32) * sizeof(uint32_t);
        
        createBuffer(output.currentBitfieldBuffer, device_, context_.getMemoryProperties(),
            bitfieldSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            
        createBuffer(output.previousBitfieldBuffer, device_, context_.getMemoryProperties(),
            bitfieldSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    
    // Initialize temporal state
    output.frameIndex = 0;
    output.isFirstFrame = true;
    output.pvsCurrentCount = 0;
    output.pvsPreviousCount = 0;
    output.pvsDifferenceCount = 0;
}

void ComputeOcclusionPass::extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4 planes[6]) {
    // Extract frustum planes from view-projection matrix
    // Left plane
    planes[0] = glm::vec4(
        viewProj[0][3] + viewProj[0][0],
        viewProj[1][3] + viewProj[1][0],
        viewProj[2][3] + viewProj[2][0],
        viewProj[3][3] + viewProj[3][0]
    );
    
    // Right plane
    planes[1] = glm::vec4(
        viewProj[0][3] - viewProj[0][0],
        viewProj[1][3] - viewProj[1][0],
        viewProj[2][3] - viewProj[2][0],
        viewProj[3][3] - viewProj[3][0]
    );
    
    // Bottom plane
    planes[2] = glm::vec4(
        viewProj[0][3] + viewProj[0][1],
        viewProj[1][3] + viewProj[1][1],
        viewProj[2][3] + viewProj[2][1],
        viewProj[3][3] + viewProj[3][1]
    );
    
    // Top plane
    planes[3] = glm::vec4(
        viewProj[0][3] - viewProj[0][1],
        viewProj[1][3] - viewProj[1][1],
        viewProj[2][3] - viewProj[2][1],
        viewProj[3][3] - viewProj[3][1]
    );
    
    // Near plane
    planes[4] = glm::vec4(
        viewProj[0][3] + viewProj[0][2],
        viewProj[1][3] + viewProj[1][2],
        viewProj[2][3] + viewProj[2][2],
        viewProj[3][3] + viewProj[3][2]
    );
    
    // Far plane
    planes[5] = glm::vec4(
        viewProj[0][3] - viewProj[0][2],
        viewProj[1][3] - viewProj[1][2],
        viewProj[2][3] - viewProj[2][2],
        viewProj[3][3] - viewProj[3][2]
    );
    
    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float length = glm::length(glm::vec3(planes[i]));
        planes[i] /= length;
    }
}

void ComputeOcclusionPass::generateHiZPyramid(
    VkCommandBuffer cmd,
    VkImage depthImage,
    VkImageView depthImageView,
    VkExtent2D depthExtent) {
    
    // Create or recreate Hi-Z pyramid if size changed
    if (hiZPyramid_.baseExtent.width != depthExtent.width ||
        hiZPyramid_.baseExtent.height != depthExtent.height) {
        createHiZPyramid(depthExtent);
    }
    
    // Transition depth image from attachment to read-only for sampling
    VkImageMemoryBarrier2 depthBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    depthBarrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    depthBarrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    depthBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.image = depthImage;
    depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthBarrier.subresourceRange.baseMipLevel = 0;
    depthBarrier.subresourceRange.levelCount = 1;
    depthBarrier.subresourceRange.baseArrayLayer = 0;
    depthBarrier.subresourceRange.layerCount = 1;
    
    // Transition entire pyramid image to general layout for compute shader write
    VkImageMemoryBarrier2 firstBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    firstBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    firstBarrier.srcAccessMask = 0;
    firstBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    firstBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    firstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    firstBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    firstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    firstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    firstBarrier.image = hiZPyramid_.images[0];
    firstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    firstBarrier.subresourceRange.baseMipLevel = 0;
    firstBarrier.subresourceRange.levelCount = hiZPyramid_.levels;  // All mip levels
    firstBarrier.subresourceRange.baseArrayLayer = 0;
    firstBarrier.subresourceRange.layerCount = 1;
    
    // Apply both transitions
    VkImageMemoryBarrier2 barriers[] = {depthBarrier, firstBarrier};
    pipelineBarrier(cmd, 0, 0, nullptr, 2, barriers);
    
    // Use compute shader to copy depth buffer to first level of Hi-Z pyramid
    // This avoids the aspect mismatch issue with vkCmdCopyImage
    {
        // Allocate descriptor set for Hi-Z init
        VkDescriptorSet hiZInitDescSet;
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = descriptorPools_[currentFrameIndex_];
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &hiZInitDescLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &hiZInitDescSet));
        
        // Update descriptor set
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = depthSampler_;  // Use the persistent sampler
        depthInfo.imageView = depthImageView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        
        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = hiZPyramid_.imageViews[0];
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        
        std::vector<VkWriteDescriptorSet> writes(2);
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = hiZInitDescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &depthInfo;
        
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = hiZInitDescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;
        
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        
        // Bind pipeline and descriptor set
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiZInitPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiZInitLayout_,
                               0, 1, &hiZInitDescSet, 0, nullptr);
        
        // Dispatch compute to copy depth to Hi-Z level 0
        uint32_t groupsX = (depthExtent.width + 7) / 8;
        uint32_t groupsY = (depthExtent.height + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        
        // Memory barrier before using level 0 as input for next levels
        VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        
        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
    
    // No need to transition remaining pyramid images - we have a single image with all mip levels
    
    // Bind Hi-Z generation pipeline once
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiZGeneratePipeline_);
    
    // Generate mip levels starting from level 1 (level 0 already has depth data)
    for (uint32_t level = 1; level < hiZPyramid_.levels; level++) {
        uint32_t srcLevel = level - 1;  // Source is previous level
        uint32_t dstLevel = level;      // Destination is current level
        
        VkExtent2D inputExtent = {
            std::max(1u, depthExtent.width >> srcLevel),
            std::max(1u, depthExtent.height >> srcLevel)
        };
        VkExtent2D outputExtent = {
            std::max(1u, depthExtent.width >> dstLevel),
            std::max(1u, depthExtent.height >> dstLevel)
        };
        
        // Allocate descriptor set for this level
        VkDescriptorSet hiZDescSet;
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = descriptorPools_[currentFrameIndex_];
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &hiZGenerateDescLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &hiZDescSet));
        
        // Update descriptor set
        VkDescriptorImageInfo inputInfo{};
        inputInfo.imageView = hiZPyramid_.imageViews[srcLevel];
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        
        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = hiZPyramid_.imageViews[dstLevel];
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        
        std::vector<VkWriteDescriptorSet> writes(2);
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = hiZDescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &inputInfo;
        
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = hiZDescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;
        
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        
        // Bind descriptor set
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiZGenerateLayout_,
                               0, 1, &hiZDescSet, 0, nullptr);
        
        // Push constants
        struct {
            uint32_t mipLevel;
            uint32_t inputWidth;
            uint32_t inputHeight;
        } pushData = {srcLevel, inputExtent.width, inputExtent.height};
        
        vkCmdPushConstants(cmd, hiZGenerateLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pushData), &pushData);
        
        // Dispatch compute
        uint32_t groupsX = (outputExtent.width + 7) / 8;
        uint32_t groupsY = (outputExtent.height + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        
        // Barrier between levels
        VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        
        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
    
    // Transition depth image back to attachment layout for subsequent rendering
    VkImageMemoryBarrier2 depthRestoreBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    depthRestoreBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    depthRestoreBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    depthRestoreBarrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    depthRestoreBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthRestoreBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depthRestoreBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthRestoreBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthRestoreBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthRestoreBarrier.image = depthImage;
    depthRestoreBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthRestoreBarrier.subresourceRange.baseMipLevel = 0;
    depthRestoreBarrier.subresourceRange.levelCount = 1;
    depthRestoreBarrier.subresourceRange.baseArrayLayer = 0;
    depthRestoreBarrier.subresourceRange.layerCount = 1;
    
    pipelineBarrier(cmd, 0, 0, nullptr, 1, &depthRestoreBarrier);
}

bool ComputeOcclusionPass::performComputeOcclusionCulling(
    VkCommandBuffer cmd,
    Output& output,
    const MinMaxOutput& minMaxOutput,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    VkImage depthImage,
    VkImageView depthImageView,
    VkExtent2D renderExtent,
    bool forceUpdate) {
    
    // Check for camera changes (similar to RasterOcclusionPass)
    bool shouldUpdate = forceUpdate || output.isFirstFrame;
    bool cameraChanged = false;
    
    if (!output.isFirstFrame && !forceUpdate) {
        // Extract camera position and rotation for movement detection
        glm::mat4 invViewProj = glm::inverse(viewProjMatrix);
        glm::mat4 invPrevViewProj = glm::inverse(output.previousViewProj);
        
        glm::vec3 currentCamPos = glm::vec3(invViewProj[3]);
        glm::vec3 prevCamPos = glm::vec3(invPrevViewProj[3]);
        
        // Check position change
        float positionDelta = glm::length(currentCamPos - prevCamPos);
        
        // Check rotation change by comparing view direction
        glm::vec3 currentViewDir = -glm::normalize(glm::vec3(invViewProj[2]));
        glm::vec3 prevViewDir = -glm::normalize(glm::vec3(invPrevViewProj[2]));
        float rotationDelta = glm::acos(glm::clamp(glm::dot(currentViewDir, prevViewDir), -1.0f, 1.0f));
        
        // Use same thresholds as raster occlusion
        const float POSITION_THRESHOLD = 2.0f;
        const float ROTATION_THRESHOLD = 0.05f;  // ~3 degrees
        
        bool positionChanged = positionDelta > POSITION_THRESHOLD;
        bool rotationChanged = rotationDelta > ROTATION_THRESHOLD;
        
        printf("[ComputeOcclusion] Frame %u: pos_delta=%.3f (thresh=%.1f), rot_delta=%.3f (thresh=%.3f)\n", 
               frameIndex_, positionDelta, POSITION_THRESHOLD, rotationDelta, ROTATION_THRESHOLD);
        
        if (positionChanged || rotationChanged) {
            cameraChanged = true;
            shouldUpdate = true;
            printf("[ComputeOcclusion] Camera changed - updating (pos: %s, rot: %s)\n", 
                   positionChanged ? "yes" : "no", rotationChanged ? "yes" : "no");
        }
    }
    
    if (!shouldUpdate) {
        frameIndex_++;
        printf("[ComputeOcclusion] Frame %u: No update needed (forceUpdate=%d, isFirstFrame=%d)\n", 
               frameIndex_, forceUpdate, output.isFirstFrame);
        return false;
    }
    
    printf("[ComputeOcclusion] Frame %u: Updating occlusion (forceUpdate=%d, isFirstFrame=%d)\n",
           frameIndex_, forceUpdate, output.isFirstFrame);
    
    // Store the view projection matrix for next frame's comparison
    // IMPORTANT: Only update this when we actually perform occlusion culling
    output.previousViewProj = viewProjMatrix;
    
    // Calculate total blocks from volume dimensions
    uint32_t blocksX = (pushConstants.volumeDim.x + 7) / 8;
    uint32_t blocksY = (pushConstants.volumeDim.y + 7) / 8;
    uint32_t blocksZ = (pushConstants.volumeDim.z + 7) / 8;
    uint32_t totalBlocks = blocksX * blocksY * blocksZ;
    
    // Initialize output if needed
    if (output.isFirstFrame || output.pvsCurrentBuffer.buffer == VK_NULL_HANDLE) {
        initializeOutput(output, totalBlocks);
    }
    
    // Bootstrap: On first frame, set initial counts
    // The GPU will compute actual values, but we need something to start with
    if (output.isFirstFrame) {
        // Conservative estimate for first frame
        output.pvsCurrentCount = 0;      // Will be computed by GPU
        output.pvsPreviousCount = 0;      // Nothing from previous frame
        output.pvsDifferenceCount = 0;    // Will be computed by GPU
        printf("[ComputeOcclusion] First frame bootstrap\n");
    }
    
    // NOTE: Hi-Z pyramid generation has been moved to BEFORE Pass 1 rendering
    // This uses the previous frame's complete depth buffer for temporal coherence
    // The pyramid generation and occlusion test are now called separately
    
    // Clear the first element of pvsCurrentBuffer to use as atomic counter
    uint32_t zero = 0;
    vkCmdFillBuffer(cmd, output.pvsCurrentBuffer.buffer, 0, sizeof(uint32_t), 0);
    
    // Also clear the difference buffer count
    vkCmdFillBuffer(cmd, output.pvsDifferenceBuffer.buffer, 0, sizeof(uint32_t), 0);
    
    // On first frame, clear the previous buffer count too (so it starts empty)
    if (output.isFirstFrame) {
        vkCmdFillBuffer(cmd, output.pvsPreviousBuffer.buffer, 0, sizeof(uint32_t), 0);
        // Also clear the previous bitfield on first frame
        vkCmdFillBuffer(cmd, output.previousBitfieldBuffer.buffer, 0, output.previousBitfieldBuffer.size, 0);
    }
    
    // Always clear current bitfield before generating new one
    vkCmdFillBuffer(cmd, output.currentBitfieldBuffer.buffer, 0, output.currentBitfieldBuffer.size, 0);
    
    // Memory barrier
    VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &clearBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    
    // For compute occlusion, we return true to indicate work needs to be done
    // The actual occlusion test and PVS update will happen in runOcclusionCulling
    // The frame counting and PVS count updates are handled there
    return true;
}

void ComputeOcclusionPass::runOcclusionCulling(
    VkCommandBuffer cmd,
    Output& output,
    const MinMaxOutput& minMaxOutput,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    VkExtent2D renderExtent) {
    
    // Run occlusion culling using the previously generated Hi-Z pyramid
    // The Hi-Z pyramid should have been generated from the previous frame's complete depth buffer
    
    // Check if Hi-Z pyramid exists - on first frame it won't
    if (hiZPyramid_.fullPyramidView == VK_NULL_HANDLE) {
        // No Hi-Z pyramid yet (first frame) - create a dummy one to avoid crashes
        // The occlusion test will effectively be skipped due to previousPVSCount == 0
        createHiZPyramid(renderExtent);
        
        // Transition the newly created pyramid to GENERAL layout
        VkImageMemoryBarrier2 pyramidBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        pyramidBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        pyramidBarrier.srcAccessMask = 0;
        pyramidBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        pyramidBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        pyramidBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        pyramidBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        pyramidBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pyramidBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pyramidBarrier.image = hiZPyramid_.images[0];
        pyramidBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        pyramidBarrier.subresourceRange.baseMipLevel = 0;
        pyramidBarrier.subresourceRange.levelCount = hiZPyramid_.levels;
        pyramidBarrier.subresourceRange.baseArrayLayer = 0;
        pyramidBarrier.subresourceRange.layerCount = 1;
        
        pipelineBarrier(cmd, 0, 0, nullptr, 1, &pyramidBarrier);
    }
    
    // Calculate total blocks from volume dimensions
    uint32_t blocksX = (pushConstants.volumeDim.x + 7) / 8;
    uint32_t blocksY = (pushConstants.volumeDim.y + 7) / 8;
    uint32_t blocksZ = (pushConstants.volumeDim.z + 7) / 8;
    uint32_t totalBlocks = blocksX * blocksY * blocksZ;
    
    // Update uniform buffer
    UniformData uniformData{};
    uniformData.viewProj = viewProjMatrix;
    uniformData.prevViewProj = output.previousViewProj;
    extractFrustumPlanes(viewProjMatrix, uniformData.frustumPlanes);
    // Calculate volume bounds and block size from push constants
    // Volume is in voxel space [0, volumeDim]
    uniformData.volumeMin = glm::vec3(0.0f, 0.0f, 0.0f);
    uniformData.volumeMax = glm::vec3(pushConstants.volumeDim.x, pushConstants.volumeDim.y, pushConstants.volumeDim.z);
    uniformData.blockSize = 8.0f;  // Each block is 8x8x8 voxels
    uniformData.isovalue = pushConstants.isovalue;
    uniformData.volumeDimensions = glm::ivec3(pushConstants.volumeDim.x,
                                              pushConstants.volumeDim.y,
                                              pushConstants.volumeDim.z);
    uniformData.totalBlocks = totalBlocks;
    uniformData.screenSize = glm::vec2(renderExtent.width, renderExtent.height);
    uniformData.hiZLevels = hiZPyramid_.levels;
    
    // Use previousPVSCount to signal whether Hi-Z testing should be enabled
    // The shader checks if previousPVSCount > 0 to decide whether to do Hi-Z testing
    // This naturally handles the bootstrap case where we don't have depth data yet
    uniformData.previousPVSCount = output.pvsPreviousCount;
    
    // Use the persistent mapping from the buffer
    memcpy(uniformBuffer_.data, &uniformData, sizeof(UniformData));
    
    // Allocate and update descriptor sets
    VkDescriptorSet occlusionDescSet;
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPools_[currentFrameIndex_];
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &occlusionDescLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &occlusionDescSet));
    
    // Update descriptor set with bindings
    std::vector<VkWriteDescriptorSet> writes;
    
    // Binding 0: Min-max texture sampler
    VkDescriptorImageInfo minMaxInfo{};
    minMaxInfo.sampler = minMaxOutput.minMaxSampler;
    minMaxInfo.imageView = minMaxOutput.minMaxImage.imageView;
    minMaxInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    writes.back().dstSet = occlusionDescSet;
    writes.back().dstBinding = 0;
    writes.back().descriptorCount = 1;
    writes.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes.back().pImageInfo = &minMaxInfo;
    
    // Binding 1: Hi-Z pyramid sampler
    // Now using the full pyramid view that spans all mip levels!
    VkDescriptorImageInfo hiZInfo{};
    hiZInfo.sampler = hiZPyramid_.sampler;
    hiZInfo.imageView = hiZPyramid_.fullPyramidView;  // Full pyramid with all mip levels
    hiZInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    writes.back().dstSet = occlusionDescSet;
    writes.back().dstBinding = 1;
    writes.back().descriptorCount = 1;
    writes.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes.back().pImageInfo = &hiZInfo;
    
    // Binding 2: Previous PVS bitfield (for temporal coherence)
    VkDescriptorBufferInfo prevBitfieldInfo{};
    prevBitfieldInfo.buffer = output.previousBitfieldBuffer.buffer;
    prevBitfieldInfo.offset = 0;
    prevBitfieldInfo.range = VK_WHOLE_SIZE;
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    writes.back().dstSet = occlusionDescSet;
    writes.back().dstBinding = 2;
    writes.back().descriptorCount = 1;
    writes.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes.back().pBufferInfo = &prevBitfieldInfo;
    
    // Binding 3: Current visibility bitfield (output)
    VkDescriptorBufferInfo currBitfieldInfo{};
    currBitfieldInfo.buffer = output.currentBitfieldBuffer.buffer;
    currBitfieldInfo.offset = 0;
    currBitfieldInfo.range = VK_WHOLE_SIZE;
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    writes.back().dstSet = occlusionDescSet;
    writes.back().dstBinding = 3;
    writes.back().descriptorCount = 1;
    writes.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes.back().pBufferInfo = &currBitfieldInfo;
    
    // Binding 4: Dummy buffer (not used but needed for compatibility)
    VkDescriptorBufferInfo dummyInfo{};
    dummyInfo.buffer = output.pvsCurrentBuffer.buffer; // Use any valid buffer
    dummyInfo.offset = 0;
    dummyInfo.range = VK_WHOLE_SIZE;
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    writes.back().dstSet = occlusionDescSet;
    writes.back().dstBinding = 4;
    writes.back().descriptorCount = 1;
    writes.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes.back().pBufferInfo = &dummyInfo;
    
    // Binding 5: Uniform buffer
    VkDescriptorBufferInfo uniformInfo{};
    uniformInfo.buffer = uniformBuffer_.buffer;
    uniformInfo.offset = 0;
    uniformInfo.range = sizeof(UniformData);
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    writes.back().dstSet = occlusionDescSet;
    writes.back().dstBinding = 5;
    writes.back().descriptorCount = 1;
    writes.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes.back().pBufferInfo = &uniformInfo;
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Bind compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, occlusionPipeline_);
    
    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, occlusionLayout_,
                           0, 1, &occlusionDescSet, 0, nullptr);
    
    // Dispatch compute with all blocks
    // New workgroup size matches shader (256 threads per workgroup)
    uint32_t workgroupSize = 256;
    uint32_t numWorkgroups = (totalBlocks + workgroupSize - 1) / workgroupSize;
    
    // No push constants needed anymore
    vkCmdDispatch(cmd, numWorkgroups, 1, 1);
    
    // Barrier before reading results
    VkMemoryBarrier2 resultBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    resultBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    resultBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    resultBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    resultBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &resultBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    
    // Clear PVS output buffers before building
    vkCmdFillBuffer(cmd, output.pvsCurrentBuffer.buffer, 0, sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, output.pvsDifferenceBuffer.buffer, 0, sizeof(uint32_t), 0);
    
    // Memory barrier after clearing
    VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    
    VkDependencyInfo clearDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    clearDepInfo.memoryBarrierCount = 1;
    clearDepInfo.pMemoryBarriers = &clearBarrier;
    vkCmdPipelineBarrier2(cmd, &clearDepInfo);
    
    // Build PVS output from bitfields using build_pvs_output.comp.glsl
    // Allocate descriptor set for build output computation
    VkDescriptorSet buildOutputDescSet;
    VkDescriptorSetAllocateInfo buildAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    buildAllocInfo.descriptorPool = descriptorPools_[currentFrameIndex_];
    buildAllocInfo.descriptorSetCount = 1;
    buildAllocInfo.pSetLayouts = &buildOutputDescLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &buildAllocInfo, &buildOutputDescSet));
    
    // Update descriptor set for build_pvs_output
    std::vector<VkWriteDescriptorSet> buildWrites(4);
    
    // Binding 0: Current frame bitfield (input)
    VkDescriptorBufferInfo buildCurrBitfieldInfo{};
    buildCurrBitfieldInfo.buffer = output.currentBitfieldBuffer.buffer;
    buildCurrBitfieldInfo.offset = 0;
    buildCurrBitfieldInfo.range = VK_WHOLE_SIZE;
    
    // Binding 1: Previous frame bitfield (input)
    VkDescriptorBufferInfo buildPrevBitfieldInfo{};
    buildPrevBitfieldInfo.buffer = output.previousBitfieldBuffer.buffer;
    buildPrevBitfieldInfo.offset = 0;
    buildPrevBitfieldInfo.range = VK_WHOLE_SIZE;
    
    // Binding 2: PVS current buffer (output)
    VkDescriptorBufferInfo pvsCurrentInfo{};
    pvsCurrentInfo.buffer = output.pvsCurrentBuffer.buffer;
    pvsCurrentInfo.offset = 0;
    pvsCurrentInfo.range = VK_WHOLE_SIZE;
    
    // Binding 3: PVS difference buffer (output)  
    VkDescriptorBufferInfo pvsDifferenceInfo{};
    pvsDifferenceInfo.buffer = output.pvsDifferenceBuffer.buffer;
    pvsDifferenceInfo.offset = 0;
    pvsDifferenceInfo.range = VK_WHOLE_SIZE;
    
    buildWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    buildWrites[0].dstSet = buildOutputDescSet;
    buildWrites[0].dstBinding = 0;
    buildWrites[0].descriptorCount = 1;
    buildWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    buildWrites[0].pBufferInfo = &buildCurrBitfieldInfo;
    
    buildWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    buildWrites[1].dstSet = buildOutputDescSet;
    buildWrites[1].dstBinding = 1;
    buildWrites[1].descriptorCount = 1;
    buildWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    buildWrites[1].pBufferInfo = &buildPrevBitfieldInfo;
    
    buildWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    buildWrites[2].dstSet = buildOutputDescSet;
    buildWrites[2].dstBinding = 2;
    buildWrites[2].descriptorCount = 1;
    buildWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    buildWrites[2].pBufferInfo = &pvsCurrentInfo;
    
    buildWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    buildWrites[3].dstSet = buildOutputDescSet;
    buildWrites[3].dstBinding = 3;
    buildWrites[3].descriptorCount = 1;
    buildWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    buildWrites[3].pBufferInfo = &pvsDifferenceInfo;
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(buildWrites.size()), buildWrites.data(), 0, nullptr);
    
    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, buildOutputPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, buildOutputLayout_,
                           0, 1, &buildOutputDescSet, 0, nullptr);
    
    // Push number of bitfield entries
    uint32_t numBitfieldEntries = (totalBlocks + 31) / 32;
    vkCmdPushConstants(cmd, buildOutputLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(uint32_t), &numBitfieldEntries);
    
    // Dispatch with 32 threads per workgroup (matching raster occlusion pass)
    uint32_t buildOutputWorkgroups = (numBitfieldEntries + 31) / 32;
    vkCmdDispatch(cmd, buildOutputWorkgroups, 1, 1);
    
    // Barrier to ensure PVS difference computation completes
    VkMemoryBarrier2 diffBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    diffBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    diffBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    diffBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    diffBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    
    VkDependencyInfo diffDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    diffDepInfo.memoryBarrierCount = 1;
    diffDepInfo.pMemoryBarriers = &diffBarrier;
    vkCmdPipelineBarrier2(cmd, &diffDepInfo);
    
    // NOTE: Buffer copying is no longer needed here!
    // The main render loop now uses swapTemporalBuffers() to swap current and previous buffers
    // This matches the behavior of the raster occlusion path
    
    // Create a per-frame readback buffer (like raster occlusion does)
    // This buffer will be saved with the frame and read back 3 frames later
    if (output.tempResources.readbackBuffer.buffer == VK_NULL_HANDLE) {
        createBuffer(output.tempResources.readbackBuffer, device_, context_.getMemoryProperties(),
            sizeof(uint32_t) * 3,  // Space for previous, current, and difference counts
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    
    // Copy the PVS counts to the per-frame readback buffer
    // Layout must match what readbackPVSCounts expects:
    // [0]: previous count
    // [1]: current count
    // [2]: difference count
    VkBufferCopy countCopy{};
    countCopy.size = sizeof(uint32_t);
    
    // Copy previous count to offset 0
    countCopy.srcOffset = 0;
    countCopy.dstOffset = 0;
    vkCmdCopyBuffer(cmd, output.pvsPreviousBuffer.buffer, output.tempResources.readbackBuffer.buffer, 1, &countCopy);
    
    // Copy current count to offset 4
    countCopy.srcOffset = 0;
    countCopy.dstOffset = sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, output.pvsCurrentBuffer.buffer, output.tempResources.readbackBuffer.buffer, 1, &countCopy);
    
    // Copy difference count to offset 8
    countCopy.srcOffset = 0;
    countCopy.dstOffset = sizeof(uint32_t) * 2;
    vkCmdCopyBuffer(cmd, output.pvsDifferenceBuffer.buffer, output.tempResources.readbackBuffer.buffer, 1, &countCopy);
    
    // Ensure copies complete before CPU reads (3 frames later)
    VkMemoryBarrier2 copyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    copyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    copyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    copyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    copyBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    
    VkDependencyInfo copyDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    copyDepInfo.memoryBarrierCount = 1;
    copyDepInfo.pMemoryBarriers = &copyBarrier;
    vkCmdPipelineBarrier2(cmd, &copyDepInfo);
    
    // Update frame state
    frameIndex_++;
    // NOTE: Do NOT increment output.frameIndex here!
    // It will be incremented by swapTemporalBuffers() in the main loop
    // Double incrementing was causing the alternating pattern
    
    // NOTE: Do NOT clear output.isFirstFrame here!
    // This flag is managed by meshtrex.cpp to handle multiple frames in flight
    // Clearing it here causes issues with temporal coherence
    
    return;
}