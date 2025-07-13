#include "dgcGenerationPipeline.h"
#include "shaders.h"
#include "resources.h"
#include "buffer.h"
#include "vulkan_utils.h"
#include <iostream>
#include <vector>

DGCGenerationPipeline::~DGCGenerationPipeline() {
    cleanup();
}

void DGCGenerationPipeline::cleanup() {
    if (device_ == VK_NULL_HANDLE) return;

    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }
    
    device_ = VK_NULL_HANDLE;
}

bool DGCGenerationPipeline::setup(VkDevice device) {
    device_ = device;

    if (!createDescriptorSetLayout()) {
        std::cerr << "Failed to create descriptor set layout for DGC generation" << std::endl;
        return false;
    }

    if (!createPipelineLayout()) {
        std::cerr << "Failed to create pipeline layout for DGC generation" << std::endl;
        return false;
    }

    if (!createComputePipeline()) {
        std::cerr << "Failed to create compute pipeline for DGC generation" << std::endl;
        return false;
    }

    if (!createDescriptorPool()) {
        std::cerr << "Failed to create descriptor pool for DGC generation" << std::endl;
        return false;
    }

    if (!allocateDescriptorSet()) {
        std::cerr << "Failed to allocate descriptor set for DGC generation" << std::endl;
        return false;
    }

    return true;
}

bool DGCGenerationPipeline::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    // Binding 0: Active block count buffer (input)
    bindings.push_back({
        0,                                          // binding
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          // descriptorType
        1,                                          // descriptorCount
        VK_SHADER_STAGE_COMPUTE_BIT,               // stageFlags
        nullptr                                     // pImmutableSamplers
    });

    // Binding 1: Indirect draw buffer (output)
    bindings.push_back({
        1,                                          // binding
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          // descriptorType
        1,                                          // descriptorCount
        VK_SHADER_STAGE_COMPUTE_BIT,               // stageFlags
        nullptr                                     // pImmutableSamplers
    });

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    return vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_) == VK_SUCCESS;
}

bool DGCGenerationPipeline::createPipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    return vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) == VK_SUCCESS;
}

bool DGCGenerationPipeline::createComputePipeline() {
    // Load shader
    Shader computeShader {};
    std::string path = "/spirv/dgcGeneration.comp.spv";
    assert(loadShader(computeShader, device_, path.c_str()));

    VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShader.module;
    computeShaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.stage = computeShaderStageInfo;

    VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);

    vkDestroyShaderModule(device_, computeShader.module, nullptr);

    return result == VK_SUCCESS;
}

bool DGCGenerationPipeline::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}
    };

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    return vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_) == VK_SUCCESS;
}

bool DGCGenerationPipeline::allocateDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;

    return vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_) == VK_SUCCESS;
}

DGCGenerationPipeline& getDGCGenerationPipeline(VkDevice device) {
    static DGCGenerationPipeline pipeline;
    static bool pipelineInitialized = false;
    
    // Initialize pipeline on first use
    if (!pipelineInitialized) {
        if (!pipeline.setup(device)) {
            throw std::runtime_error("Failed to setup DGC generation pipeline");
        }
        pipelineInitialized = true;
    }
    
    return pipeline;
}

void generateIndirectDrawCommands(VkCommandBuffer cmd,
                                VkDevice device,
                                const Buffer& activeBlockCountBuffer,
                                const Buffer& indirectDrawBuffer) {
    DGCGenerationPipeline& pipeline = getDGCGenerationPipeline(device);
    
    // Update descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    
    VkDescriptorBufferInfo activeCountInfo = {activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo indirectDrawInfo = {indirectDrawBuffer.buffer, 0, VK_WHOLE_SIZE};
    
    writes.push_back({
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        pipeline.descriptorSet_,
        0, // binding
        0, // arrayElement
        1, // descriptorCount
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr,
        &activeCountInfo,
        nullptr
    });
    
    writes.push_back({
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        pipeline.descriptorSet_,
        1, // binding
        0, // arrayElement
        1, // descriptorCount
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr,
        &indirectDrawInfo,
        nullptr
    });
    
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout_, 
                           0, 1, &pipeline.descriptorSet_, 0, nullptr);
    
    // Dispatch single workgroup
    vkCmdDispatch(cmd, 1, 1, 1);
    
    // Barrier to ensure write completes before mesh shader reads
    VkBufferMemoryBarrier2 barrier = bufferBarrier(
        indirectDrawBuffer.buffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &barrier, 0, {});
}