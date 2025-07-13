#include "dgcRenderingPipeline.h"
#include "buffer.h"
#include <iostream>
#include <stdexcept>

DGCRenderingPipeline::~DGCRenderingPipeline() {
    cleanup();
}

DGCRenderingPipeline::DGCRenderingPipeline(DGCRenderingPipeline&& other) noexcept
    : device_(other.device_),
      pipelineLayout_(other.pipelineLayout_),
      pipeline_(other.pipeline_),
      descriptorSetLayout_(other.descriptorSetLayout_),
      descriptorPool_(other.descriptorPool_),
      descriptorSet_(other.descriptorSet_),
      computeShader_(std::move(other.computeShader_)) {
    other.device_ = VK_NULL_HANDLE;
    other.pipelineLayout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
    other.descriptorSetLayout_ = VK_NULL_HANDLE;
    other.descriptorPool_ = VK_NULL_HANDLE;
    other.descriptorSet_ = VK_NULL_HANDLE;
}

DGCRenderingPipeline& DGCRenderingPipeline::operator=(DGCRenderingPipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        device_ = other.device_;
        pipelineLayout_ = other.pipelineLayout_;
        pipeline_ = other.pipeline_;
        descriptorSetLayout_ = other.descriptorSetLayout_;
        descriptorPool_ = other.descriptorPool_;
        descriptorSet_ = other.descriptorSet_;
        computeShader_ = std::move(other.computeShader_);
        
        other.device_ = VK_NULL_HANDLE;
        other.pipelineLayout_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
        other.descriptorSetLayout_ = VK_NULL_HANDLE;
        other.descriptorPool_ = VK_NULL_HANDLE;
        other.descriptorSet_ = VK_NULL_HANDLE;
    }
    return *this;
}

bool DGCRenderingPipeline::setup(VkDevice device) {
    device_ = device;
    
    try {
        // Load compute shader
        if (!loadShader(computeShader_, device_, "/spirv/dgcRenderingGeneration.comp.spv")) {
            throw std::runtime_error("Failed to load DGC rendering generation compute shader");
        }
        
        createDescriptorSetLayout();
        createPipelineLayout();
        createDescriptorPool();
        allocateDescriptorSet();
        createComputePipeline();
        
        std::cout << "DGC Rendering Pipeline setup complete" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "DGC Rendering Pipeline setup failed: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void DGCRenderingPipeline::cleanup() {
    releaseResources();
}

void DGCRenderingPipeline::releaseResources() {
    if (device_ != VK_NULL_HANDLE) {
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
        
        vkDestroyShaderModule(device_, computeShader_.module, nullptr);
        device_ = VK_NULL_HANDLE;
    }
}

void DGCRenderingPipeline::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        // Binding 0: Meshlet count buffer (input)
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        // Binding 1: Indirect draw buffer (output)
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));
}

void DGCRenderingPipeline::createPipelineLayout() {
    // Push constants for task workgroup size
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(DGCRenderingPushConstants);
    
    VkPipelineLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    
    VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_));
}

void DGCRenderingPipeline::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}  // 2 storage buffers
    };
    
    VkDescriptorPoolCreateInfo poolInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
}

void DGCRenderingPipeline::allocateDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;
    
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_));
}

void DGCRenderingPipeline::createComputePipeline() {
    VkPipelineShaderStageCreateInfo shaderStage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
    };
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = computeShader_.module;
    shaderStage.pName = "main";
    
    VkComputePipelineCreateInfo pipelineInfo{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = pipelineLayout_;
    
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, 
                                      nullptr, &pipeline_));
}

// Helper function to generate indirect draw commands for rendering
void generateRenderingIndirectCommands(
    VkCommandBuffer cmd,
    VkDevice device,
    const Buffer& meshletCountBuffer,
    const Buffer& indirectDrawBuffer,
    uint32_t taskWorkgroupSize) {
    
    static DGCRenderingPipeline dgcPipeline;
    static bool initialized = false;
    
    // Initialize pipeline on first use
    if (!initialized) {
        if (!dgcPipeline.setup(device)) {
            throw std::runtime_error("Failed to setup DGC rendering pipeline");
        }
        initialized = true;
    }
    
    // Update descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    VkDescriptorBufferInfo meshletCountInfo = {meshletCountBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo indirectInfo = {indirectDrawBuffer.buffer, 0, VK_WHOLE_SIZE};
    
    writes.push_back({
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
        dgcPipeline.descriptorSet_, 0, 0, 1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr, &meshletCountInfo, nullptr
    });
    
    writes.push_back({
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
        dgcPipeline.descriptorSet_, 1, 0, 1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr, &indirectInfo, nullptr
    });
    
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), 
                          writes.data(), 0, nullptr);
    
    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dgcPipeline.pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                           dgcPipeline.pipelineLayout_, 0, 1, 
                           &dgcPipeline.descriptorSet_, 0, nullptr);
    
    // Push constants
    DGCRenderingPushConstants pushConstants{taskWorkgroupSize};
    vkCmdPushConstants(cmd, dgcPipeline.pipelineLayout_, 
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, 
                      sizeof(pushConstants), &pushConstants);
    
    // Dispatch single workgroup to generate command
    vkCmdDispatch(cmd, 1, 1, 1);
    
    // Barrier to ensure command is written before being consumed
    VkBufferMemoryBarrier2 barrier = bufferBarrier(
        indirectDrawBuffer.buffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        0, VK_WHOLE_SIZE
    );
    pipelineBarrier(cmd, {}, 1, &barrier, 0, {});
}