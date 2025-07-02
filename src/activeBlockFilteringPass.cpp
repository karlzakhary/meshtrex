#include "activeBlockFilteringPass.h"
#include "vulkan_context.h"
#include "shaders.h"
#include <cassert>
#include <vector>
#include <iostream>

ActiveBlockFilteringPass::ActiveBlockFilteringPass(const VulkanContext& context, const char* shaderPath) :
    context_(context),
    device_(context.getDevice())
{
    createPipelineLayout();
    createPipeline(shaderPath);
}

ActiveBlockFilteringPass::~ActiveBlockFilteringPass() {
     if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
     if (computeShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, computeShader_.module, nullptr);
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
    }
}

void ActiveBlockFilteringPass::createPipelineLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    bindings[0] = { // MinMax Input Image
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    bindings[1] = { // Compacted IDs Output Buffer
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    bindings[2] = { // Atomic Counter Output Buffer
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));

    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants)
        };

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange
    };

    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout_));
}

void ActiveBlockFilteringPass::createPipeline(const char* shaderPath) {
    assert(loadShader(computeShader_, device_, shaderPath));
    pipeline_ = createComputePipeline(device_, nullptr, computeShader_, pipelineLayout_); // Assuming null pipeline cache
    assert(pipeline_ != VK_NULL_HANDLE);
    std::cout << "Filtering pipeline created." << std::endl; // Keep relevant info
}


void ActiveBlockFilteringPass::recordDispatch(VkCommandBuffer cmd,
                                           VkImageView minMaxImageView,
                                           VkSampler sampler,
                                           const Buffer& compactedBlockIdBuffer,
                                           const Buffer& activeBlockCountBuffer,
                                           const PushConstants& pushConstants) const
{

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    // Push Descriptors
    VkDescriptorImageInfo minMaxImageInfo = {
        .sampler = sampler,
        .imageView = minMaxImageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL // Must match layout from previous pass output
    };

    VkDescriptorBufferInfo compactedIdBufferInfo = {
        .buffer = compactedBlockIdBuffer.buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    VkDescriptorBufferInfo countBufferInfo = {
        .buffer = activeBlockCountBuffer.buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    std::vector<VkWriteDescriptorSet> writes(3);
    writes[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &minMaxImageInfo
    };
    writes[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &compactedIdBufferInfo
    };
    writes[2] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 2,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &countBufferInfo
    };

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // Push Constants
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pushConstants);

    // Calculate dispatch size (1D)
    uint32_t totalBlocks = pushConstants.blockGridDim.x * pushConstants.blockGridDim.y * pushConstants.blockGridDim.z;
    uint32_t localSizeX = 128; // TODO: Consider making this configurable or querying from shader
    uint32_t groupCountX = (totalBlocks + localSizeX - 1) / localSizeX;

    std::cout << "Dispatching " << groupCountX << " workgroups (" << totalBlocks << " total blocks)..." << std::endl;
    vkCmdDispatch(cmd, groupCountX, 1, 1);
}