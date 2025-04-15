#include "minMaxPass.h"
#include "vulkan_context.h"
#include "shaders.h"
#include <cassert>
#include <vector>

MinMaxPass::MinMaxPass(const VulkanContext& context, const char* shaderPath) :
    context_(context),
    device_(context.getDevice())
{
    createPipelineLayout();
    createPipeline(shaderPath);
}

MinMaxPass::~MinMaxPass() {
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

void MinMaxPass::createPipelineLayout() {
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants)
    };

    std::vector<VkDescriptorSetLayoutBinding> bindings(2);

    // Binding 0: Input Volume Image (Read-Only in shader)
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    // Binding 1: Output MinMax Image (Write-Only in shader)
    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
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

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange
    };

    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout_));
}

void MinMaxPass::createPipeline(const char* shaderPath) {
    assert(loadShader(computeShader_, device_, shaderPath));
    pipeline_ = createComputePipeline(device_, nullptr, computeShader_, pipelineLayout_);
    assert(pipeline_ != VK_NULL_HANDLE);
}

void MinMaxPass::recordDispatch(VkCommandBuffer cmd,
                             VkImageView inputVolumeView,
                             VkImageView outputMinMaxView,
                             const PushConstants& pushConstants) {

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    // Push Descriptors
    VkDescriptorImageInfo inputImageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = inputVolumeView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL // Or SHADER_READ_ONLY_OPTIMAL if layout matches
    };

    VkDescriptorImageInfo outputImageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = outputMinMaxView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL // Shader writes via imageStore
    };

    std::vector<VkWriteDescriptorSet> writes(2);

    writes[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &inputImageInfo
    };

    writes[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &outputImageInfo
    };

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout_, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // Push Constants
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pushConstants);

    // Dispatch
    vkCmdDispatch(cmd, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);
}