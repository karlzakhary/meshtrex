#include "minMaxPass.h"
#include "vulkan_context.h"
#include "shaders.h"
#include <cassert>
#include <vector>

MinMaxPass::MinMaxPass(const VulkanContext& context, const char* leafShaderPath, const char* octreeShaderPath) :
    context_(context),
    device_(context.getDevice())
{
    createLeafPipelineLayout();
    createLeafPipeline(leafShaderPath);

    createOctreePipelineLayout();
    createOctreePipeline(octreeShaderPath);
}

MinMaxPass::~MinMaxPass() {
    if (leafPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, leafPipeline_, nullptr);
    }
    if (leafMinMaxComputeShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, leafMinMaxComputeShader_.module, nullptr);
    }
    if (octreePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, octreePipeline_, nullptr);
    }
    if (octreeMinMaxComputeShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, octreeMinMaxComputeShader_.module, nullptr);
    }
     if (leafPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, leafPipelineLayout_, nullptr);
    }
    if (octreePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, octreePipelineLayout_, nullptr);
    }
    if (leafDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, leafDescriptorSetLayout_, nullptr);
    }
    if (octreeDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, octreeDescriptorSetLayout_, nullptr);
    }
}

void MinMaxPass::createLeafPipelineLayout() {
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

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &leafDescriptorSetLayout_));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &leafDescriptorSetLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange
    };

    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &leafPipelineLayout_));
}

void MinMaxPass::createOctreePipelineLayout() {
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size =  28
    };

    std::vector<VkDescriptorSetLayoutBinding> bindings(2);

    // Binding 0: Src MinMax Image (read-Only in shader)
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    // Binding 1: Dest MinMax Image (read-Only in shader)
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

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &octreeDescriptorSetLayout_));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &octreeDescriptorSetLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange
    };

    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &octreePipelineLayout_));
}

void MinMaxPass::createLeafPipeline(const char* shaderPath) {
    assert(loadShader(leafMinMaxComputeShader_, device_, shaderPath));
    leafPipeline_ = createComputePipeline(device_, nullptr, leafMinMaxComputeShader_, leafPipelineLayout_);
    assert(leafPipeline_ != VK_NULL_HANDLE);
}

void MinMaxPass::createOctreePipeline(const char* shaderPath) {
    assert(loadShader(octreeMinMaxComputeShader_, device_, shaderPath));
    octreePipeline_ = createComputePipeline(device_, nullptr, octreeMinMaxComputeShader_, octreePipelineLayout_);
    assert(octreePipeline_ != VK_NULL_HANDLE);
}

void MinMaxPass::recordLeafDispatch(VkCommandBuffer cmd,
                             VkImageView inputVolumeView,
                             VkImageView outputMinMaxView,
                             const PushConstants& pushConstants) {

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, leafPipeline_);

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
                              leafPipelineLayout_, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // Push Constants
    vkCmdPushConstants(cmd, leafPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pushConstants);

    // Dispatch
    vkCmdDispatch(cmd, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);
}

void MinMaxPass::recordOctreeDispatch(VkCommandBuffer cmd,
                             VkImageView srcView,
                             VkExtent3D srcExtent,
                             VkImageView dstView,
                             VkExtent3D dstExtent) {

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, octreePipeline_);
    struct { uint32_t s[3]; uint32_t d[3]; } pc = {
        { srcExtent.width, srcExtent.height, srcExtent.depth },
        { dstExtent.width, dstExtent.height, dstExtent.depth }
    };            
    // Push Descriptors
    VkDescriptorImageInfo srcImageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = srcView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL // Or SHADER_READ_ONLY_OPTIMAL if layout matches
    };

    VkDescriptorImageInfo dstImageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = dstView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL // Shader writes via imageStore
    };

    std::vector<VkWriteDescriptorSet> writes(2);

    writes[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &srcImageInfo
    };

    writes[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dstImageInfo
    };

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              octreePipelineLayout_, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // Push Constants
    vkCmdPushConstants(cmd, octreePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);

    // Dispatch - one workgroup per destination texel
    vkCmdDispatch(cmd, dstExtent.width, dstExtent.height, dstExtent.depth);
}