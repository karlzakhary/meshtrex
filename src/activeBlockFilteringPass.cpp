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

ActiveBlockFilteringPass::ActiveBlockFilteringPass(const VulkanContext& context, 
                                                 const char* regularShaderPath,
                                                 const char* streamingShaderPath) :
    context_(context),
    device_(context.getDevice())
{
    // Create regular pipeline
    createPipelineLayout();
    createPipeline(regularShaderPath);
    
    // Create streaming pipeline if path provided
    if (streamingShaderPath) {
        createStreamingPipelineLayout();
        createStreamingPipeline(streamingShaderPath);
    }
}

ActiveBlockFilteringPass::~ActiveBlockFilteringPass() {
    // Destroy regular pipeline resources
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
    
    // Destroy streaming pipeline resources
    if (streamingPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, streamingPipeline_, nullptr);
    }
    if (streamingComputeShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, streamingComputeShader_.module, nullptr);
    }
    if (streamingPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, streamingPipelineLayout_, nullptr);
    }
    if (streamingDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, streamingDescriptorSetLayout_, nullptr);
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

void ActiveBlockFilteringPass::createStreamingPipelineLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(4);

    bindings[0] = { // Page Table SSBO
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    bindings[1] = { // MinMax Input Image
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    bindings[2] = { // Compacted IDs Output Buffer
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    bindings[3] = { // Atomic Counter Output Buffer
        .binding = 3,
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

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &streamingDescriptorSetLayout_));

    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 48 // 12 uint32_t values as defined in shader (including volume dimensions)
    };

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &streamingDescriptorSetLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange
    };

    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &streamingPipelineLayout_));
}

void ActiveBlockFilteringPass::createStreamingPipeline(const char* shaderPath) {
    assert(loadShader(streamingComputeShader_, device_, shaderPath));
    streamingPipeline_ = createComputePipeline(device_, nullptr, streamingComputeShader_, streamingPipelineLayout_);
    assert(streamingPipeline_ != VK_NULL_HANDLE);
    // std::cout << "Streaming filtering pipeline created." << std::endl;
}

void ActiveBlockFilteringPass::recordStreamingDispatch(VkCommandBuffer cmd,
                                                      VkImageView minMaxImageView,
                                                      VkSampler sampler,
                                                      const Buffer& pageTableBuffer,
                                                      const Buffer& compactedBlockIdBuffer,
                                                      const Buffer& activeBlockCountBuffer,
                                                      const PushConstants& pushConstants,
                                                      const PageCoord& pageCoord) const
{
    // Use streaming pipeline if available, otherwise fall back to regular
    VkPipeline pipelineToUse = (streamingPipeline_ != VK_NULL_HANDLE) ? streamingPipeline_ : pipeline_;
    VkPipelineLayout layoutToUse = (streamingPipelineLayout_ != VK_NULL_HANDLE) ? streamingPipelineLayout_ : pipelineLayout_;
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineToUse);

    if (streamingPipelineLayout_ != VK_NULL_HANDLE) {
        // For streaming pipeline, we expect the pageTableSet to contain:
        // - Binding 0: Page table (from streaming system)
        // - We need to create a complete descriptor set with all 4 bindings
        
        // TODO: This is a design issue - the streaming system provides bindings 0 & 1,
        // but this pass needs to add bindings 2 & 3. For now, use push descriptors
        // for all bindings until we have a better integration approach.
        
        VkDescriptorBufferInfo pageTableInfo = {
            .buffer = pageTableBuffer.buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };
        VkDescriptorImageInfo minMaxImageInfo = {
            .sampler = sampler,
            .imageView = minMaxImageView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
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
        
        std::vector<VkWriteDescriptorSet> writes;
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &pageTableInfo
        });
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &minMaxImageInfo
        });
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &compactedIdBufferInfo
        });
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &countBufferInfo
        });
        
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layoutToUse, 0,
                                  static_cast<uint32_t>(writes.size()), writes.data());
    } else {
        // Regular pipeline
        VkDescriptorImageInfo minMaxImageInfo = {
            .sampler = sampler,
            .imageView = minMaxImageView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
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
        
        std::vector<VkWriteDescriptorSet> writes;
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &minMaxImageInfo
        });
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &compactedIdBufferInfo
        });
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &countBufferInfo
        });
        
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layoutToUse, 0,
                                  static_cast<uint32_t>(writes.size()), writes.data());
    }

    // Push Constants
    if (streamingPipelineLayout_ != VK_NULL_HANDLE) {
        // Streaming push constants matching shader
        struct StreamingPushConstants {
            uint32_t pageCoordX;
            uint32_t pageCoordY;
            uint32_t pageCoordZ;
            uint32_t mipLevel;
            float isoValue;
            uint32_t blockSize;
            uint32_t pageSizeX;
            uint32_t pageSizeY;
            uint32_t pageSizeZ;
            uint32_t volumeSizeX;
            uint32_t volumeSizeY;
            uint32_t volumeSizeZ;
        } streamingPC = {
            pageCoord.x,
            pageCoord.y,
            pageCoord.z,
            0, // mipLevel
            pushConstants.isovalue,
            4, // blockSize
            64, // pageSizeX
            32, // pageSizeY
            32, // pageSizeZ
            pushConstants.volumeDim.x,
            pushConstants.volumeDim.y,
            pushConstants.volumeDim.z
        };
        
        vkCmdPushConstants(cmd, layoutToUse, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(streamingPC), &streamingPC);
    } else {
        vkCmdPushConstants(cmd, layoutToUse, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PushConstants), &pushConstants);
    }

    // Calculate dispatch size for streaming (page-aware)
    if (streamingPipelineLayout_ != VK_NULL_HANDLE) {
        // For streaming, process blocks within a single page
        uint32_t pageBlocksX = 32 / pushConstants.blockDim.x;
        uint32_t pageBlocksY = 32 / pushConstants.blockDim.y;
        uint32_t pageBlocksZ = 32 / pushConstants.blockDim.z;
        uint32_t pageBlocks = pageBlocksX * pageBlocksY * pageBlocksZ;
        
        uint32_t localSizeX = 128;
        uint32_t groupCountX = (pageBlocks + localSizeX - 1) / localSizeX;
        
        // std::cout << "Dispatching " << groupCountX << " streaming workgroups for page (" 
        //           << pageCoord.x << ", " << pageCoord.y << ", " << pageCoord.z << ")..." << std::endl;
        vkCmdDispatch(cmd, groupCountX, 1, 1);
    } else {
        // Fall back to regular dispatch
        uint32_t totalBlocks = pushConstants.blockGridDim.x * pushConstants.blockGridDim.y * pushConstants.blockGridDim.z;
        uint32_t localSizeX = 128;
        uint32_t groupCountX = (totalBlocks + localSizeX - 1) / localSizeX;
        
        std::cout << "Dispatching " << groupCountX << " workgroups (" << totalBlocks << " total blocks)..." << std::endl;
        vkCmdDispatch(cmd, groupCountX, 1, 1);
    }
}