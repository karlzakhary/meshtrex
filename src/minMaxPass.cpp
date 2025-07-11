#include "minMaxPass.h"
#include "vulkan_context.h"
#include "streamingShaderInterface.h"
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

MinMaxPass::MinMaxPass(const VulkanContext& context, 
                       const char* leafShaderPath, 
                       const char* octreeShaderPath,
                       const char* streamingLeafShaderPath,
                       const char* streamingOctreeShaderPath) :
    context_(context),
    device_(context.getDevice())
{
    // Create regular pipelines
    createLeafPipelineLayout();
    createLeafPipeline(leafShaderPath);
    createOctreePipelineLayout();
    createOctreePipeline(octreeShaderPath);
    
    // Create streaming pipelines if paths provided
    if (streamingLeafShaderPath && streamingOctreeShaderPath) {
        createStreamingPipelineLayouts();
        
        // Load streaming shaders
        if (loadShader(streamingLeafMinMaxComputeShader_, device_, streamingLeafShaderPath)) {
            if (loadShader(streamingOctreeMinMaxComputeShader_, device_, streamingOctreeShaderPath)) {
                createStreamingPipelines();
            }
        }
    }
}

MinMaxPass::~MinMaxPass() {
    // Destroy regular pipelines
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
    
    // Destroy streaming pipelines
    if (streamingLeafPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, streamingLeafPipeline_, nullptr);
    }
    if (streamingLeafMinMaxComputeShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, streamingLeafMinMaxComputeShader_.module, nullptr);
    }
    if (streamingOctreePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, streamingOctreePipeline_, nullptr);
    }
    if (streamingOctreeMinMaxComputeShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, streamingOctreeMinMaxComputeShader_.module, nullptr);
    }
    if (streamingLeafPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, streamingLeafPipelineLayout_, nullptr);
    }
    if (streamingOctreePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, streamingOctreePipelineLayout_, nullptr);
    }
    if (streamingLeafDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, streamingLeafDescriptorSetLayout_, nullptr);
    }
    if (streamingOctreeDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, streamingOctreeDescriptorSetLayout_, nullptr);
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

// Streaming implementations
void MinMaxPass::recordStreamingLeafDispatch(VkCommandBuffer cmd,
                                            VkImageView volumeAtlasView,
                                            VkSampler volumeSampler,
                                            const Buffer& pageTableBuffer,
                                            VkImageView minMaxView,
                                            const PushConstants& pushConstants,
                                            const PageCoord& pageCoord) {
    // The streaming pipeline must be available
    assert(streamingLeafPipeline_ != VK_NULL_HANDLE);
    assert(streamingLeafPipelineLayout_ != VK_NULL_HANDLE);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, streamingLeafPipeline_);

    // Create descriptors for all 3 bindings
    VkDescriptorBufferInfo pageTableInfo = {
        .buffer = pageTableBuffer.buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };
    
    VkDescriptorImageInfo atlasImageInfo = {
        .sampler = volumeSampler,
        .imageView = volumeAtlasView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    
    VkDescriptorImageInfo outputImageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = minMaxView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    
    std::vector<VkWriteDescriptorSet> writes(3);
    
    writes[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &pageTableInfo
    };
    
    writes[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &atlasImageInfo
    };
    
    writes[2] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &outputImageInfo
    };
    
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              streamingLeafPipelineLayout_, 0, 3, writes.data());

    // Use the StreamingMinMaxPushConstants struct that matches the shader
    vkCmdPushConstants(cmd, streamingLeafPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pushConstants),
                       &pushConstants);

    // Dispatch - for streaming, we process one page worth of blocks
    // Calculate dispatch size for single page (64x32x32 voxels)
    // Each workgroup processes a 4x4x4 block
    uint32_t pageBlocksX = 64 / 4;  // 16
    uint32_t pageBlocksY = 32 / 4;  // 8
    uint32_t pageBlocksZ = 32 / 4;  // 8
    vkCmdDispatch(cmd, pageBlocksX, pageBlocksY, pageBlocksZ);
}

void MinMaxPass::recordStreamingOctreeDispatch(VkCommandBuffer cmd,
                                              VkImageView srcView,
                                              VkExtent3D srcExtent,
                                              VkImageView dstView,
                                              VkExtent3D dstExtent,
                                              const Buffer& pageTableBuffer,
                                              const PageCoord& pageCoord) {
    // The streaming pipeline must be available
    assert(streamingOctreePipeline_ != VK_NULL_HANDLE);
    assert(streamingOctreePipelineLayout_ != VK_NULL_HANDLE);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, streamingOctreePipeline_);
    
    // Prepare push constants matching shader (identical to non-streaming)
    struct StreamingOctreePushConstants {
        uint32_t srcDimX;
        uint32_t srcDimY;
        uint32_t srcDimZ;
        uint32_t dstDimX;
        uint32_t dstDimY;
        uint32_t dstDimZ;
    } pc = {
        srcExtent.width,
        srcExtent.height,
        srcExtent.depth,
        dstExtent.width,
        dstExtent.height,
        dstExtent.depth
    };
    
    // Setup descriptors for streaming pipeline (identical to non-streaming)
    VkDescriptorImageInfo srcImageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = srcView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    
    VkDescriptorImageInfo dstImageInfo = {
        .sampler = VK_NULL_HANDLE,
        .imageView = dstView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    
    std::vector<VkWriteDescriptorSet> writes(2);
    
    // Binding 0: Src MinMax Image
    writes[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &srcImageInfo
    };
    
    // Binding 1: Dst MinMax Image
    writes[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dstImageInfo
    };
    
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              streamingOctreePipelineLayout_, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());
    
    vkCmdPushConstants(cmd, streamingOctreePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Dispatch - one workgroup per destination texel
    vkCmdDispatch(cmd, dstExtent.width, dstExtent.height, dstExtent.depth);
}

void MinMaxPass::createStreamingPipelineLayouts() {
    // Create streaming leaf pipeline layout with page table
    {
        VkPushConstantRange pcRange = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(StreamingMinMaxPushConstants) // 15 uint32_t values including granularity
        };

        // Create descriptor set layout with all 3 bindings
        std::vector<VkDescriptorSetLayoutBinding> bindings(3);
        
        // Binding 0: Page Table SSBO
        bindings[0] = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };

        // Binding 1: Input Volume Atlas Image (Sparse 3D texture)
        bindings[1] = {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };

        // Binding 2: Output MinMax Image
        bindings[2] = {
            .binding = 2,
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

        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &streamingLeafDescriptorSetLayout_));

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &streamingLeafDescriptorSetLayout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange
        };

        VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &streamingLeafPipelineLayout_));
    }

    // Create streaming octree pipeline layout
    {
        VkPushConstantRange pcRange = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 28 // 6 uint32_t values + padding (uvec3 is padded to 16 bytes)
        };

        std::vector<VkDescriptorSetLayoutBinding> bindings(2); // Src, Dst only

        // Binding 0: Src MinMax Image
        bindings[0] = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };

        // Binding 1: Dest MinMax Image
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

        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &streamingOctreeDescriptorSetLayout_));

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &streamingOctreeDescriptorSetLayout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange
        };

        VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &streamingOctreePipelineLayout_));
    }
}

void MinMaxPass::createStreamingPipelines() {
    // Create streaming leaf pipeline
    if (streamingLeafMinMaxComputeShader_.module != VK_NULL_HANDLE) {
        streamingLeafPipeline_ = createComputePipeline(device_, nullptr, streamingLeafMinMaxComputeShader_, streamingLeafPipelineLayout_);
        assert(streamingLeafPipeline_ != VK_NULL_HANDLE);
    }
    
    // Create streaming octree pipeline
    if (streamingOctreeMinMaxComputeShader_.module != VK_NULL_HANDLE) {
        streamingOctreePipeline_ = createComputePipeline(device_, nullptr, streamingOctreeMinMaxComputeShader_, streamingOctreePipelineLayout_);
        assert(streamingOctreePipeline_ != VK_NULL_HANDLE);
    }
}