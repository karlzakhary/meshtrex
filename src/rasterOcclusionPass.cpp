#include "common.h"
#include "rasterOcclusionPass.h"
#include "vulkan_context.h"
#include "vulkan_utils.h"
#include "shaders.h"
#include "minMaxOutput.h"
#include "filteringOutput.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <algorithm>
#include <cstring>

RasterOcclusionPass::RasterOcclusionPass(const VulkanContext& context) 
    : context_(context), device_(context.getDevice()) {
    loadShaders();
    createPipelineLayout();
    createOcclusionPipeline();
    createVisibilityCompactionPipeline();
    createBuildOutputPipeline();
    createIndirectDrawBuffer();
    createIndirectUpdatePipeline();
    useIndirectDraw_ = true;  // Enable indirect drawing by default
    createPersistentResources();
    
    // Create debug statistics buffer
    createBuffer(debugStatsBuffer_, device_, context.getMemoryProperties(),
        sizeof(DebugStats),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

RasterOcclusionPass::~RasterOcclusionPass() {
    if (occlusionPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, occlusionPipeline_, nullptr);
    }
    if (occlusionPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, occlusionPipelineLayout_, nullptr);
    }
    if (occlusionDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, occlusionDescriptorSetLayout_, nullptr);
    }
    
    if (visibilityCompactionPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, visibilityCompactionPipeline_, nullptr);
    }
    if (visibilityCompactionPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, visibilityCompactionPipelineLayout_, nullptr);
    }
    if (visibilityCompactionDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, visibilityCompactionDescriptorSetLayout_, nullptr);
    }
    
    if (buildOutputPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, buildOutputPipeline_, nullptr);
    }
    if (buildOutputPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, buildOutputPipelineLayout_, nullptr);
    }
    if (buildOutputDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, buildOutputDescriptorSetLayout_, nullptr);
    }
    
    if (taskShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, taskShader_, nullptr);
    }
    if (meshShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, meshShader_, nullptr);
    }
    if (fragmentShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, fragmentShader_, nullptr);
    }
    if (visibilityCompactionShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, visibilityCompactionShader_, nullptr);
    }
    if (buildOutputShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, buildOutputShader_, nullptr);
    }
    
    // Cleanup debug buffer
    if (debugStatsBuffer_.buffer != VK_NULL_HANDLE) {
        destroyBuffer(debugStatsBuffer_, device_);
    }
    
    // Cleanup indirect draw resources
    if (indirectDrawBuffer_.buffer != VK_NULL_HANDLE) {
        destroyBuffer(indirectDrawBuffer_, device_);
    }
    if (indirectUpdatePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, indirectUpdatePipeline_, nullptr);
    }
    if (indirectUpdatePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, indirectUpdatePipelineLayout_, nullptr);
    }
    if (indirectUpdateDescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, indirectUpdateDescriptorSetLayout_, nullptr);
    }
    if (indirectUpdateComputeShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, indirectUpdateComputeShader_, nullptr);
    }
    
    // Destroy persistent resources
    destroyPersistentResources();
}

void RasterOcclusionPass::loadShaders() {
    // Load shaders using Shader structures
    Shader taskShaderData{}, meshShaderData{}, fragmentShaderData{}, compactionShaderData{};
    
    assert(loadShader(taskShaderData, device_, "/spirv/raster_occlusion.task.spv"));
    taskShader_ = taskShaderData.module;
    
    assert(loadShader(meshShaderData, device_, "/spirv/raster_occlusion.mesh.spv"));
    meshShader_ = meshShaderData.module;
    
    assert(loadShader(fragmentShaderData, device_, "/spirv/raster_occlusion.frag.spv"));
    fragmentShader_ = fragmentShaderData.module;
    
    assert(loadShader(compactionShaderData, device_, "/spirv/visibility_compaction.comp.spv"));
    visibilityCompactionShader_ = compactionShaderData.module;
    
    Shader buildOutputShaderData{};
    assert(loadShader(buildOutputShaderData, device_, "/spirv/build_pvs_output.comp.spv"));
    buildOutputShader_ = buildOutputShaderData.module;
}

void RasterOcclusionPass::createPipelineLayout() {
    // Descriptor set layout for occlusion culling pipeline
    std::vector<VkDescriptorSetLayoutBinding> occlusionBindings = {
        // Binding 0: UBO (view uniforms)
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, 
         VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        // Binding 1: Min-max hierarchy texture
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, 
         VK_SHADER_STAGE_TASK_BIT_EXT, nullptr},
        // Binding 2: Visibility buffer SSBO
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, 
         VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        // Binding 3: Debug statistics buffer (optional)
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
         VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    };
    
    VkDescriptorSetLayoutCreateInfo occlusionLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    occlusionLayoutInfo.bindingCount = static_cast<uint32_t>(occlusionBindings.size());
    occlusionLayoutInfo.pBindings = occlusionBindings.data();
    
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &occlusionLayoutInfo, nullptr, &occlusionDescriptorSetLayout_));
    
    // Push constants for view-projection matrix
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4); // view-proj matrix
    
    VkPipelineLayoutCreateInfo occlusionPipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    occlusionPipelineLayoutInfo.setLayoutCount = 1;
    occlusionPipelineLayoutInfo.pSetLayouts = &occlusionDescriptorSetLayout_;
    occlusionPipelineLayoutInfo.pushConstantRangeCount = 1;
    occlusionPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    VK_CHECK(vkCreatePipelineLayout(device_, &occlusionPipelineLayoutInfo, nullptr, &occlusionPipelineLayout_));
    
    // Descriptor set layout for visibility compaction pipeline
    std::vector<VkDescriptorSetLayoutBinding> visibilityCompactionBindings = {
        // Binding 0: Visibility buffer (input)
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        // Binding 1: Compacted bitfield (output)
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    
    VkDescriptorSetLayoutCreateInfo visCompactionLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    visCompactionLayoutInfo.bindingCount = static_cast<uint32_t>(visibilityCompactionBindings.size());
    visCompactionLayoutInfo.pBindings = visibilityCompactionBindings.data();
    
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &visCompactionLayoutInfo, nullptr, &visibilityCompactionDescriptorSetLayout_));
    
    // Push constants for visibility compaction
    VkPushConstantRange visCompactionPushConstant{};
    visCompactionPushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    visCompactionPushConstant.offset = 0;
    visCompactionPushConstant.size = sizeof(uint32_t); // total blocks
    
    VkPipelineLayoutCreateInfo visCompactionPipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    visCompactionPipelineLayoutInfo.setLayoutCount = 1;
    visCompactionPipelineLayoutInfo.pSetLayouts = &visibilityCompactionDescriptorSetLayout_;
    visCompactionPipelineLayoutInfo.pushConstantRangeCount = 1;
    visCompactionPipelineLayoutInfo.pPushConstantRanges = &visCompactionPushConstant;
    
    VK_CHECK(vkCreatePipelineLayout(device_, &visCompactionPipelineLayoutInfo, nullptr, &visibilityCompactionPipelineLayout_));
    
    // Descriptor set layout for build output pipeline
    std::vector<VkDescriptorSetLayoutBinding> buildOutputBindings = {
        // Binding 0: Current frame bitfield (input)
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        // Binding 1: Previous frame bitfield (input)
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        // Binding 2: PVS current buffer (output)
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        // Binding 3: PVS difference buffer (output)
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    
    VkDescriptorSetLayoutCreateInfo buildOutputLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    buildOutputLayoutInfo.bindingCount = static_cast<uint32_t>(buildOutputBindings.size());
    buildOutputLayoutInfo.pBindings = buildOutputBindings.data();
    
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &buildOutputLayoutInfo, nullptr, &buildOutputDescriptorSetLayout_));
    
    // Push constants for build output
    VkPushConstantRange buildOutputPushConstant{};
    buildOutputPushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    buildOutputPushConstant.offset = 0;
    buildOutputPushConstant.size = sizeof(uint32_t); // num bitfield entries
    
    VkPipelineLayoutCreateInfo buildOutputPipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    buildOutputPipelineLayoutInfo.setLayoutCount = 1;
    buildOutputPipelineLayoutInfo.pSetLayouts = &buildOutputDescriptorSetLayout_;
    buildOutputPipelineLayoutInfo.pushConstantRangeCount = 1;
    buildOutputPipelineLayoutInfo.pPushConstantRanges = &buildOutputPushConstant;
    
    VK_CHECK(vkCreatePipelineLayout(device_, &buildOutputPipelineLayoutInfo, nullptr, &buildOutputPipelineLayout_));
}

void RasterOcclusionPass::createOcclusionPipeline() {
    // Create mesh shader pipeline for occlusion culling
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        // Task shader
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_TASK_BIT_EXT, taskShader_, "main", nullptr},
        // Mesh shader
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_MESH_BIT_EXT, meshShader_, "main", nullptr},
        // Fragment shader
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader_, "main", nullptr}
    };
    
    // No vertex input
    VkPipelineVertexInputStateCreateInfo vertexInputState{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    
    // Input assembly (not used with mesh shaders)
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    
    // Viewport state
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    // Rasterization state
    VkPipelineRasterizationStateCreateInfo rasterizationState{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationState.cullMode = VK_CULL_MODE_NONE; // No culling for proxy geometry
    rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationState.lineWidth = 1.0f;
    // Depth bias similar to Kreskowski's glPolygonOffset(-1, -60)
    // In Vulkan, positive values push away from camera (increase depth)
    // We use positive values to push proxy quads slightly behind for proper occlusion
    rasterizationState.depthBiasEnable = VK_TRUE;
    rasterizationState.depthBiasConstantFactor = 60.0f; // Push back to avoid z-fighting with actual geometry
    rasterizationState.depthBiasSlopeFactor = 1.0f;
    
    // Multisample state
    VkPipelineMultisampleStateCreateInfo multisampleState{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    // Depth stencil state - enable early depth test
    VkPipelineDepthStencilStateCreateInfo depthStencilState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencilState.depthTestEnable = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_FALSE; // Don't write to depth
    depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;  // Reversed-Z: greater values are closer
    
    // Color blend state - no color output
    VkPipelineColorBlendStateCreateInfo colorBlendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlendState.attachmentCount = 0; // No color attachments
    
    // Dynamic state
    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();
    
    // Pipeline create info
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputState;
    pipelineInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizationState;
    pipelineInfo.pMultisampleState = &multisampleState;
    pipelineInfo.pDepthStencilState = &depthStencilState;
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = occlusionPipelineLayout_;
    
    // Rendering info for dynamic rendering
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT; // Assuming D32 depth format
    pipelineInfo.pNext = &renderingInfo;
    
    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &occlusionPipeline_));
}

void RasterOcclusionPass::createVisibilityCompactionPipeline() {
    // Create compute pipeline for visibility buffer compaction to bitfield
    VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = visibilityCompactionShader_;
    shaderStage.pName = "main";
    
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = visibilityCompactionPipelineLayout_;
    
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &visibilityCompactionPipeline_));
}

void RasterOcclusionPass::createBuildOutputPipeline() {
    // Create compute pipeline for building PVS output from bitfield
    VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = buildOutputShader_;
    shaderStage.pName = "main";
    
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = buildOutputPipelineLayout_;
    
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &buildOutputPipeline_));
}

void RasterOcclusionPass::createVisibilityBuffer(Output& output, uint32_t numBlocks) {
    // Create visibility buffer - one uint per block
    size_t bufferSize = numBlocks * sizeof(uint32_t);
    
    createBuffer(output.visibilityBuffer, device_, context_.getMemoryProperties(),
                bufferSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void RasterOcclusionPass::createBitfieldBuffers(Output& output, uint32_t numBlocks) {
    // Create bitfield buffers - 1 bit per block, so divide by 32
    uint32_t numBitfieldEntries = (numBlocks + 31) / 32;
    size_t bufferSize = numBitfieldEntries * sizeof(uint32_t);
    
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    
    // Current frame bitfield
    createBuffer(output.currentBitfieldBuffer, device_, context_.getMemoryProperties(),
                bufferSize, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Previous frame bitfield
    createBuffer(output.previousBitfieldBuffer, device_, context_.getMemoryProperties(),
                bufferSize, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

RasterOcclusionPass::Output RasterOcclusionPass::performOcclusionCulling(
    VkCommandBuffer cmd,
    const MinMaxOutput& minMaxOutput,
    const FilteringOutput& previousPVS,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    VkImageView depthImageView,
    VkExtent2D renderExtent,
    bool ownCommandBuffer) {
    
    Output output;
    
    // Calculate total number of blocks
    uint32_t totalBlocks = pushConstants.blockGridDim.x * 
                          pushConstants.blockGridDim.y * 
                          pushConstants.blockGridDim.z;
    
    // Create visibility buffer
    createVisibilityBuffer(output, totalBlocks);
    
    // Create bitfield buffers
    createBitfieldBuffers(output, totalBlocks);
    
    // Create PVS output buffers
    VkDeviceSize pvsBufferSize = (totalBlocks + 1) * sizeof(uint32_t); // +1 for count
    
    VkBufferUsageFlags pvsUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    
    // PVS current buffer
    createBuffer(output.pvsCurrentBuffer, device_, context_.getMemoryProperties(),
                pvsBufferSize, pvsUsage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // PVS difference buffer
    createBuffer(output.pvsDifferenceBuffer, device_, context_.getMemoryProperties(),
                pvsBufferSize, pvsUsage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Clear buffers
    vkCmdFillBuffer(cmd, output.visibilityBuffer.buffer, 0, output.visibilityBuffer.size, 0);
    vkCmdFillBuffer(cmd, output.currentBitfieldBuffer.buffer, 0, output.currentBitfieldBuffer.size, 0);
    vkCmdFillBuffer(cmd, output.pvsCurrentBuffer.buffer, 0, sizeof(uint32_t), 0); // Clear count
    vkCmdFillBuffer(cmd, output.pvsDifferenceBuffer.buffer, 0, sizeof(uint32_t), 0); // Clear count
    
    // Memory barrier after clear
    VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &clearBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    
    // Create uniform buffer for view parameters
    struct ViewUniforms {
        alignas(16) glm::mat4 viewProj;
        alignas(16) glm::uvec4 volumeDim;
        alignas(16) glm::uvec4 blockDim;
        alignas(16) glm::uvec4 blockGridDim;
        alignas(4) float isovalue;
    } viewUniforms;
    
    viewUniforms.viewProj = viewProjMatrix;
    viewUniforms.volumeDim = pushConstants.volumeDim;
    viewUniforms.blockDim = pushConstants.blockDim;
    viewUniforms.blockGridDim = pushConstants.blockGridDim;
    viewUniforms.isovalue = pushConstants.isovalue;
    
    // Create uniform buffer using Buffer struct
    createBuffer(output.tempResources.uniformBuffer, device_, context_.getMemoryProperties(),
                sizeof(ViewUniforms),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy data to uniform buffer (data is already mapped for host-visible buffers)
    memcpy(output.tempResources.uniformBuffer.data, &viewUniforms, sizeof(ViewUniforms));
    
    // Allocate descriptor sets from persistent pool
    VkDescriptorSet occlusionDescriptorSet, visibilityCompactionDescriptorSet, buildOutputDescriptorSet;
    std::array<VkDescriptorSetLayout, 3> layouts = {
        occlusionDescriptorSetLayout_, 
        visibilityCompactionDescriptorSetLayout_,
        buildOutputDescriptorSetLayout_
    };
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = persistentDescriptorPools_[currentFrameIndex_];
    allocInfo.descriptorSetCount = 3;
    allocInfo.pSetLayouts = layouts.data();
    
    std::array<VkDescriptorSet, 3> sets;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, sets.data()));
    occlusionDescriptorSet = sets[0];
    visibilityCompactionDescriptorSet = sets[1];
    buildOutputDescriptorSet = sets[2];
    
    // Update occlusion descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    
    // Binding 0: View uniforms
    VkDescriptorBufferInfo viewUboInfo{output.tempResources.uniformBuffer.buffer, 0, sizeof(ViewUniforms)};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet, 
                     0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewUboInfo, nullptr});
    
    // Binding 1: Min-max hierarchy texture
    VkDescriptorImageInfo minMaxInfo{minMaxOutput.minMaxSampler, minMaxOutput.minMaxImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &minMaxInfo, nullptr, nullptr});
    
    // Binding 2: Visibility buffer
    VkDescriptorBufferInfo visibilityInfo{output.visibilityBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibilityInfo, nullptr});
    
    // Binding 3: Debug statistics buffer (optional)
    VkDescriptorBufferInfo debugInfo{debugStatsBuffer_.buffer, 0, sizeof(DebugStats)};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &debugInfo, nullptr});
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Set viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderExtent.width);
    viewport.height = static_cast<float>(renderExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = renderExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    
    // Begin dynamic rendering with depth attachment only
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Load existing depth
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = scissor;
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = &depthAttachment;
    
    // If using indirect draw, update the indirect buffer BEFORE beginning rendering
    if (useIndirectDraw_) {
        updateIndirectDrawBufferGPU(cmd, totalBlocks);
    }
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Bind occlusion pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipelineLayout_,
                           0, 1, &occlusionDescriptorSet, 0, nullptr);
    
    // Push view-projection matrix
    vkCmdPushConstants(cmd, occlusionPipelineLayout_, 
                      VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                      0, sizeof(glm::mat4), &viewProjMatrix);
    
    if (useIndirectDraw_) {
        // Use indirect draw - GPU calculates workgroup count
        vkCmdDrawMeshTasksIndirectEXT(cmd, indirectDrawBuffer_.buffer, 0, 1, 0);
    } else {
        // Dispatch task shaders - blocks are processed in 8x8x8 groups, split into two 8x8x4 halves
        // So we need 2 workgroups per 8x8x8 = 512 blocks
        uint32_t blocksPerGroup = 8 * 8 * 8; // 512
        uint32_t numGroups = (totalBlocks + blocksPerGroup - 1) / blocksPerGroup;
        uint32_t numWorkgroups = numGroups * 2; // 2 workgroups per 8x8x8 group
        
        // Debug output for first few frames
        static int occlusionFrame = 0;
        
        vkCmdDrawMeshTasksEXT(cmd, numWorkgroups, 1, 1);
    }
    
    // End rendering
    vkCmdEndRendering(cmd);
    
    // Barrier for visibility buffer before compaction
    VkMemoryBarrier2 visibilityBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    visibilityBarrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    visibilityBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    visibilityBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    visibilityBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    
    VkDependencyInfo visDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    visDepInfo.memoryBarrierCount = 1;
    visDepInfo.pMemoryBarriers = &visibilityBarrier;
    vkCmdPipelineBarrier2(cmd, &visDepInfo);
    
    // Stage 1: Visibility compaction to bitfield
    writes.clear();
    
    // Binding 0: Visibility buffer (input)
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, visibilityCompactionDescriptorSet,
                     0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibilityInfo, nullptr});
    
    // Binding 1: Compacted bitfield (output)
    VkDescriptorBufferInfo currentBitfieldInfo{output.currentBitfieldBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, visibilityCompactionDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &currentBitfieldInfo, nullptr});
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Dispatch visibility compaction
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visibilityCompactionPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visibilityCompactionPipelineLayout_,
                           0, 1, &visibilityCompactionDescriptorSet, 0, nullptr);
    
    // Push total blocks count
    vkCmdPushConstants(cmd, visibilityCompactionPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(uint32_t), &totalBlocks);
    
    // Dispatch with 1024 threads per workgroup
    uint32_t compactionWorkgroups = (totalBlocks + 1023) / 1024;
    vkCmdDispatch(cmd, compactionWorkgroups, 1, 1);
    
    // Barrier between stages
    VkMemoryBarrier2 stageBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    stageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    stageBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    stageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    stageBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    
    VkDependencyInfo stageDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    stageDepInfo.memoryBarrierCount = 1;
    stageDepInfo.pMemoryBarriers = &stageBarrier;
    vkCmdPipelineBarrier2(cmd, &stageDepInfo);
    
    // Copy current bitfield to previous for next frame
    // (In a real implementation, you'd swap buffers instead of copying)
    if (previousPVS.compactedBlockIdBuffer.buffer != VK_NULL_HANDLE) {
        // For now, use the previous PVS bitfield as-is
        // In production, maintain separate bitfield for previous frame
    }
    
    // Stage 2: Build PVS output from bitfield
    writes.clear();
    
    // Binding 0: Current frame bitfield (input)
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &currentBitfieldInfo, nullptr});
    
    // Binding 1: Previous frame bitfield (input)
    // For now, use the same buffer - in production, maintain separate previous frame bitfield
    VkDescriptorBufferInfo previousBitfieldInfo{output.previousBitfieldBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &previousBitfieldInfo, nullptr});
    
    // Binding 2: PVS current buffer (output)
    VkDescriptorBufferInfo pvsCurrentInfo{output.pvsCurrentBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pvsCurrentInfo, nullptr});
    
    // Binding 3: PVS difference buffer (output)
    VkDescriptorBufferInfo pvsDifferenceInfo{output.pvsDifferenceBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pvsDifferenceInfo, nullptr});
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Dispatch build output
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, buildOutputPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, buildOutputPipelineLayout_,
                           0, 1, &buildOutputDescriptorSet, 0, nullptr);
    
    // Push number of bitfield entries
    uint32_t numBitfieldEntries = (totalBlocks + 31) / 32;
    vkCmdPushConstants(cmd, buildOutputPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(uint32_t), &numBitfieldEntries);
    
    // Dispatch with 32 threads per workgroup (warp size)
    uint32_t buildOutputWorkgroups = (numBitfieldEntries + 31) / 32;
    vkCmdDispatch(cmd, buildOutputWorkgroups, 1, 1);
    
    // Final barrier
    VkMemoryBarrier2 finalBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    finalBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    finalBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    
    VkDependencyInfo finalDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    finalDepInfo.memoryBarrierCount = 1;
    finalDepInfo.pMemoryBarriers = &finalBarrier;
    vkCmdPipelineBarrier2(cmd, &finalDepInfo);
    
    return output;
}

void RasterOcclusionPass::Output::destroy(VkDevice device) {
    // Clean up temporary resources first
    tempResources.destroy(device);
    
    if (visibilityBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(visibilityBuffer, device);
    }
    if (currentBitfieldBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(currentBitfieldBuffer, device);
    }
    if (previousBitfieldBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(previousBitfieldBuffer, device);
    }
    if (pvsCurrentBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(pvsCurrentBuffer, device);
    }
    if (pvsDifferenceBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(pvsDifferenceBuffer, device);
    }
    if (pvsPreviousBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(pvsPreviousBuffer, device);
    }
}

void RasterOcclusionPass::Output::swapTemporalBuffers() {
    // Swap current PVS to become previous PVS for next frame
    std::swap(pvsPreviousBuffer, pvsCurrentBuffer);
    std::swap(pvsPreviousCount, pvsCurrentCount);
    
    // Swap bitfield buffers as well
    std::swap(previousBitfieldBuffer, currentBitfieldBuffer);
    
    // Increment frame index
    frameIndex++;
    // Note: isFirstFrame is now set to false in the main render loop after rendering
}

void RasterOcclusionPass::Output::copyCurrentToPrevious(VkDevice device, VkCommandBuffer cmd) {
    // Copy current bitfield to previous bitfield (only if buffers exist - may not exist in compute-only mode)
    if (currentBitfieldBuffer.buffer != VK_NULL_HANDLE && previousBitfieldBuffer.buffer != VK_NULL_HANDLE) {
        VkBufferCopy bitfieldCopy{};
        bitfieldCopy.size = currentBitfieldBuffer.size;
        vkCmdCopyBuffer(cmd, currentBitfieldBuffer.buffer, previousBitfieldBuffer.buffer, 1, &bitfieldCopy);
    }
    
    // Copy current PVS to previous PVS (check for null in case of compute-only initialization)
    // Copy the entire buffer to ensure all data is transferred
    // The buffer size includes space for the count + all block indices
    if (pvsCurrentBuffer.buffer != VK_NULL_HANDLE && pvsPreviousBuffer.buffer != VK_NULL_HANDLE) {
        VkBufferCopy pvsCopy{};
        pvsCopy.size = pvsCurrentBuffer.size;
        vkCmdCopyBuffer(cmd, pvsCurrentBuffer.buffer, pvsPreviousBuffer.buffer, 1, &pvsCopy);
    }
    
    // Memory barrier to ensure copies complete
    VkMemoryBarrier2 copyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    copyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    copyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    copyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    copyBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &copyBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void RasterOcclusionPass::createPVSBuffers(Output& output, uint32_t maxBlocks) {
    VkDeviceSize pvsBufferSize = (maxBlocks + 1) * sizeof(uint32_t); // +1 for count
    
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    // Create PVS current buffer
    createBuffer(output.pvsCurrentBuffer, device_, context_.getMemoryProperties(),
                pvsBufferSize, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Create PVS difference buffer
    createBuffer(output.pvsDifferenceBuffer, device_, context_.getMemoryProperties(),
                pvsBufferSize, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Create PVS previous buffer (PVS_prev)
    createBuffer(output.pvsPreviousBuffer, device_, context_.getMemoryProperties(),
                pvsBufferSize, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void RasterOcclusionPass::initializeOutput(Output& output, uint32_t numBlocks) {
    // Create all necessary buffers
    createVisibilityBuffer(output, numBlocks);
    createBitfieldBuffers(output, numBlocks);
    createPVSBuffers(output, numBlocks);
    
    // Initialize temporal state
    output.frameIndex = 0;
    output.isFirstFrame = true;
    output.pvsCurrentCount = 0;
    output.pvsPreviousCount = 0;
    output.pvsDifferenceCount = 0;
}

bool RasterOcclusionPass::performTemporalOcclusionCulling(
    VkCommandBuffer cmd,
    Output& output,
    const MinMaxOutput& minMaxOutput,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    VkImageView depthImageView,
    VkExtent2D renderExtent,
    bool forceUpdate) {
    
    // Calculate total number of blocks
    uint32_t totalBlocks = pushConstants.blockGridDim.x * 
                          pushConstants.blockGridDim.y * 
                          pushConstants.blockGridDim.z;
    
    // Check if we need to update occlusion based on camera movement and PVS stability
    bool cameraChanged = false;
    bool shouldCheckOcclusion = false;
    
    if (!output.isFirstFrame) {
        // Decompose view-projection to analyze camera changes more precisely
        // Extract camera position from inverse view-projection
        glm::mat4 invViewProj = glm::inverse(viewProjMatrix);
        glm::mat4 invPrevViewProj = glm::inverse(output.previousViewProj);
        
        // Camera position is the translation part of inverse view matrix
        glm::vec3 currentCamPos = glm::vec3(invViewProj[3]);
        glm::vec3 prevCamPos = glm::vec3(invPrevViewProj[3]);
        
        // Check position change
        float positionDelta = glm::length(currentCamPos - prevCamPos);
        
        // Check rotation change by comparing view direction
        glm::vec3 currentViewDir = -glm::normalize(glm::vec3(invViewProj[2]));
        glm::vec3 prevViewDir = -glm::normalize(glm::vec3(invPrevViewProj[2]));
        float rotationDelta = glm::acos(glm::clamp(glm::dot(currentViewDir, prevViewDir), -1.0f, 1.0f));
        
        // Adaptive thresholds based on distance from volume
        // When zoomed in close, be less sensitive to small movements
        float distanceFromVolume = glm::length(currentCamPos - glm::vec3(128.0f, 128.0f, 128.0f)); // Assuming volume center
        float distanceScale = glm::clamp(distanceFromVolume / 500.0f, 0.5f, 2.0f);
        
        // Thresholds for camera movement (adaptive based on zoom)
        // Increase thresholds to reduce false positives from numerical errors
        const float BASE_POSITION_THRESHOLD = 2.0f;  // Increased from 1.0
        const float BASE_ROTATION_THRESHOLD = 0.05f;  // Increased from 0.02 (~3 degrees)
        
        float positionThreshold = BASE_POSITION_THRESHOLD * distanceScale;
        float rotationThreshold = BASE_ROTATION_THRESHOLD;
        
        // Debug: Log what's causing camera detection when zoomed in
        static int debugCounter = 0;
        bool positionChanged = positionDelta > positionThreshold;
        bool rotationChanged = rotationDelta > rotationThreshold;
        
        if ((positionChanged || rotationChanged) && distanceFromVolume < 100.0f && debugCounter++ < 20) {
            printf("  Camera detection: pos_delta=%.6f (thresh=%.3f) rot_delta=%.6f (thresh=%.3f) dist=%.1f\n",
                   positionDelta, positionThreshold, rotationDelta, rotationThreshold, distanceFromVolume);
        }
        
        cameraChanged = positionChanged || rotationChanged;
        
        // Be VERY conservative about occlusion updates to avoid z-fighting
        if (cameraChanged) {
            // Only update if camera moved significantly
            shouldCheckOcclusion = true;
        }
        else {
            // Camera hasn't moved - don't update unless absolutely necessary
            // No periodic updates - they cause z-fighting for no benefit
            shouldCheckOcclusion = false;
        }
    } else {
        shouldCheckOcclusion = true;  // Always check on first frame
    }
    
    // Determine if we need to run occlusion culling
    output.needsOcclusionUpdate = output.isFirstFrame || shouldCheckOcclusion || forceUpdate;
    
    // If no update needed, just increment frame counter and return
    if (!output.needsOcclusionUpdate) {
        output.framesSinceLastUpdate++;
        return false;  // No occlusion culling performed
    }
    
    // Reset frame counter when we do update
    output.framesSinceLastUpdate = 0;
    output.previousViewProj = viewProjMatrix;
    
    // Initialize output on first frame
    if (output.isFirstFrame || output.visibilityBuffer.buffer == VK_NULL_HANDLE) {
        initializeOutput(output, totalBlocks);
        
        // Clear the previous frame bitfield buffer on first frame
        vkCmdFillBuffer(cmd, output.previousBitfieldBuffer.buffer, 0, output.currentBitfieldBuffer.size, 0);
        
        VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        
        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &clearBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
    
    // Clear current frame buffers
    vkCmdFillBuffer(cmd, output.visibilityBuffer.buffer, 0, output.visibilityBuffer.size, 0);
    vkCmdFillBuffer(cmd, output.currentBitfieldBuffer.buffer, 0, output.currentBitfieldBuffer.size, 0);
    vkCmdFillBuffer(cmd, output.pvsCurrentBuffer.buffer, 0, sizeof(uint32_t), 0); // Clear count
    vkCmdFillBuffer(cmd, output.pvsDifferenceBuffer.buffer, 0, sizeof(uint32_t), 0); // Clear count
    
    // Memory barrier after clear
    VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &clearBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    
    // Create uniform buffer for view parameters
    struct ViewUniforms {
        alignas(16) glm::mat4 viewProj;
        alignas(16) glm::uvec4 volumeDim;
        alignas(16) glm::uvec4 blockDim;
        alignas(16) glm::uvec4 blockGridDim;
        alignas(4) float isovalue;
    } viewUniforms;
    
    viewUniforms.viewProj = viewProjMatrix;
    viewUniforms.volumeDim = pushConstants.volumeDim;
    viewUniforms.blockDim = pushConstants.blockDim;
    viewUniforms.blockGridDim = pushConstants.blockGridDim;
    viewUniforms.isovalue = pushConstants.isovalue;
    
    // Create uniform buffer using Buffer struct
    createBuffer(output.tempResources.uniformBuffer, device_, context_.getMemoryProperties(),
                sizeof(ViewUniforms),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy data to uniform buffer (data is already mapped for host-visible buffers)
    memcpy(output.tempResources.uniformBuffer.data, &viewUniforms, sizeof(ViewUniforms));
    
    // Allocate descriptor sets from persistent pool
    VkDescriptorSet occlusionDescriptorSet, visibilityCompactionDescriptorSet, buildOutputDescriptorSet;
    std::array<VkDescriptorSetLayout, 3> layouts = {
        occlusionDescriptorSetLayout_, 
        visibilityCompactionDescriptorSetLayout_,
        buildOutputDescriptorSetLayout_
    };
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = persistentDescriptorPools_[currentFrameIndex_];
    allocInfo.descriptorSetCount = 3;
    allocInfo.pSetLayouts = layouts.data();
    
    std::array<VkDescriptorSet, 3> sets;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, sets.data()));
    occlusionDescriptorSet = sets[0];
    visibilityCompactionDescriptorSet = sets[1];
    buildOutputDescriptorSet = sets[2];
    
    // Update occlusion descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    
    // Binding 0: View uniforms
    VkDescriptorBufferInfo viewUboInfo{output.tempResources.uniformBuffer.buffer, 0, sizeof(ViewUniforms)};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet, 
                     0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewUboInfo, nullptr});
    
    // Binding 1: Min-max hierarchy texture
    VkDescriptorImageInfo minMaxInfo{minMaxOutput.minMaxSampler, minMaxOutput.minMaxImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &minMaxInfo, nullptr, nullptr});
    
    // Binding 2: Visibility buffer
    VkDescriptorBufferInfo visibilityInfo{output.visibilityBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibilityInfo, nullptr});
    
    // Binding 3: Debug statistics buffer (optional)
    VkDescriptorBufferInfo debugInfo{debugStatsBuffer_.buffer, 0, sizeof(DebugStats)};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &debugInfo, nullptr});
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Set viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderExtent.width);
    viewport.height = static_cast<float>(renderExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = renderExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    
    // Begin dynamic rendering with depth attachment only
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = scissor;
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = &depthAttachment;
    
    // If using indirect draw, update the indirect buffer BEFORE beginning rendering
    if (useIndirectDraw_) {
        updateIndirectDrawBufferGPU(cmd, totalBlocks);
    }
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Bind occlusion pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipelineLayout_,
                           0, 1, &occlusionDescriptorSet, 0, nullptr);
    
    // Push view-projection matrix
    vkCmdPushConstants(cmd, occlusionPipelineLayout_, 
                      VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                      0, sizeof(glm::mat4), &viewProjMatrix);
    
    if (useIndirectDraw_) {
        // Use indirect draw - GPU calculates workgroup count
        vkCmdDrawMeshTasksIndirectEXT(cmd, indirectDrawBuffer_.buffer, 0, 1, 0);
    } else {
        // Dispatch task shaders
        uint32_t blocksPerGroup = pushConstants.blockDim.x * pushConstants.blockDim.y * pushConstants.blockDim.z; // 512
        uint32_t numGroups = (totalBlocks + blocksPerGroup - 1) / blocksPerGroup;
        uint32_t numWorkgroups = numGroups * 2; // 2 workgroups per block
        vkCmdDrawMeshTasksEXT(cmd, numWorkgroups, 1, 1);
    }
    
    // End rendering
    vkCmdEndRendering(cmd);
    
    // Barrier for visibility buffer before compaction
    VkMemoryBarrier2 visibilityBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    visibilityBarrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    visibilityBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    visibilityBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    visibilityBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    
    VkDependencyInfo visDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    visDepInfo.memoryBarrierCount = 1;
    visDepInfo.pMemoryBarriers = &visibilityBarrier;
    vkCmdPipelineBarrier2(cmd, &visDepInfo);
    
    // Stage 1: Visibility compaction to bitfield
    writes.clear();
    
    // Binding 0: Visibility buffer (input)
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, visibilityCompactionDescriptorSet,
                     0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibilityInfo, nullptr});
    
    // Binding 1: Compacted bitfield (output)
    VkDescriptorBufferInfo currentBitfieldInfo{output.currentBitfieldBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, visibilityCompactionDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &currentBitfieldInfo, nullptr});
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Dispatch visibility compaction
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visibilityCompactionPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visibilityCompactionPipelineLayout_,
                           0, 1, &visibilityCompactionDescriptorSet, 0, nullptr);
    
    // Push total blocks count
    vkCmdPushConstants(cmd, visibilityCompactionPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(uint32_t), &totalBlocks);
    
    // Dispatch with 1024 threads per workgroup
    uint32_t compactionWorkgroups = (totalBlocks + 1023) / 1024;
    vkCmdDispatch(cmd, compactionWorkgroups, 1, 1);
    
    // Barrier between stages
    VkMemoryBarrier2 stageBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    stageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    stageBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    stageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    stageBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    
    VkDependencyInfo stageDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    stageDepInfo.memoryBarrierCount = 1;
    stageDepInfo.pMemoryBarriers = &stageBarrier;
    vkCmdPipelineBarrier2(cmd, &stageDepInfo);
    
    // Stage 2: Build PVS output from bitfield
    writes.clear();
    
    // Binding 0: Current frame bitfield (input)
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &currentBitfieldInfo, nullptr});
    
    // Binding 1: Previous frame bitfield (input)
    VkDescriptorBufferInfo previousBitfieldInfo{output.previousBitfieldBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &previousBitfieldInfo, nullptr});
    
    // Binding 2: PVS current buffer (output)
    VkDescriptorBufferInfo pvsCurrentInfo{output.pvsCurrentBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pvsCurrentInfo, nullptr});
    
    // Binding 3: PVS difference buffer (output)
    VkDescriptorBufferInfo pvsDifferenceInfo{output.pvsDifferenceBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pvsDifferenceInfo, nullptr});
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Dispatch build output
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, buildOutputPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, buildOutputPipelineLayout_,
                           0, 1, &buildOutputDescriptorSet, 0, nullptr);
    
    // Push number of bitfield entries
    uint32_t numBitfieldEntries = (totalBlocks + 31) / 32;
    vkCmdPushConstants(cmd, buildOutputPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(uint32_t), &numBitfieldEntries);
    
    // Dispatch with 32 threads per workgroup (warp size)
    uint32_t buildOutputWorkgroups = (numBitfieldEntries + 31) / 32;
    vkCmdDispatch(cmd, buildOutputWorkgroups, 1, 1);
    
    // Final barrier
    VkMemoryBarrier2 finalBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    finalBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    finalBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    
    VkDependencyInfo finalDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    finalDepInfo.memoryBarrierCount = 1;
    finalDepInfo.pMemoryBarriers = &finalBarrier;
    vkCmdPipelineBarrier2(cmd, &finalDepInfo);
    
    // Create readback buffer for PVS counts using Buffer struct
    createBuffer(output.tempResources.readbackBuffer, device_, context_.getMemoryProperties(),
                3 * sizeof(uint32_t), // For prev, current, and difference counts
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy counts from GPU buffers to readback buffer
    VkBufferCopy countCopy{};
    countCopy.size = sizeof(uint32_t);
    
    // Copy previous count
    countCopy.srcOffset = 0;
    countCopy.dstOffset = 0;
    vkCmdCopyBuffer(cmd, output.pvsPreviousBuffer.buffer, output.tempResources.readbackBuffer.buffer, 1, &countCopy);
    
    // Copy current count
    countCopy.srcOffset = 0;
    countCopy.dstOffset = sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, output.pvsCurrentBuffer.buffer, output.tempResources.readbackBuffer.buffer, 1, &countCopy);
    
    // Copy difference count
    countCopy.srcOffset = 0;
    countCopy.dstOffset = 2 * sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, output.pvsDifferenceBuffer.buffer, output.tempResources.readbackBuffer.buffer, 1, &countCopy);
    
    // Copy current frame data to previous frame for next frame's temporal coherence
    // This ensures the previous PVS buffer contains the current frame's visible blocks
    output.copyCurrentToPrevious(device_, cmd);
    
    // Mark that isFirstFrame should be cleared after this frame
    if (output.isFirstFrame) {
        output.isFirstFrame = false;
    }
    
    return true;  // Occlusion culling was performed
}

void RasterOcclusionPass::Output::readbackPVSCounts(VkDevice device) {
    if (tempResources.readbackBuffer.buffer != VK_NULL_HANDLE) {
        // The readback buffer is already mapped (host-visible buffers are mapped by createBuffer)
        uint32_t* counts = (uint32_t*)tempResources.readbackBuffer.data;
        
        uint32_t oldPVSCount = pvsCurrentCount;
        pvsPreviousCount = counts[0];
        pvsCurrentCount = counts[1];
        pvsDifferenceCount = counts[2];
        
        // Check if PVS actually changed
        // Don't use tolerance - it was causing issues
        // Any change means we had z-fighting or actual visibility change
        pvsChanged = (pvsCurrentCount != oldPVSCount) || (pvsDifferenceCount > 0);
        
        if (!pvsChanged) {
            framesWithStablePVS++;
            if (framesWithStablePVS == 1) {
                lastStablePVSCount = pvsCurrentCount;
            }
        } else {
            framesWithStablePVS = 0;
        }
    }
}

void RasterOcclusionPass::createPersistentResources() {
    // Create descriptor pools for each frame in flight
    // Each pool needs enough descriptors for one frame's worth of operations
    const uint32_t descriptorMultiplier = 10; // For all passes in a frame
    
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 * descriptorMultiplier},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * descriptorMultiplier},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20 * descriptorMultiplier}
    };
    
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = descriptorMultiplier * 3; // Enough sets for all operations in a frame
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.flags = 0; // No individual free - we'll reset the entire pool each frame
    
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &persistentDescriptorPools_[i]));
    }
}

void RasterOcclusionPass::destroyPersistentResources() {
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (persistentDescriptorPools_[i] != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, persistentDescriptorPools_[i], nullptr);
            persistentDescriptorPools_[i] = VK_NULL_HANDLE;
        }
    }
}

void RasterOcclusionPass::createIndirectDrawBuffer() {
    // Create buffer for indirect draw command
    // One command with 3 uint32_t values (groupCountX, Y, Z)
    Buffer indirectBuf;
    createBuffer(indirectBuf, device_, context_.getMemoryProperties(),
                sizeof(uint32_t) * 3,
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    indirectDrawBuffer_ = indirectBuf;
}

void RasterOcclusionPass::createIndirectUpdatePipeline() {
    // Load compute shader
    Shader computeShaderData{};
    assert(loadShader(computeShaderData, device_, "/spirv/update_occlusion_indirect.comp.spv"));
    indirectUpdateComputeShader_ = computeShaderData.module;
    
    // Create descriptor set layout
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &indirectUpdateDescriptorSetLayout_));
    
    // Create pipeline layout with push constants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(uint32_t);  // Just totalBlockCount
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &indirectUpdateDescriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &indirectUpdatePipelineLayout_));
    
    // Create compute pipeline
    VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = indirectUpdateComputeShader_;
    shaderStage.pName = "main";
    
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = indirectUpdatePipelineLayout_;
    
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &indirectUpdatePipeline_));
}

void RasterOcclusionPass::updateIndirectDrawBufferGPU(VkCommandBuffer cmd, uint32_t totalBlocks) {
    // Allocate descriptor set from persistent pool
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = persistentDescriptorPools_[currentFrameIndex_];
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &indirectUpdateDescriptorSetLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet));
    
    // Update descriptor set
    VkDescriptorBufferInfo bufferInfo{indirectDrawBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    
    // Dispatch compute shader
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, indirectUpdatePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, indirectUpdatePipelineLayout_,
                           0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, indirectUpdatePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(uint32_t), &totalBlocks);
    vkCmdDispatch(cmd, 1, 1, 1);
    
    // Memory barrier to ensure compute writes are visible to indirect draw
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

RasterOcclusionPass::DebugStats RasterOcclusionPass::getDebugStats() {
    DebugStats stats;
    if (debugStatsBuffer_.buffer != VK_NULL_HANDLE && debugStatsBuffer_.data != nullptr) {
        // Read from the persistently mapped buffer
        memcpy(&stats, debugStatsBuffer_.data, sizeof(DebugStats));
    }
    return stats;
}

void RasterOcclusionPass::clearDebugStats(VkCommandBuffer cmd) {
    if (debugStatsBuffer_.buffer != VK_NULL_HANDLE) {
        // Clear the debug statistics buffer to zero
        vkCmdFillBuffer(cmd, debugStatsBuffer_.buffer, 0, sizeof(DebugStats), 0);
        
        // Memory barrier to ensure clear completes before next use
        VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        
        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &clearBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
}
