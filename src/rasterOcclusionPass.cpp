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

RasterOcclusionPass::RasterOcclusionPass(const VulkanContext& context) 
    : context_(context), device_(context.getDevice()) {
    loadShaders();
    createPipelineLayout();
    createOcclusionPipeline();
    createVisibilityCompactionPipeline();
    createBuildOutputPipeline();
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
         VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
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
    
    // Multisample state
    VkPipelineMultisampleStateCreateInfo multisampleState{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    // Depth stencil state - enable early depth test
    VkPipelineDepthStencilStateCreateInfo depthStencilState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencilState.depthTestEnable = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_FALSE; // Don't write to depth
    depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
    
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
    output.visibilityBufferSize = numBlocks * sizeof(uint32_t);
    
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = output.visibilityBufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.visibilityBuffer));
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, output.visibilityBuffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(), 
                                                 memRequirements.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.visibilityMemory));
    VK_CHECK(vkBindBufferMemory(device_, output.visibilityBuffer, output.visibilityMemory, 0));
}

void RasterOcclusionPass::createBitfieldBuffers(Output& output, uint32_t numBlocks) {
    // Create bitfield buffers - 1 bit per block, so divide by 32
    uint32_t numBitfieldEntries = (numBlocks + 31) / 32;
    output.bitfieldBufferSize = numBitfieldEntries * sizeof(uint32_t);
    
    // Current frame bitfield
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = output.bitfieldBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.currentBitfieldBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, output.currentBitfieldBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.currentBitfieldMemory));
        VK_CHECK(vkBindBufferMemory(device_, output.currentBitfieldBuffer, output.currentBitfieldMemory, 0));
    }
    
    // Previous frame bitfield
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = output.bitfieldBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.previousBitfieldBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, output.previousBitfieldBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.previousBitfieldMemory));
        VK_CHECK(vkBindBufferMemory(device_, output.previousBitfieldBuffer, output.previousBitfieldMemory, 0));
    }
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
    output.pvsBufferSize = pvsBufferSize; // Store for later use
    
    // PVS current buffer
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = pvsBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.pvsCurrentBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, output.pvsCurrentBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(), 
                                                   memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.pvsCurrentMemory));
        VK_CHECK(vkBindBufferMemory(device_, output.pvsCurrentBuffer, output.pvsCurrentMemory, 0));
    }
    
    // PVS difference buffer
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = pvsBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.pvsDifferenceBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, output.pvsDifferenceBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(), 
                                                   memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.pvsDifferenceMemory));
        VK_CHECK(vkBindBufferMemory(device_, output.pvsDifferenceBuffer, output.pvsDifferenceMemory, 0));
    }
    
    // Clear buffers
    vkCmdFillBuffer(cmd, output.visibilityBuffer, 0, output.visibilityBufferSize, 0);
    vkCmdFillBuffer(cmd, output.currentBitfieldBuffer, 0, output.bitfieldBufferSize, 0);
    vkCmdFillBuffer(cmd, output.pvsCurrentBuffer, 0, sizeof(uint32_t), 0); // Clear count
    vkCmdFillBuffer(cmd, output.pvsDifferenceBuffer, 0, sizeof(uint32_t), 0); // Clear count
    
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
    
    // Create uniform buffer
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformMemory;
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = sizeof(ViewUniforms);
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &uniformBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, uniformBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &uniformMemory));
        VK_CHECK(vkBindBufferMemory(device_, uniformBuffer, uniformMemory, 0));
        
        // Upload data
        void* mapped;
        VK_CHECK(vkMapMemory(device_, uniformMemory, 0, sizeof(ViewUniforms), 0, &mapped));
        memcpy(mapped, &viewUniforms, sizeof(ViewUniforms));
        vkUnmapMemory(device_, uniformMemory);
    }
    
    // Create descriptor pool and sets
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}  // Increased for new pipeline
    };
    
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 3; // One for occlusion, two for compaction stages
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool));
    
    // Allocate descriptor sets
    VkDescriptorSet occlusionDescriptorSet, visibilityCompactionDescriptorSet, buildOutputDescriptorSet;
    std::array<VkDescriptorSetLayout, 3> layouts = {
        occlusionDescriptorSetLayout_, 
        visibilityCompactionDescriptorSetLayout_,
        buildOutputDescriptorSetLayout_
    };
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 3;
    allocInfo.pSetLayouts = layouts.data();
    
    std::array<VkDescriptorSet, 3> sets;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, sets.data()));
    occlusionDescriptorSet = sets[0];
    visibilityCompactionDescriptorSet = sets[1];
    buildOutputDescriptorSet = sets[2];
    
    // Create sampler for min-max texture
    VkSampler minMaxSampler;
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &minMaxSampler));
    
    // Update occlusion descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    
    // Binding 0: View uniforms
    VkDescriptorBufferInfo viewUboInfo{uniformBuffer, 0, sizeof(ViewUniforms)};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet, 
                     0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewUboInfo, nullptr});
    
    // Binding 1: Min-max hierarchy texture
    VkDescriptorImageInfo minMaxInfo{minMaxSampler, minMaxOutput.minMaxImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &minMaxInfo, nullptr, nullptr});
    
    // Binding 2: Visibility buffer
    VkDescriptorBufferInfo visibilityInfo{output.visibilityBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibilityInfo, nullptr});
    
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
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Bind occlusion pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipelineLayout_,
                           0, 1, &occlusionDescriptorSet, 0, nullptr);
    
    // Push view-projection matrix
    vkCmdPushConstants(cmd, occlusionPipelineLayout_, 
                      VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                      0, sizeof(glm::mat4), &viewProjMatrix);
    
    // Dispatch task shaders - blocks are processed in 8x8x8 groups, split into two 8x8x4 halves
    // So we need 2 workgroups per 8x8x8 = 512 blocks
    uint32_t blocksPerGroup = 8 * 8 * 8; // 512
    uint32_t numGroups = (totalBlocks + blocksPerGroup - 1) / blocksPerGroup;
    uint32_t numWorkgroups = numGroups * 2; // 2 workgroups per 8x8x8 group
    
    // Debug output for first few frames
    static int occlusionFrame = 0;
    if (occlusionFrame++ < 5) {
        printf("Occlusion culling dispatch: totalBlocks=%u, numGroups=%u, numWorkgroups=%u\n", 
               totalBlocks, numGroups, numWorkgroups);
    }
    
    vkCmdDrawMeshTasksEXT(cmd, numWorkgroups, 1, 1);
    
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
    VkDescriptorBufferInfo currentBitfieldInfo{output.currentBitfieldBuffer, 0, VK_WHOLE_SIZE};
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
    VkDescriptorBufferInfo previousBitfieldInfo{output.previousBitfieldBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &previousBitfieldInfo, nullptr});
    
    // Binding 2: PVS current buffer (output)
    VkDescriptorBufferInfo pvsCurrentInfo{output.pvsCurrentBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pvsCurrentInfo, nullptr});
    
    // Binding 3: PVS difference buffer (output)
    VkDescriptorBufferInfo pvsDifferenceInfo{output.pvsDifferenceBuffer, 0, VK_WHOLE_SIZE};
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
    
    // Store temporary resources in output (to be cleaned up after command buffer submission)
    output.tempResources.minMaxSampler = minMaxSampler;
    output.tempResources.descriptorPool = descriptorPool;
    output.tempResources.uniformBuffer = uniformBuffer;
    output.tempResources.uniformMemory = uniformMemory;
    
    return output;
}

void RasterOcclusionPass::Output::destroy(VkDevice device) {
    // Clean up temporary resources first
    tempResources.destroy(device);
    
    if (visibilityBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, visibilityBuffer, nullptr);
    }
    if (visibilityMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, visibilityMemory, nullptr);
    }
    if (currentBitfieldBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, currentBitfieldBuffer, nullptr);
    }
    if (currentBitfieldMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, currentBitfieldMemory, nullptr);
    }
    if (previousBitfieldBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, previousBitfieldBuffer, nullptr);
    }
    if (previousBitfieldMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, previousBitfieldMemory, nullptr);
    }
    if (pvsCurrentBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, pvsCurrentBuffer, nullptr);
    }
    if (pvsCurrentMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, pvsCurrentMemory, nullptr);
    }
    if (pvsDifferenceBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, pvsDifferenceBuffer, nullptr);
    }
    if (pvsDifferenceMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, pvsDifferenceMemory, nullptr);
    }
    if (pvsPreviousBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, pvsPreviousBuffer, nullptr);
    }
    if (pvsPreviousMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, pvsPreviousMemory, nullptr);
    }
}

void RasterOcclusionPass::Output::swapTemporalBuffers() {
    // Swap current PVS to become previous PVS for next frame
    std::swap(pvsPreviousBuffer, pvsCurrentBuffer);
    std::swap(pvsPreviousMemory, pvsCurrentMemory);
    std::swap(pvsPreviousCount, pvsCurrentCount);
    
    // Swap bitfield buffers as well
    std::swap(previousBitfieldBuffer, currentBitfieldBuffer);
    std::swap(previousBitfieldMemory, currentBitfieldMemory);
    
    // Increment frame index
    frameIndex++;
    // Note: isFirstFrame is now set to false in the main render loop after rendering
}

void RasterOcclusionPass::Output::copyCurrentToPrevious(VkDevice device, VkCommandBuffer cmd) {
    // Copy current bitfield to previous bitfield
    VkBufferCopy bitfieldCopy{};
    bitfieldCopy.size = bitfieldBufferSize;
    vkCmdCopyBuffer(cmd, currentBitfieldBuffer, previousBitfieldBuffer, 1, &bitfieldCopy);
    
    // Copy current PVS to previous PVS
    // Copy the entire buffer to ensure all data is transferred
    // The buffer size includes space for the count + all block indices
    VkBufferCopy pvsCopy{};
    pvsCopy.size = pvsBufferSize; // Use the full buffer size
    vkCmdCopyBuffer(cmd, pvsCurrentBuffer, pvsPreviousBuffer, 1, &pvsCopy);
    
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
    output.pvsBufferSize = pvsBufferSize; // Store for later use
    
    // Create PVS current buffer
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = pvsBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.pvsCurrentBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, output.pvsCurrentBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.pvsCurrentMemory));
        VK_CHECK(vkBindBufferMemory(device_, output.pvsCurrentBuffer, output.pvsCurrentMemory, 0));
    }
    
    // Create PVS difference buffer
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = pvsBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.pvsDifferenceBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, output.pvsDifferenceBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.pvsDifferenceMemory));
        VK_CHECK(vkBindBufferMemory(device_, output.pvsDifferenceBuffer, output.pvsDifferenceMemory, 0));
    }
    
    // Create PVS previous buffer (PVS_prev)
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = pvsBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &output.pvsPreviousBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, output.pvsPreviousBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &output.pvsPreviousMemory));
        VK_CHECK(vkBindBufferMemory(device_, output.pvsPreviousBuffer, output.pvsPreviousMemory, 0));
    }
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

void RasterOcclusionPass::performTemporalOcclusionCulling(
    VkCommandBuffer cmd,
    Output& output,
    const MinMaxOutput& minMaxOutput,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    VkImageView depthImageView,
    VkExtent2D renderExtent) {
    
    // Calculate total number of blocks
    uint32_t totalBlocks = pushConstants.blockGridDim.x * 
                          pushConstants.blockGridDim.y * 
                          pushConstants.blockGridDim.z;
    
    // Debug output
    static int occlusionCallCount = 0;
    if (occlusionCallCount++ < 5) {
        printf("performTemporalOcclusionCulling called: totalBlocks=%u, blockGridDim=(%u,%u,%u)\n",
               totalBlocks, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);
    }
    
    // Initialize output on first frame
    if (output.isFirstFrame || output.visibilityBuffer == VK_NULL_HANDLE) {
        initializeOutput(output, totalBlocks);
        
        // Clear the previous frame bitfield buffer on first frame
        vkCmdFillBuffer(cmd, output.previousBitfieldBuffer, 0, output.bitfieldBufferSize, 0);
        
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
    vkCmdFillBuffer(cmd, output.visibilityBuffer, 0, output.visibilityBufferSize, 0);
    vkCmdFillBuffer(cmd, output.currentBitfieldBuffer, 0, output.bitfieldBufferSize, 0);
    vkCmdFillBuffer(cmd, output.pvsCurrentBuffer, 0, sizeof(uint32_t), 0); // Clear count
    vkCmdFillBuffer(cmd, output.pvsDifferenceBuffer, 0, sizeof(uint32_t), 0); // Clear count
    
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
    
    // Create uniform buffer
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformMemory;
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = sizeof(ViewUniforms);
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &uniformBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, uniformBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &uniformMemory));
        VK_CHECK(vkBindBufferMemory(device_, uniformBuffer, uniformMemory, 0));
        
        // Upload data
        void* mapped;
        VK_CHECK(vkMapMemory(device_, uniformMemory, 0, sizeof(ViewUniforms), 0, &mapped));
        memcpy(mapped, &viewUniforms, sizeof(ViewUniforms));
        vkUnmapMemory(device_, uniformMemory);
    }
    
    // Create descriptor pool and sets
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}
    };
    
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 3;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool));
    
    // Allocate descriptor sets
    VkDescriptorSet occlusionDescriptorSet, visibilityCompactionDescriptorSet, buildOutputDescriptorSet;
    std::array<VkDescriptorSetLayout, 3> layouts = {
        occlusionDescriptorSetLayout_, 
        visibilityCompactionDescriptorSetLayout_,
        buildOutputDescriptorSetLayout_
    };
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 3;
    allocInfo.pSetLayouts = layouts.data();
    
    std::array<VkDescriptorSet, 3> sets;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, sets.data()));
    occlusionDescriptorSet = sets[0];
    visibilityCompactionDescriptorSet = sets[1];
    buildOutputDescriptorSet = sets[2];
    
    // Create sampler for min-max texture
    VkSampler minMaxSampler;
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &minMaxSampler));
    
    // Update occlusion descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    
    // Binding 0: View uniforms
    VkDescriptorBufferInfo viewUboInfo{uniformBuffer, 0, sizeof(ViewUniforms)};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet, 
                     0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewUboInfo, nullptr});
    
    // Binding 1: Min-max hierarchy texture
    VkDescriptorImageInfo minMaxInfo{minMaxSampler, minMaxOutput.minMaxImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &minMaxInfo, nullptr, nullptr});
    
    // Binding 2: Visibility buffer
    VkDescriptorBufferInfo visibilityInfo{output.visibilityBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occlusionDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibilityInfo, nullptr});
    
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
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Bind occlusion pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, occlusionPipelineLayout_,
                           0, 1, &occlusionDescriptorSet, 0, nullptr);
    
    // Push view-projection matrix
    vkCmdPushConstants(cmd, occlusionPipelineLayout_, 
                      VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                      0, sizeof(glm::mat4), &viewProjMatrix);
    
    // Dispatch task shaders
    uint32_t blocksPerGroup = 8 * 8 * 8; // 512
    uint32_t numGroups = (totalBlocks + blocksPerGroup - 1) / blocksPerGroup;
    uint32_t numWorkgroups = numGroups * 2; // 2 workgroups per 8x8x8 group
    vkCmdDrawMeshTasksEXT(cmd, numWorkgroups, 1, 1);
    
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
    VkDescriptorBufferInfo currentBitfieldInfo{output.currentBitfieldBuffer, 0, VK_WHOLE_SIZE};
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
    VkDescriptorBufferInfo previousBitfieldInfo{output.previousBitfieldBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &previousBitfieldInfo, nullptr});
    
    // Binding 2: PVS current buffer (output)
    VkDescriptorBufferInfo pvsCurrentInfo{output.pvsCurrentBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, buildOutputDescriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pvsCurrentInfo, nullptr});
    
    // Binding 3: PVS difference buffer (output)
    VkDescriptorBufferInfo pvsDifferenceInfo{output.pvsDifferenceBuffer, 0, VK_WHOLE_SIZE};
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
    
    // Debug: Print some stats and force some visibility for testing
    static int debugFrameCounter = 0;
    if (debugFrameCounter++ % 60 == 0) {
        printf("Occlusion culling - Total blocks: %u, First frame: %s\n", 
               totalBlocks, output.isFirstFrame ? "true" : "false");
        printf("  Volume: %ux%ux%u, Block: %ux%ux%u, Grid: %ux%ux%u\n",
               pushConstants.volumeDim.x, pushConstants.volumeDim.y, pushConstants.volumeDim.z,
               pushConstants.blockDim.x, pushConstants.blockDim.y, pushConstants.blockDim.z,
               pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);
        printf("  Isovalue: %.3f\n", pushConstants.isovalue);
        printf("  Test blocks added: first frame=%u, subsequent=%u\n",
               output.isFirstFrame ? std::min(totalBlocks, 100u) : 120,
               output.isFirstFrame ? std::min(totalBlocks, 100u) : 20);
    }
    
    // TEMPORARY: Force some blocks to be visible for testing
    // In production, these counts should be read back from GPU
    // COMMENTED OUT to allow GPU to compute actual visibility
    /*
    if (output.isFirstFrame) {
        output.pvsCurrentCount = 512;  // All blocks for testing
        output.pvsDifferenceCount = 512;
        
        // Also populate the PVS buffers with some test block IDs
        std::vector<uint32_t> testData;
        
        // For a 64x64x64 volume with 8x8x8 blocks, we have an 8x8x8 grid
        // The sphere is centered at (32,32,32) with radius ~25.6
        // So center block is at grid position (4,4,4)
        
        // Add all blocks that might contain the sphere
        // Sphere radius is 25.6, centered at (32,32,32)
        // So it spans roughly from voxel 6 to 58
        // In block coordinates: 0-1 to 7 (all blocks!)
        for (uint32_t z = 0; z < 8; z++) {
            for (uint32_t y = 0; y < 8; y++) {
                for (uint32_t x = 0; x < 8; x++) {
                    uint32_t blockID = z * pushConstants.blockGridDim.x * pushConstants.blockGridDim.y +
                                      y * pushConstants.blockGridDim.x + x;
                    testData.push_back(blockID);
                    
                    // Debug: print block details for center block
                    if (x == 4 && y == 4 && z == 4) {
                        std::cout << "DEBUG: Center block (4,4,4) -> ID " << blockID << std::endl;
                        std::cout << "  Block grid: " << pushConstants.blockGridDim.x << "x" 
                                  << pushConstants.blockGridDim.y << "x" << pushConstants.blockGridDim.z << std::endl;
                        std::cout << "  Block covers voxels: " << (x * 8) << "-" << ((x+1) * 8 - 1) << " x "
                                  << (y * 8) << "-" << ((y+1) * 8 - 1) << " x "
                                  << (z * 8) << "-" << ((z+1) * 8 - 1) << std::endl;
                    }
                }
            }
        }
        
        std::cout << "DEBUG: Forcing all " << testData.size() << " blocks to test sphere" << std::endl;
        
        // Insert count at the beginning
        uint32_t actualCount = testData.size();
        testData.insert(testData.begin(), actualCount);
        output.pvsCurrentCount = actualCount;
        output.pvsDifferenceCount = actualCount;
        
        // Upload test data to PVS buffers using staging buffer
        VkDeviceSize dataSize = testData.size() * sizeof(uint32_t);
        
        // Create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        {
            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.size = dataSize;
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            
            VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &stagingBuffer));
            
            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(device_, stagingBuffer, &memRequirements);
            
            VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                         memRequirements.memoryTypeBits,
                                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            
            VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &stagingMemory));
            VK_CHECK(vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));
            
            // Copy data to staging buffer
            void* mapped;
            VK_CHECK(vkMapMemory(device_, stagingMemory, 0, dataSize, 0, &mapped));
            memcpy(mapped, testData.data(), dataSize);
            vkUnmapMemory(device_, stagingMemory);
        }
        
        // Copy from staging to PVS buffers (include count)
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0; // Start from beginning (includes count)
        copyRegion.size = dataSize;
        
        vkCmdCopyBuffer(cmd, stagingBuffer, output.pvsPreviousBuffer, 1, &copyRegion);
        vkCmdCopyBuffer(cmd, stagingBuffer, output.pvsDifferenceBuffer, 1, &copyRegion);
        
        // Barrier to ensure copies complete before shader reads
        VkMemoryBarrier2 copyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        copyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        copyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        copyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
        copyBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        
        VkDependencyInfo copyDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        copyDepInfo.memoryBarrierCount = 1;
        copyDepInfo.pMemoryBarriers = &copyBarrier;
        vkCmdPipelineBarrier2(cmd, &copyDepInfo);
        
        // Add to temp resources for cleanup
        output.tempResources.stagingBuffer = stagingBuffer;
        output.tempResources.stagingMemory = stagingMemory;
    } else {
        output.pvsCurrentCount = 512;  // All blocks
        output.pvsDifferenceCount = 0;  // No new blocks for now
        
        // First update the previous buffer with all previous frame blocks
        std::vector<uint32_t> prevData;
        prevData.push_back(512); // Count for previous buffer (all blocks)
        
        // Add all blocks
        for (uint32_t z = 0; z < 8; z++) {
            for (uint32_t y = 0; y < 8; y++) {
                for (uint32_t x = 0; x < 8; x++) {
                    uint32_t blockID = z * pushConstants.blockGridDim.x * pushConstants.blockGridDim.y +
                                      y * pushConstants.blockGridDim.x + x;
                    prevData.push_back(blockID);
                }
            }
        }
        
        // Upload previous frame data
        {
            VkDeviceSize prevDataSize = prevData.size() * sizeof(uint32_t);
            
            // Create staging buffer for previous data
            VkBuffer prevStagingBuffer;
            VkDeviceMemory prevStagingMemory;
            {
                VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                bufferInfo.size = prevDataSize;
                bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                
                VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &prevStagingBuffer));
                
                VkMemoryRequirements memRequirements;
                vkGetBufferMemoryRequirements(device_, prevStagingBuffer, &memRequirements);
                
                VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocInfo.allocationSize = memRequirements.size;
                allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                             memRequirements.memoryTypeBits,
                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                
                VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &prevStagingMemory));
                VK_CHECK(vkBindBufferMemory(device_, prevStagingBuffer, prevStagingMemory, 0));
                
                // Copy data to staging buffer
                void* mapped;
                VK_CHECK(vkMapMemory(device_, prevStagingMemory, 0, prevDataSize, 0, &mapped));
                memcpy(mapped, prevData.data(), prevDataSize);
                vkUnmapMemory(device_, prevStagingMemory);
            }
            
            // Copy to previous buffer
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = prevDataSize;
            
            vkCmdCopyBuffer(cmd, prevStagingBuffer, output.pvsPreviousBuffer, 1, &copyRegion);
            
            // Store for cleanup after command buffer submission
            output.tempResources.stagingBuffer2 = prevStagingBuffer;
            output.tempResources.stagingMemory2 = prevStagingMemory;
        }
        
        // For now, no new blocks in difference buffer
        std::vector<uint32_t> newData;
        newData.push_back(0); // No new blocks
        
        // Update only difference buffer with new blocks using staging
        VkDeviceSize dataSize = newData.size() * sizeof(uint32_t);
        
        // Create staging buffer for new blocks
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        {
            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.size = dataSize;
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            
            VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &stagingBuffer));
            
            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(device_, stagingBuffer, &memRequirements);
            
            VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                         memRequirements.memoryTypeBits,
                                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            
            VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &stagingMemory));
            VK_CHECK(vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));
            
            // Copy data to staging buffer  
            void* mapped;
            VK_CHECK(vkMapMemory(device_, stagingMemory, 0, dataSize, 0, &mapped));
            memcpy(mapped, newData.data(), dataSize);
            vkUnmapMemory(device_, stagingMemory);
        }
        
        // Copy from staging to difference buffer
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0; // Start from beginning (includes count)
        copyRegion.size = dataSize;
        
        vkCmdCopyBuffer(cmd, stagingBuffer, output.pvsDifferenceBuffer, 1, &copyRegion);
        
        // Barrier to ensure copy completes before shader reads
        VkMemoryBarrier2 copyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        copyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        copyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        copyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
        copyBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        
        VkDependencyInfo copyDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        copyDepInfo.memoryBarrierCount = 1;
        copyDepInfo.pMemoryBarriers = &copyBarrier;
        vkCmdPipelineBarrier2(cmd, &copyDepInfo);
        
        // Add to temp resources for cleanup
        output.tempResources.stagingBuffer = stagingBuffer;
        output.tempResources.stagingMemory = stagingMemory;
        
        // Also need to update current buffer with the same blocks as previous
        // Re-use the copyRegion from previous buffer copy
        VkBufferCopy currentCopyRegion{};
        currentCopyRegion.srcOffset = 0;
        currentCopyRegion.dstOffset = 0;
        currentCopyRegion.size = prevData.size() * sizeof(uint32_t);
        vkCmdCopyBuffer(cmd, output.tempResources.stagingBuffer2, output.pvsCurrentBuffer, 1, &currentCopyRegion);
        
        // Update the actual count
        output.pvsCurrentCount = 512;
    }
    */
    
    // Instead of hardcoded values, we should read back the counts from GPU
    // For now, we'll rely on the GPU compute shaders to populate the PVS buffers correctly
    
    // Create readback buffers for PVS counts
    VkBuffer readbackBuffer;
    VkDeviceMemory readbackMemory;
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = 3 * sizeof(uint32_t); // For prev, current, and difference counts
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &readbackBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, readbackBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &readbackMemory));
        VK_CHECK(vkBindBufferMemory(device_, readbackBuffer, readbackMemory, 0));
    }
    
    // Copy counts from GPU buffers to readback buffer
    VkBufferCopy countCopy{};
    countCopy.size = sizeof(uint32_t);
    
    // Copy previous count
    countCopy.srcOffset = 0;
    countCopy.dstOffset = 0;
    vkCmdCopyBuffer(cmd, output.pvsPreviousBuffer, readbackBuffer, 1, &countCopy);
    
    // Copy current count
    countCopy.srcOffset = 0;
    countCopy.dstOffset = sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, output.pvsCurrentBuffer, readbackBuffer, 1, &countCopy);
    
    // Copy difference count
    countCopy.srcOffset = 0;
    countCopy.dstOffset = 2 * sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, output.pvsDifferenceBuffer, readbackBuffer, 1, &countCopy);
    
    // Store readback resources for later retrieval
    output.tempResources.readbackBuffer = readbackBuffer;
    output.tempResources.readbackMemory = readbackMemory;
    
    // Copy current frame data to previous frame for next frame's temporal coherence
    // This ensures the previous PVS buffer contains the current frame's visible blocks
    output.copyCurrentToPrevious(device_, cmd);
    
    // Mark that isFirstFrame should be cleared after this frame
    if (output.isFirstFrame) {
        output.isFirstFrame = false;
    }
}

void RasterOcclusionPass::Output::readbackPVSCounts(VkDevice device) {
    if (tempResources.readbackBuffer && tempResources.readbackMemory) {
        // Map the readback buffer and read the counts
        uint32_t* counts;
        VK_CHECK(vkMapMemory(device, tempResources.readbackMemory, 0, 3 * sizeof(uint32_t), 0, (void**)&counts));
        
        pvsPreviousCount = counts[0];
        pvsCurrentCount = counts[1];
        pvsDifferenceCount = counts[2];
        
        vkUnmapMemory(device, tempResources.readbackMemory);
        
        // Debug print for first few frames
        static int readbackFrame = 0;
        if (readbackFrame++ < 10) {
            printf("GPU Readback [Frame %d]: prev=%u, curr=%u, diff=%u\n", 
                   readbackFrame-1, pvsPreviousCount, pvsCurrentCount, pvsDifferenceCount);
        }
    }
}
