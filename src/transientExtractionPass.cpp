#include "common.h"
#include "transientExtractionPass.h"
#include "vulkan_context.h"
#include "vulkan_utils.h"
#include "shaders.h"
#include "minMaxOutput.h"
#include "rasterOcclusionPass.h"
#include "buffer.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>

TransientExtractionPass::TransientExtractionPass(const VulkanContext& context, VkFormat swapchainFormat)
    : context_(context), device_(context.getDevice()), swapchainFormat_(swapchainFormat) {
    loadShaders();
    createPipelineLayouts();
    createPipelines();
    createShadingParametersBuffer();
    createMarchingCubesTables(true);  // Use unique tables by default
}

TransientExtractionPass::~TransientExtractionPass() {
    if (pass1Pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pass1Pipeline_, nullptr);
    }
    if (pass1PipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pass1PipelineLayout_, nullptr);
    }
    if (pass1DescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, pass1DescriptorSetLayout_, nullptr);
    }
    
    if (pass2Pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pass2Pipeline_, nullptr);
    }
    if (pass2PipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pass2PipelineLayout_, nullptr);
    }
    if (pass2DescriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, pass2DescriptorSetLayout_, nullptr);
    }
    
    if (pass1TaskShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, pass1TaskShader_, nullptr);
    }
    if (pass2TaskShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, pass2TaskShader_, nullptr);
    }
    if (meshShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, meshShader_, nullptr);
    }
    if (fragmentShader_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, fragmentShader_, nullptr);
    }
    
    if (shadingParamsBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, shadingParamsBuffer_, nullptr);
    }
    if (shadingParamsMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, shadingParamsMemory_, nullptr);
    }
    
    // Clean up marching cubes tables
    destroyMarchingCubesTables();
}

void TransientExtractionPass::loadShaders() {
    // Load pass 1 task shader
    Shader pass1TaskShaderData{};
    assert(loadShader(pass1TaskShaderData, device_, "/spirv/transient_extraction_prev.task.spv"));
    pass1TaskShader_ = pass1TaskShaderData.module;
    
    // Load pass 2 task shader
    Shader pass2TaskShaderData{};
    assert(loadShader(pass2TaskShaderData, device_, "/spirv/transient_extraction_curr_minus_prev.task.spv"));
    pass2TaskShader_ = pass2TaskShaderData.module;
    
    // Load shared mesh shader
    Shader meshShaderData{};
    assert(loadShader(meshShaderData, device_, "/spirv/transient_extraction_temporal.mesh.spv"));
    meshShader_ = meshShaderData.module;
    
    // Load shared fragment shader
    Shader fragmentShaderData{};
    assert(loadShader(fragmentShaderData, device_, "/spirv/transient_extraction_shading.frag.spv"));
    fragmentShader_ = fragmentShaderData.module;
}

void TransientExtractionPass::createPipelineLayouts() {
    // Common bindings for both passes
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        // Binding 0: View parameters UBO
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, 
         VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        // Binding 1: Volume texture
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, 
         VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        // Binding 2: Marching cubes numVertices TBO (shared by task and mesh shaders)
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1,
         VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        // Binding 3: Marching cubes triTable TBO (mesh shader only)
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1,
         VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        // Binding 5: Min-max hierarchy texture
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, 
         VK_SHADER_STAGE_TASK_BIT_EXT, nullptr},
        // Binding 10: Shading parameters UBO
        {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, 
         VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    };
    
    // Pass 1 specific binding
    std::vector<VkDescriptorSetLayoutBinding> pass1Bindings = bindings;
    pass1Bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, 
                            VK_SHADER_STAGE_TASK_BIT_EXT, nullptr}); // PVS_prev
    
    // Pass 2 specific binding
    std::vector<VkDescriptorSetLayoutBinding> pass2Bindings = bindings;
    pass2Bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, 
                            VK_SHADER_STAGE_TASK_BIT_EXT, nullptr}); // PVS_curr-prev
    
    // Create pass 1 descriptor set layout
    VkDescriptorSetLayoutCreateInfo pass1LayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    pass1LayoutInfo.bindingCount = static_cast<uint32_t>(pass1Bindings.size());
    pass1LayoutInfo.pBindings = pass1Bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &pass1LayoutInfo, nullptr, &pass1DescriptorSetLayout_));
    
    // Create pass 2 descriptor set layout
    VkDescriptorSetLayoutCreateInfo pass2LayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    pass2LayoutInfo.bindingCount = static_cast<uint32_t>(pass2Bindings.size());
    pass2LayoutInfo.pBindings = pass2Bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &pass2LayoutInfo, nullptr, &pass2DescriptorSetLayout_));
    
    // Push constants for render pass index and table type
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 2 * sizeof(uint32_t); // renderPass index + useUniqueTables flag
    
    // Create pass 1 pipeline layout
    VkPipelineLayoutCreateInfo pass1PipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pass1PipelineLayoutInfo.setLayoutCount = 1;
    pass1PipelineLayoutInfo.pSetLayouts = &pass1DescriptorSetLayout_;
    pass1PipelineLayoutInfo.pushConstantRangeCount = 1;
    pass1PipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(device_, &pass1PipelineLayoutInfo, nullptr, &pass1PipelineLayout_));
    
    // Create pass 2 pipeline layout
    VkPipelineLayoutCreateInfo pass2PipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pass2PipelineLayoutInfo.setLayoutCount = 1;
    pass2PipelineLayoutInfo.pSetLayouts = &pass2DescriptorSetLayout_;
    pass2PipelineLayoutInfo.pushConstantRangeCount = 1;
    pass2PipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(device_, &pass2PipelineLayoutInfo, nullptr, &pass2PipelineLayout_));
}

void TransientExtractionPass::createPipelines() {
    // Common pipeline state for both passes
    VkPipelineVertexInputStateCreateInfo vertexInputState{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    VkPipelineRasterizationStateCreateInfo rasterizationState{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationState.lineWidth = 1.0f;
    
    VkPipelineMultisampleStateCreateInfo multisampleState{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    // Pass 1: Depth test with LESS (renders behind existing geometry)
    VkPipelineDepthStencilStateCreateInfo pass1DepthStencilState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    pass1DepthStencilState.depthTestEnable = VK_TRUE;
    pass1DepthStencilState.depthWriteEnable = VK_TRUE;
    pass1DepthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
    
    // Pass 2: Depth test with LESS (only newly visible parts)
    VkPipelineDepthStencilStateCreateInfo pass2DepthStencilState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    pass2DepthStencilState.depthTestEnable = VK_TRUE;
    pass2DepthStencilState.depthWriteEnable = VK_TRUE;
    pass2DepthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
    
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    
    VkPipelineColorBlendStateCreateInfo colorBlendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &colorBlendAttachment;
    
    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();
    
    // Pass 1 shader stages
    std::vector<VkPipelineShaderStageCreateInfo> pass1ShaderStages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_TASK_BIT_EXT, pass1TaskShader_, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_MESH_BIT_EXT, meshShader_, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader_, "main", nullptr}
    };
    
    // Pass 2 shader stages
    std::vector<VkPipelineShaderStageCreateInfo> pass2ShaderStages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_TASK_BIT_EXT, pass2TaskShader_, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_MESH_BIT_EXT, meshShader_, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader_, "main", nullptr}
    };
    
    // Rendering info for dynamic rendering
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat_;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    
    // Create pass 1 pipeline
    VkGraphicsPipelineCreateInfo pass1PipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pass1PipelineInfo.pNext = &renderingInfo;
    pass1PipelineInfo.stageCount = static_cast<uint32_t>(pass1ShaderStages.size());
    pass1PipelineInfo.pStages = pass1ShaderStages.data();
    pass1PipelineInfo.pVertexInputState = &vertexInputState;
    pass1PipelineInfo.pInputAssemblyState = &inputAssemblyState;
    pass1PipelineInfo.pViewportState = &viewportState;
    pass1PipelineInfo.pRasterizationState = &rasterizationState;
    pass1PipelineInfo.pMultisampleState = &multisampleState;
    pass1PipelineInfo.pDepthStencilState = &pass1DepthStencilState;
    pass1PipelineInfo.pColorBlendState = &colorBlendState;
    pass1PipelineInfo.pDynamicState = &dynamicState;
    pass1PipelineInfo.layout = pass1PipelineLayout_;
    
    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pass1PipelineInfo, nullptr, &pass1Pipeline_));
    
    // Create pass 2 pipeline
    VkGraphicsPipelineCreateInfo pass2PipelineInfo = pass1PipelineInfo;
    pass2PipelineInfo.pStages = pass2ShaderStages.data();
    pass2PipelineInfo.pDepthStencilState = &pass2DepthStencilState;
    pass2PipelineInfo.layout = pass2PipelineLayout_;
    
    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pass2PipelineInfo, nullptr, &pass2Pipeline_));
}

void TransientExtractionPass::createShadingParametersBuffer() {
    VkDeviceSize bufferSize = sizeof(ShadingParameters);
    
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &shadingParamsBuffer_));
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, shadingParamsBuffer_, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                 memRequirements.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &shadingParamsMemory_));
    VK_CHECK(vkBindBufferMemory(device_, shadingParamsBuffer_, shadingParamsMemory_, 0));
}

VkDescriptorSet TransientExtractionPass::createPassDescriptorSet(
    VkDescriptorPool pool,
    VkDescriptorSetLayout layout,
    VkBuffer viewUniformBuffer,
    VkImageView minMaxImageView,
    VkSampler minMaxSampler,
    VkImageView volumeImageView,
    VkSampler volumeSampler,
    VkBuffer pvsBuffer,
    uint32_t bindingIndex) {
    
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet));
    
    std::vector<VkWriteDescriptorSet> writes;
    
    // Binding 0: View parameters
    VkDescriptorBufferInfo viewBufferInfo{viewUniformBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                     0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewBufferInfo, nullptr});
    
    // Binding 1: Volume texture
    VkDescriptorImageInfo volumeInfo{volumeSampler, volumeImageView, VK_IMAGE_LAYOUT_GENERAL};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &volumeInfo, nullptr, nullptr});
    
    // Binding 2: Marching cubes numVertices TBO (shared by task and mesh shaders)
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                     2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, nullptr, nullptr, &mcTables_.numVerticesView});
    
    // Binding 3: Marching cubes triTable TBO (mesh shader only)
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                     3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, nullptr, nullptr, &mcTables_.triTableView});
    
    // Binding 5: Min-max texture
    VkDescriptorImageInfo minMaxInfo{minMaxSampler, minMaxImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                     5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &minMaxInfo, nullptr, nullptr});
    
    // Binding 10: Shading parameters
    VkDescriptorBufferInfo shadingInfo{shadingParamsBuffer_, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                     10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &shadingInfo, nullptr});
    
    // Binding 14/15: PVS buffer
    VkDescriptorBufferInfo pvsInfo{pvsBuffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                     bindingIndex, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pvsInfo, nullptr});
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    return descriptorSet;
}

void TransientExtractionPass::renderTransientPasses(
    VkCommandBuffer cmd,
    const RasterOcclusionPass::Output& occlusionOutput,
    const MinMaxOutput& minMaxOutput,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    const glm::vec3& cameraPos,
    VkImageView colorImageView,
    VkImageView depthImageView,
    VkExtent2D renderExtent,
    const ShadingParameters& shadingParams) {
    
    // Debug: track calls
    static int callCount = 0;
    // printf("renderTransientPasses called: %d\n", callCount++);
    
    // On first frame, we still need to render using the forced PVS data
    // Don't skip!
    
    // Transition volume image from GENERAL to SHADER_READ_ONLY_OPTIMAL
    VkImageMemoryBarrier2 volumeBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    volumeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    volumeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    volumeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    volumeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    volumeBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrier.image = minMaxOutput.volumeImage.image;
    volumeBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    volumeBarrier.subresourceRange.baseMipLevel = 0;
    volumeBarrier.subresourceRange.levelCount = 1;
    volumeBarrier.subresourceRange.baseArrayLayer = 0;
    volumeBarrier.subresourceRange.layerCount = 1;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &volumeBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    
    // Update shading parameters
    {
        void* mapped;
        VK_CHECK(vkMapMemory(device_, shadingParamsMemory_, 0, sizeof(ShadingParameters), 0, &mapped));
        memcpy(mapped, &shadingParams, sizeof(ShadingParameters));
        vkUnmapMemory(device_, shadingParamsMemory_);
    }
    
    // Create view parameters uniform buffer
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
    
    VkBuffer viewUniformBuffer;
    VkDeviceMemory viewUniformMemory;
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = sizeof(ViewUniforms);
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &viewUniformBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, viewUniformBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &viewUniformMemory));
        VK_CHECK(vkBindBufferMemory(device_, viewUniformBuffer, viewUniformMemory, 0));
        
        void* mapped;
        VK_CHECK(vkMapMemory(device_, viewUniformMemory, 0, sizeof(ViewUniforms), 0, &mapped));
        memcpy(mapped, &viewUniforms, sizeof(ViewUniforms));
        vkUnmapMemory(device_, viewUniformMemory);
    }
    
    // Create descriptor pool
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 4}  // TBOs for marching cubes tables (2 per set, 2 sets)
    };
    
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool));
    
    // Create samplers
    VkSampler volumeSampler, minMaxSampler;
    
    // Volume sampler - can use linear filtering for R8_UINT volume
    VkSamplerCreateInfo volumeSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    volumeSamplerInfo.magFilter = volumeSamplerInfo.minFilter = VK_FILTER_NEAREST;
    volumeSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    volumeSamplerInfo.addressModeU = volumeSamplerInfo.addressModeV = volumeSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &volumeSamplerInfo, nullptr, &volumeSampler));
    
    // Min-max sampler - must use nearest filtering for R32G32_UINT format
    VkSamplerCreateInfo minMaxSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    minMaxSamplerInfo.magFilter = minMaxSamplerInfo.minFilter = VK_FILTER_NEAREST;
    minMaxSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    minMaxSamplerInfo.addressModeU = minMaxSamplerInfo.addressModeV = minMaxSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &minMaxSamplerInfo, nullptr, &minMaxSampler));
    
    // Get volume image view from minMaxOutput
    VkImageView volumeImageView = minMaxOutput.volumeImage.imageView;
    
    // Create descriptor sets
    VkDescriptorSet pass1DescriptorSet = createPassDescriptorSet(
        descriptorPool, pass1DescriptorSetLayout_,
        viewUniformBuffer, minMaxOutput.minMaxImage.imageView, minMaxSampler,
        volumeImageView, volumeSampler,
        occlusionOutput.pvsPreviousBuffer, 14
    );
    
    VkDescriptorSet pass2DescriptorSet = createPassDescriptorSet(
        descriptorPool, pass2DescriptorSetLayout_,
        viewUniformBuffer, minMaxOutput.minMaxImage.imageView, minMaxSampler,
        volumeImageView, volumeSampler,
        occlusionOutput.pvsDifferenceBuffer, 15
    );
    
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
    
    // Begin dynamic rendering
    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = colorImageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;    // Load existing content
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;    // Load depth from occlusion pass
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = scissor;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Debug: Print PVS counts
    static int frameCounter = 0;
    if (frameCounter++ % 60 == 0) {  // Print every 60 frames
        printf("PVS counts - Previous: %u, Difference: %u, Current: %u\n", 
               occlusionOutput.pvsPreviousCount, 
               occlusionOutput.pvsDifferenceCount,
               occlusionOutput.pvsCurrentCount);
        printf("  Dispatching workgroups - Pass1: %u, Pass2: %u\n",
               occlusionOutput.pvsPreviousCount * 2,
               occlusionOutput.pvsDifferenceCount * 2);
    }
    
    // Store counts locally to ensure they don't change during rendering
    uint32_t localPrevCount = occlusionOutput.pvsPreviousCount;
    uint32_t localDiffCount = occlusionOutput.pvsDifferenceCount;
    
    // Pass 1: Render previous frame's visible geometry (PVS_prev)
    if (localPrevCount > 0) {
        // printf("  Rendering Pass 1: %u blocks, %u workgroups\n", occlusionOutput.pvsPreviousCount, occlusionOutput.pvsPreviousCount * 2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass1Pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass1PipelineLayout_,
                               0, 1, &pass1DescriptorSet, 0, nullptr);
        
        struct { uint32_t renderPass; uint32_t useUniqueTables; } pushConstants;
        pushConstants.renderPass = 0; // Pass 1
        pushConstants.useUniqueTables = mcTables_.isUnique ? 1 : 0;
        vkCmdPushConstants(cmd, pass1PipelineLayout_, VK_SHADER_STAGE_MESH_BIT_EXT,
                          0, sizeof(pushConstants), &pushConstants);
        
        // Dispatch task shaders - 2 workgroups per block (for split processing)
        uint32_t pass1Workgroups = localPrevCount * 2;
        vkCmdDrawMeshTasksEXT(cmd, pass1Workgroups, 1, 1);
    } else {
        // printf("  Skipping Pass 1: no previous blocks\n");
    }
    
    // Pass 2: Render newly visible geometry (PVS_curr-prev)
    // Add frame ID to debug output
    static int globalFrameCount = 0;
    int currentFrame = globalFrameCount++;
    if (currentFrame < 10) {
        printf("  [Frame %d] Pass 1: prevCount = %u, Pass 2: diffCount = %u, currCount = %u\n", 
               currentFrame, localPrevCount, localDiffCount, occlusionOutput.pvsCurrentCount);
    }
    // Pass 2 is disabled when bypassing PVS
    if (!bypassPVS_ && localDiffCount > 0) {
        // printf("  Rendering Pass 2: %u blocks, %u workgroups\n", occlusionOutput.pvsDifferenceCount, occlusionOutput.pvsDifferenceCount * 2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass2Pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass2PipelineLayout_,
                               0, 1, &pass2DescriptorSet, 0, nullptr);
        
        struct { uint32_t renderPass; uint32_t useUniqueTables; } pushConstants;
        pushConstants.renderPass = 1; // Pass 2
        pushConstants.useUniqueTables = mcTables_.isUnique ? 1 : 0;
        vkCmdPushConstants(cmd, pass2PipelineLayout_, VK_SHADER_STAGE_MESH_BIT_EXT,
                          0, sizeof(pushConstants), &pushConstants);
        
        // Dispatch task shaders - 2 workgroups per block (for split processing)
        uint32_t pass2Workgroups = localDiffCount * 2;
        vkCmdDrawMeshTasksEXT(cmd, pass2Workgroups, 1, 1);
    } else {
        // printf("  Skipping Pass 2: no difference blocks\n");
    }
    
    // End rendering
    vkCmdEndRendering(cmd);
    
    // Transition volume image back to GENERAL for compute shaders
    VkImageMemoryBarrier2 volumeBarrierBack{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    volumeBarrierBack.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    volumeBarrierBack.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    volumeBarrierBack.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    volumeBarrierBack.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    volumeBarrierBack.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrierBack.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrierBack.image = minMaxOutput.volumeImage.image;
    volumeBarrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    volumeBarrierBack.subresourceRange.baseMipLevel = 0;
    volumeBarrierBack.subresourceRange.levelCount = 1;
    volumeBarrierBack.subresourceRange.baseArrayLayer = 0;
    volumeBarrierBack.subresourceRange.layerCount = 1;
    
    VkDependencyInfo depInfoBack{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfoBack.imageMemoryBarrierCount = 1;
    depInfoBack.pImageMemoryBarriers = &volumeBarrierBack;
    vkCmdPipelineBarrier2(cmd, &depInfoBack);
    
    // Store temporary resources (to be cleaned up after command buffer submission)
    // Using pass1 slots for the combined function
    tempResources_.volumeSampler_pass1 = volumeSampler;
    tempResources_.minMaxSampler_pass1 = minMaxSampler;
    tempResources_.descriptorPool_pass1 = descriptorPool;
    tempResources_.viewUniformBuffer_pass1 = viewUniformBuffer;
    tempResources_.viewUniformMemory_pass1 = viewUniformMemory;
}

void TransientExtractionPass::renderPass1_PreviousVisible(
    VkCommandBuffer cmd,
    const RasterOcclusionPass::Output& occlusionOutput,
    const MinMaxOutput& minMaxOutput,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    const glm::vec3& cameraPos,
    VkImageView colorImageView,
    VkImageView depthImageView,
    VkExtent2D renderExtent,
    const ShadingParameters& shadingParams) {
    
    // Debug tracking
    static int pass1CallCount = 0;
    printf("renderPass1_PreviousVisible called: %d, pvsPreviousCount=%u\n", 
           pass1CallCount++, occlusionOutput.pvsPreviousCount);
    
    // Skip if no previous blocks to render
    if (occlusionOutput.pvsPreviousCount == 0) {
        printf("  Pass1: No previous blocks to render\n");
        return;
    }
    
    // Transition volume image for reading
    VkImageMemoryBarrier2 volumeBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    volumeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    volumeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    volumeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    volumeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    volumeBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrier.image = minMaxOutput.volumeImage.image;
    volumeBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    volumeBarrier.subresourceRange.baseMipLevel = 0;
    volumeBarrier.subresourceRange.levelCount = 1;
    volumeBarrier.subresourceRange.baseArrayLayer = 0;
    volumeBarrier.subresourceRange.layerCount = 1;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &volumeBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    
    // Update shading parameters
    {
        void* mapped;
        VK_CHECK(vkMapMemory(device_, shadingParamsMemory_, 0, sizeof(ShadingParameters), 0, &mapped));
        memcpy(mapped, &shadingParams, sizeof(ShadingParameters));
        vkUnmapMemory(device_, shadingParamsMemory_);
    }
    
    // Create view parameters uniform buffer
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
    
    VkBuffer viewUniformBuffer;
    VkDeviceMemory viewUniformMemory;
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = sizeof(ViewUniforms);
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &viewUniformBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, viewUniformBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &viewUniformMemory));
        VK_CHECK(vkBindBufferMemory(device_, viewUniformBuffer, viewUniformMemory, 0));
        
        void* mapped;
        VK_CHECK(vkMapMemory(device_, viewUniformMemory, 0, sizeof(ViewUniforms), 0, &mapped));
        memcpy(mapped, &viewUniforms, sizeof(ViewUniforms));
        vkUnmapMemory(device_, viewUniformMemory);
    }
    
    // Create descriptor pool (for Pass 1 only)
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2}  // TBOs for marching cubes tables
    };
    
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool));
    
    // Create samplers
    VkSampler volumeSampler, minMaxSampler;
    
    VkSamplerCreateInfo volumeSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    volumeSamplerInfo.magFilter = volumeSamplerInfo.minFilter = VK_FILTER_NEAREST;
    volumeSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    volumeSamplerInfo.addressModeU = volumeSamplerInfo.addressModeV = volumeSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &volumeSamplerInfo, nullptr, &volumeSampler));
    
    VkSamplerCreateInfo minMaxSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    minMaxSamplerInfo.magFilter = minMaxSamplerInfo.minFilter = VK_FILTER_NEAREST;
    minMaxSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    minMaxSamplerInfo.addressModeU = minMaxSamplerInfo.addressModeV = minMaxSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &minMaxSamplerInfo, nullptr, &minMaxSampler));
    
    VkImageView volumeImageView = minMaxOutput.volumeImage.imageView;
    
    // Create descriptor set for Pass 1
    VkDescriptorSet pass1DescriptorSet = createPassDescriptorSet(
        descriptorPool, pass1DescriptorSetLayout_,
        viewUniformBuffer, minMaxOutput.minMaxImage.imageView, minMaxSampler,
        volumeImageView, volumeSampler,
        occlusionOutput.pvsPreviousBuffer, 14
    );
    
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
    
    // Begin dynamic rendering - CLEAR depth buffer for Pass 1
    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = colorImageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;    // Clear for first pass
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.2f, 0.3f, 0.8f, 1.0f};  // Blue background
    
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;    // Clear depth for Pass 1
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};
    
    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = scissor;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Render Pass 1: Previous frame's visible geometry
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass1Pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass1PipelineLayout_,
                           0, 1, &pass1DescriptorSet, 0, nullptr);
    
    struct { uint32_t renderPass; uint32_t useUniqueTables; } pc;
    pc.renderPass = 0; // Pass 1
    pc.useUniqueTables = mcTables_.isUnique ? 1 : 0;
    vkCmdPushConstants(cmd, pass1PipelineLayout_, VK_SHADER_STAGE_MESH_BIT_EXT,
                      0, sizeof(pc), &pc);
    
    // Parameterized PVS bypass for testing
    uint32_t pass1Workgroups;
    if (bypassPVS_) {
        // BYPASS PVS - Dispatch ALL blocks for testing
        uint32_t totalBlocks = pushConstants.blockGridDim.x * pushConstants.blockGridDim.y * pushConstants.blockGridDim.z;
        pass1Workgroups = totalBlocks * 2; // 2 workgroups per block
        printf("  Pass1 (BYPASS PVS): Dispatching %u workgroups for ALL %u blocks\n", 
               pass1Workgroups, totalBlocks);
    } else {
        // Use PVS from occlusion culling
        pass1Workgroups = occlusionOutput.pvsPreviousCount * 2;
        printf("  Pass1: Dispatching %u workgroups for %u blocks from PVS\n", 
               pass1Workgroups, occlusionOutput.pvsPreviousCount);
    }
    vkCmdDrawMeshTasksEXT(cmd, pass1Workgroups, 1, 1);
    
    // End rendering
    vkCmdEndRendering(cmd);
    
    // Transition volume image back
    VkImageMemoryBarrier2 volumeBarrierBack{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    volumeBarrierBack.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    volumeBarrierBack.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    volumeBarrierBack.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    volumeBarrierBack.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    volumeBarrierBack.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrierBack.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrierBack.image = minMaxOutput.volumeImage.image;
    volumeBarrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    volumeBarrierBack.subresourceRange.baseMipLevel = 0;
    volumeBarrierBack.subresourceRange.levelCount = 1;
    volumeBarrierBack.subresourceRange.baseArrayLayer = 0;
    volumeBarrierBack.subresourceRange.layerCount = 1;
    
    VkDependencyInfo depInfoBack{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfoBack.imageMemoryBarrierCount = 1;
    depInfoBack.pImageMemoryBarriers = &volumeBarrierBack;
    vkCmdPipelineBarrier2(cmd, &depInfoBack);
    
    // Store Pass 1 temporary resources
    tempResources_.volumeSampler_pass1 = volumeSampler;
    tempResources_.minMaxSampler_pass1 = minMaxSampler;
    tempResources_.descriptorPool_pass1 = descriptorPool;
    tempResources_.viewUniformBuffer_pass1 = viewUniformBuffer;
    tempResources_.viewUniformMemory_pass1 = viewUniformMemory;
}

void TransientExtractionPass::renderPass2_NewlyVisible(
    VkCommandBuffer cmd,
    const RasterOcclusionPass::Output& occlusionOutput,
    const MinMaxOutput& minMaxOutput,
    const PushConstants& pushConstants,
    const glm::mat4& viewProjMatrix,
    const glm::vec3& cameraPos,
    VkImageView colorImageView,
    VkImageView depthImageView,
    VkExtent2D renderExtent,
    const ShadingParameters& shadingParams) {
    
    // Debug tracking
    static int pass2CallCount = 0;
    printf("renderPass2_NewlyVisible called: %d, pvsDifferenceCount=%u\n", 
           pass2CallCount++, occlusionOutput.pvsDifferenceCount);
    
    // Skip if no new blocks to render
    if (occlusionOutput.pvsDifferenceCount == 0) {
        // printf("  Pass2: No new blocks to render\n");
        return;
    }
    
    // Transition volume image for reading
    VkImageMemoryBarrier2 volumeBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    volumeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    volumeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    volumeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    volumeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    volumeBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrier.image = minMaxOutput.volumeImage.image;
    volumeBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    volumeBarrier.subresourceRange.baseMipLevel = 0;
    volumeBarrier.subresourceRange.levelCount = 1;
    volumeBarrier.subresourceRange.baseArrayLayer = 0;
    volumeBarrier.subresourceRange.layerCount = 1;
    
    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &volumeBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    
    // Update shading parameters
    {
        void* mapped;
        VK_CHECK(vkMapMemory(device_, shadingParamsMemory_, 0, sizeof(ShadingParameters), 0, &mapped));
        memcpy(mapped, &shadingParams, sizeof(ShadingParameters));
        vkUnmapMemory(device_, shadingParamsMemory_);
    }
    
    // Create view parameters uniform buffer
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
    
    VkBuffer viewUniformBuffer;
    VkDeviceMemory viewUniformMemory;
    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = sizeof(ViewUniforms);
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &viewUniformBuffer));
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device_, viewUniformBuffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = selectMemoryType(context_.getMemoryProperties(),
                                                     memRequirements.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &viewUniformMemory));
        VK_CHECK(vkBindBufferMemory(device_, viewUniformBuffer, viewUniformMemory, 0));
        
        void* mapped;
        VK_CHECK(vkMapMemory(device_, viewUniformMemory, 0, sizeof(ViewUniforms), 0, &mapped));
        memcpy(mapped, &viewUniforms, sizeof(ViewUniforms));
        vkUnmapMemory(device_, viewUniformMemory);
    }
    
    // Create descriptor pool (for Pass 2 only)
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2}  // TBOs for marching cubes tables
    };
    
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool));
    
    // Create samplers
    VkSampler volumeSampler, minMaxSampler;
    
    VkSamplerCreateInfo volumeSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    volumeSamplerInfo.magFilter = volumeSamplerInfo.minFilter = VK_FILTER_NEAREST;
    volumeSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    volumeSamplerInfo.addressModeU = volumeSamplerInfo.addressModeV = volumeSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &volumeSamplerInfo, nullptr, &volumeSampler));
    
    VkSamplerCreateInfo minMaxSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    minMaxSamplerInfo.magFilter = minMaxSamplerInfo.minFilter = VK_FILTER_NEAREST;
    minMaxSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    minMaxSamplerInfo.addressModeU = minMaxSamplerInfo.addressModeV = minMaxSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &minMaxSamplerInfo, nullptr, &minMaxSampler));
    
    VkImageView volumeImageView = minMaxOutput.volumeImage.imageView;
    
    // Create descriptor set for Pass 2
    VkDescriptorSet pass2DescriptorSet = createPassDescriptorSet(
        descriptorPool, pass2DescriptorSetLayout_,
        viewUniformBuffer, minMaxOutput.minMaxImage.imageView, minMaxSampler,
        volumeImageView, volumeSampler,
        occlusionOutput.pvsDifferenceBuffer, 15
    );
    
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
    
    // Begin dynamic rendering - LOAD existing depth from Pass 1
    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = colorImageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;    // Load from Pass 1
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;    // Load depth from Pass 1
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = scissor;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Render Pass 2: Newly visible geometry
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass2Pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass2PipelineLayout_,
                           0, 1, &pass2DescriptorSet, 0, nullptr);
    
    struct { uint32_t renderPass; uint32_t useUniqueTables; } pc;
    pc.renderPass = 1; // Pass 2
    pc.useUniqueTables = mcTables_.isUnique ? 1 : 0;
    vkCmdPushConstants(cmd, pass2PipelineLayout_, VK_SHADER_STAGE_MESH_BIT_EXT,
                      0, sizeof(pc), &pc);
    
    // Parameterized PVS bypass for testing
    uint32_t pass2Workgroups;
    if (bypassPVS_) {
        // When bypassing PVS, Pass 2 shouldn't run (handled in outer condition)
        // This code shouldn't be reached when bypassPVS is true
        pass2Workgroups = 0;
        printf("  Pass2: SKIPPED (PVS bypass mode)\n");
    } else {
        // Use PVS difference from occlusion culling
        pass2Workgroups = occlusionOutput.pvsDifferenceCount * 2;
        printf("  Pass2: Dispatching %u workgroups for %u blocks from PVS difference\n", 
               pass2Workgroups, occlusionOutput.pvsDifferenceCount);
    }
    
    if (pass2Workgroups > 0) {
        vkCmdDrawMeshTasksEXT(cmd, pass2Workgroups, 1, 1);
    }
    
    // End rendering
    vkCmdEndRendering(cmd);
    
    // Transition volume image back
    VkImageMemoryBarrier2 volumeBarrierBack{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    volumeBarrierBack.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    volumeBarrierBack.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    volumeBarrierBack.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    volumeBarrierBack.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    volumeBarrierBack.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrierBack.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    volumeBarrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    volumeBarrierBack.image = minMaxOutput.volumeImage.image;
    volumeBarrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    volumeBarrierBack.subresourceRange.baseMipLevel = 0;
    volumeBarrierBack.subresourceRange.levelCount = 1;
    volumeBarrierBack.subresourceRange.baseArrayLayer = 0;
    volumeBarrierBack.subresourceRange.layerCount = 1;
    
    VkDependencyInfo depInfoBack{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfoBack.imageMemoryBarrierCount = 1;
    depInfoBack.pImageMemoryBarriers = &volumeBarrierBack;
    vkCmdPipelineBarrier2(cmd, &depInfoBack);
    
    // Store Pass 2 temporary resources
    tempResources_.volumeSampler_pass2 = volumeSampler;
    tempResources_.minMaxSampler_pass2 = minMaxSampler;
    tempResources_.descriptorPool_pass2 = descriptorPool;
    tempResources_.viewUniformBuffer_pass2 = viewUniformBuffer;
    tempResources_.viewUniformMemory_pass2 = viewUniformMemory;
}

void TransientExtractionPass::createMarchingCubesTables(bool useUniqueTables) {
    mcTables_.isUnique = useUniqueTables;
    
    // Create buffer for numVertices table (256 entries of uint8)
    size_t numVerticesSize = 256 * sizeof(uint8_t);
    createBuffer(mcTables_.numVerticesBuffer, device_, context_.getMemoryProperties(),
                 numVerticesSize, 
                 VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Create buffer for triangle table
    size_t triTableSize;
    const void* triTableData;
    const void* numVerticesData;
    
    if (useUniqueTables) {
        // Unique tables: 256 x 16 uint8 values (changed from int32)
        triTableSize = 256 * 16 * sizeof(uint8_t);
        triTableData = &MarchingCubes::uniqueTriTable[0][0];  // Flatten 2D array
        numVerticesData = &MarchingCubes::numUniqueVertsTable[0];
    } else {
        // Standard tables: 256 x 16 uint8 values  
        triTableSize = 256 * 16 * sizeof(uint8_t);
        triTableData = &MarchingCubes::triTable[0][0];  // Flatten 2D array
        numVerticesData = &MarchingCubes::numVerticesTable[0];
    }
    
    createBuffer(mcTables_.triTableBuffer, device_, context_.getMemoryProperties(),
                 triTableSize,
                 VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Upload data to buffers using staging
    Buffer stagingBuffer;
    
    // Upload numVertices table
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 numVerticesSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, numVerticesData, numVerticesSize);
    
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion{};
    copyRegion.size = numVerticesSize;
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, mcTables_.numVerticesBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    destroyBuffer(stagingBuffer, device_);
    
    // Upload triangle table
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 triTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, triTableData, triTableSize);
    
    cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    copyRegion.size = triTableSize;
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, mcTables_.triTableBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    destroyBuffer(stagingBuffer, device_);
    
    // Create buffer views for shader access
    VkBufferViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
    
    // NumVertices view (R8_UINT format for uint8 data)
    viewInfo.buffer = mcTables_.numVerticesBuffer.buffer;
    viewInfo.format = VK_FORMAT_R8_UINT;
    viewInfo.offset = 0;
    viewInfo.range = numVerticesSize;
    VK_CHECK(vkCreateBufferView(device_, &viewInfo, nullptr, &mcTables_.numVerticesView));
    
    // Triangle table view (R8_UINT format for uint8 data)
    viewInfo.buffer = mcTables_.triTableBuffer.buffer;
    viewInfo.format = VK_FORMAT_R8_UINT;  // Changed from R32_SINT to R8_UINT
    viewInfo.offset = 0;
    viewInfo.range = triTableSize;
    VK_CHECK(vkCreateBufferView(device_, &viewInfo, nullptr, &mcTables_.triTableView));
}

void TransientExtractionPass::destroyMarchingCubesTables() {
    if (mcTables_.numVerticesView != VK_NULL_HANDLE) {
        vkDestroyBufferView(device_, mcTables_.numVerticesView, nullptr);
        mcTables_.numVerticesView = VK_NULL_HANDLE;
    }
    if (mcTables_.triTableView != VK_NULL_HANDLE) {
        vkDestroyBufferView(device_, mcTables_.triTableView, nullptr);
        mcTables_.triTableView = VK_NULL_HANDLE;
    }
    
    destroyBuffer(mcTables_.numVerticesBuffer, device_);
    destroyBuffer(mcTables_.triTableBuffer, device_);
}