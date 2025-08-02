#include "transientExtractionPipeline.h"
#include "vulkan_utils.h"
#include "shaders.h"
#include <iostream>
#include <stdexcept>

TransientExtractionPipeline::~TransientExtractionPipeline() {
    cleanup();
}

TransientExtractionPipeline::TransientExtractionPipeline(TransientExtractionPipeline&& other) noexcept {
    device_ = other.device_;
    pipelineLayout_ = other.pipelineLayout_;
    pipeline_ = other.pipeline_;
    descriptorSetLayout_ = other.descriptorSetLayout_;
    descriptorPool_ = other.descriptorPool_;
    descriptorSet_ = other.descriptorSet_;
    taskShader_ = std::move(other.taskShader_);
    meshShader_ = std::move(other.meshShader_);
    fragShader_ = std::move(other.fragShader_);
    
    other.transferResourceOwnership();
}

TransientExtractionPipeline& TransientExtractionPipeline::operator=(TransientExtractionPipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        device_ = other.device_;
        pipelineLayout_ = other.pipelineLayout_;
        pipeline_ = other.pipeline_;
        descriptorSetLayout_ = other.descriptorSetLayout_;
        descriptorPool_ = other.descriptorPool_;
        descriptorSet_ = other.descriptorSet_;
        taskShader_ = std::move(other.taskShader_);
        meshShader_ = std::move(other.meshShader_);
        fragShader_ = std::move(other.fragShader_);
        
        other.transferResourceOwnership();
    }
    return *this;
}

bool TransientExtractionPipeline::setup(
    VkDevice device,
    VkFormat colorFormat,
    VkFormat depthFormat,
    uint32_t blockX,
    uint32_t blockY,
    uint32_t blockZ,
    bool pmb
) {
    const char* taskShaderPath = pmb ? "/spirv/transient_marching_cubes_pmb.task.spv" : "/spirv/transient_marching_cubes.task.spv";
    const char* meshShaderPath = pmb ? "/spirv/transient_marching_cubes_pmb.mesh.spv" : "/spirv/transient_marching_cubes.mesh.spv";
    const char* fragShaderPath = "/spirv/transient_marching_cubes.frag.spv";
    
    return setupWithShaders(device, colorFormat, depthFormat, blockX, blockY, blockZ, pmb, 
                           taskShaderPath, meshShaderPath, fragShaderPath);
}

bool TransientExtractionPipeline::setupWithShaders(
    VkDevice device,
    VkFormat colorFormat,
    VkFormat depthFormat,
    uint32_t blockX,
    uint32_t blockY,
    uint32_t blockZ,
    bool pmb,
    const char* taskShaderPath,
    const char* meshShaderPath,
    const char* fragShaderPath
) {
    device_ = device;
    
    try {
        if (!loadShader(taskShader_, device, taskShaderPath)) {
            throw std::runtime_error("Failed to load task shader");
        }
        if (!loadShader(meshShader_, device, meshShaderPath)) {
            throw std::runtime_error("Failed to load mesh shader");
        }
        if (!loadShader(fragShader_, device, fragShaderPath)) {
            throw std::runtime_error("Failed to load fragment shader");
        }
        
        createPipelineLayout();
        createTransientExtractionGraphicsPipeline(colorFormat, depthFormat, blockX, blockY, blockZ, pmb);
        createDescriptorPool();
        allocateDescriptorSets();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to setup transient extraction pipeline: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void TransientExtractionPipeline::cleanup() {
    if (device_ != VK_NULL_HANDLE) {
        releaseResources();
    }
}

void TransientExtractionPipeline::releaseResources() {
    if (device_ == VK_NULL_HANDLE) return;
    
    vkDeviceWaitIdle(device_);
    
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }
    
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }
    
    if (taskShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, taskShader_.module, nullptr);
        taskShader_.module = VK_NULL_HANDLE;
    }
    if (meshShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, meshShader_.module, nullptr);
        meshShader_.module = VK_NULL_HANDLE;
    }
    if (fragShader_.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, fragShader_.module, nullptr);
        fragShader_.module = VK_NULL_HANDLE;
    }
    
    device_ = VK_NULL_HANDLE;
}

void TransientExtractionPipeline::createPipelineLayout() {
    // Descriptor set layout for transient extraction
    // Bindings: UBO, Volume Image, Min-Max Image, Active Block Count, Active Block IDs, MC Tables
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_TASK_BIT_EXT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
    };
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    
    // Push constants for view-projection matrix and frustum culling
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4) + sizeof(glm::vec4) * 6; // viewProj + 6 frustum planes
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
}

void TransientExtractionPipeline::createTransientExtractionGraphicsPipeline(
    VkFormat colorFormat,
    VkFormat depthFormat,
    uint32_t blockX,
    uint32_t blockY,
    uint32_t blockZ,
    bool pmb
) {
    // Shader stages
    VkPipelineShaderStageCreateInfo shaderStages[3] = {};
    
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_TASK_BIT_EXT;
    shaderStages[0].module = taskShader_.module;
    shaderStages[0].pName = "main";
    
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    shaderStages[1].module = meshShader_.module;
    shaderStages[1].pName = "main";
    
    shaderStages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[2].module = fragShader_.module;
    shaderStages[2].pName = "main";
    
    // Specialization constants
    struct SpecializationData {
        uint32_t blockX;
        uint32_t blockY;
        uint32_t blockZ;
    } specializationData = {blockX, blockY, blockZ};
    
    VkSpecializationMapEntry specializationMapEntries[3] = {
        {0, offsetof(SpecializationData, blockX), sizeof(uint32_t)},
        {1, offsetof(SpecializationData, blockY), sizeof(uint32_t)},
        {2, offsetof(SpecializationData, blockZ), sizeof(uint32_t)},
    };
    
    VkSpecializationInfo specializationInfo{};
    specializationInfo.mapEntryCount = 3;
    specializationInfo.pMapEntries = specializationMapEntries;
    specializationInfo.dataSize = sizeof(SpecializationData);
    specializationInfo.pData = &specializationData;
    
    shaderStages[0].pSpecializationInfo = &specializationInfo;
    shaderStages[1].pSpecializationInfo = &specializationInfo;
    
    VkGraphicsPipelineCreateInfo createInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    createInfo.stageCount = 3;
    createInfo.pStages = shaderStages;

    VkPipelineVertexInputStateCreateInfo vertexInput = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    createInfo.pVertexInputState = &vertexInput;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    createInfo.pInputAssemblyState = &inputAssembly;

    VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    createInfo.pViewportState = &viewportState;

    VkPipelineRasterizationStateCreateInfo rasterizationState = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizationState.lineWidth = 1.f;
    rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // Marching cubes typically generates CCW triangles
    rasterizationState.cullMode = VK_CULL_MODE_NONE; // Disable culling for debugging
    createInfo.pRasterizationState = &rasterizationState;

    VkPipelineMultisampleStateCreateInfo multisampleState = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    createInfo.pMultisampleState = &multisampleState;

    // Depth State Configuration
    VkPipelineDepthStencilStateCreateInfo depthStencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencilState.depthTestEnable = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_TRUE;
    depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER; // Standard depth test (near plane at 0.0)
    createInfo.pDepthStencilState = &depthStencilState;

    VkPipelineColorBlendAttachmentState colorAttachmentState = {};
    colorAttachmentState.blendEnable = VK_FALSE; // Disable blending for opaque geometry
    colorAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlendState = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &colorAttachmentState;
    createInfo.pColorBlendState = &colorBlendState;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
    createInfo.pDynamicState = &dynamicState;

    VkPipelineRenderingCreateInfo renderingInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat = depthFormat;
    createInfo.pNext = &renderingInfo;

    createInfo.layout = pipelineLayout_;
    createInfo.renderPass = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline_));
}

void TransientExtractionPipeline::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
    };
    
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void TransientExtractionPipeline::allocateDescriptorSets() {
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;
    
    if (vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }
}