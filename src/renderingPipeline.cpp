#include "renderingPipeline.h"
#include "vulkan_utils.h"

#include <vector>
#include <iostream>
#include <utility>

// Move Constructor/Assignment (Similar to ExtractionPipeline)
RenderingPipeline::RenderingPipeline(RenderingPipeline&& other) noexcept :
    device_(other.device_),
    pipelineLayout_(other.pipelineLayout_),
    pipeline_(other.pipeline_),
    descriptorSetLayout_(other.descriptorSetLayout_),
    descriptorPool_(other.descriptorPool_),
    descriptorSet_(other.descriptorSet_),
    meshShader_(std::move(other.meshShader_)),
    taskShader_(std::move(other.taskShader_)),
    fragmentShader_(std::move(other.fragmentShader_))
{ /* Nullify other's handles */ other.device_ = VK_NULL_HANDLE; other.pipelineLayout_ = VK_NULL_HANDLE; other.pipeline_ = VK_NULL_HANDLE; other.descriptorSetLayout_ = VK_NULL_HANDLE; other.descriptorPool_ = VK_NULL_HANDLE; other.descriptorSet_ = VK_NULL_HANDLE; }
RenderingPipeline& RenderingPipeline::operator=(RenderingPipeline&& other) noexcept {
    if (this != &other) { releaseResources(); /* Move members */ device_ = other.device_; pipelineLayout_ = other.pipelineLayout_; pipeline_ = other.pipeline_; descriptorSetLayout_ = other.descriptorSetLayout_; descriptorPool_ = other.descriptorPool_; descriptorSet_ = other.descriptorSet_; meshShader_ = std::move(other.meshShader_); taskShader_ = std::move(other.taskShader_); fragmentShader_ = std::move(other.fragmentShader_); /* Nullify other */ other.device_ = VK_NULL_HANDLE; other.pipelineLayout_ = VK_NULL_HANDLE; other.pipeline_ = VK_NULL_HANDLE; other.descriptorSetLayout_ = VK_NULL_HANDLE; other.descriptorPool_ = VK_NULL_HANDLE; other.descriptorSet_ = VK_NULL_HANDLE; } return *this;
}

RenderingPipeline::~RenderingPipeline() {
    releaseResources();
}

void RenderingPipeline::cleanup() {
    releaseResources();
}

void RenderingPipeline::releaseResources() {
    if (device_ == VK_NULL_HANDLE) return;
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
    vkDestroyShaderModule(device_, meshShader_.module, nullptr);
    vkDestroyShaderModule(device_, taskShader_.module, nullptr);
    vkDestroyShaderModule(device_, fragmentShader_.module, nullptr);
    if (pipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    if (descriptorPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (descriptorSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
    pipeline_ = VK_NULL_HANDLE; pipelineLayout_ = VK_NULL_HANDLE; descriptorPool_ = VK_NULL_HANDLE; descriptorSetLayout_ = VK_NULL_HANDLE; descriptorSet_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

void RenderingPipeline::createPipelineLayout() {
    // --- Create Descriptor Set Layout ---
    // Define bindings needed by VS/FS (e.g., UBO for MVP)
    std::vector<VkDescriptorSetLayoutBinding> bindings(4);

    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
    }; // Binding 0: Scene UBO (MVP, Lighting - TS/MS/FS)

    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT,
    }; // Binding 1: Meshlet Descriptor Buffer (TS)

    bindings[2] = {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
    }; //Binding 2: Vertex Buffer (MS)

    bindings[3] = {
        .binding = 3,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
    }; //Binding 3: Index buffer (MS)

    VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));

    // --- Create Pipeline Layout ---
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));
}

bool RenderingPipeline::setup(
    VkDevice device,
    VkFormat colorFormat,
    VkFormat depthFormat,
    VkSampleCountFlagBits msaaSamples
) {
    if (device_ != VK_NULL_HANDLE) { /* Error */ return false; }
    device_ = device;

    // --- Load Shaders ---
    std::string meshShaderPath = "/spirv/render.mesh.spv"; // Example name
    std::string taskShaderPath = "/spirv/render.task.spv"; // Example name
    std::string fragShaderPath = "/spirv/render.frag.spv"; // Example name
    assert(loadShader(meshShader_, device_, meshShaderPath.c_str()));
    assert(loadShader(taskShader_, device_, taskShaderPath.c_str()));
    assert(loadShader(fragmentShader_, device_, fragShaderPath.c_str()));
    std::cout << "Rendering MS/TS/FS shaders loaded." << std::endl;

    createPipelineLayout();
    // --- Define Rendering Info ---
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &colorFormat;
    pipelineRenderingInfo.depthAttachmentFormat = depthFormat;
    pipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    // --- Create Graphics Pipeline (Standard VS/FS) ---
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshShader_.module, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_TASK_BIT_EXT, taskShader_.module, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader_.module, "main", nullptr}
    };

    // --- Other Pipeline States (Standard rendering setup) ---
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE };
    VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr };
    VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f };
    VkPipelineMultisampleStateCreateInfo multisampling = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, msaaSamples, VK_FALSE, 0.0f, nullptr, VK_FALSE, VK_FALSE };
    VkPipelineDepthStencilStateCreateInfo depthStencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL, VK_FALSE, VK_FALSE, {}, {}, 0.0f, 1.0f }; // Usual depth op
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlending = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &colorBlendAttachment, {0.0f, 0.0f, 0.0f, 0.0f} };
    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, static_cast<uint32_t>(dynamicStates.size()), dynamicStates.data() };

    VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext = &pipelineRenderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer; // Rasterizer enabled
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencilState; // Depth enabled
    pipelineInfo.pColorBlendState = &colorBlending;       // Color blend enabled
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_));
    std::cout << "Rendering pipeline (VS/FS) created." << std::endl;

    // --- Create Descriptor Pool & Allocate Set ---
    std::vector<VkDescriptorPoolSize> poolSizes (2);
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}); // Scene UBO
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}); // MeshletDesc + VB + IB
    VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1; // Assuming one set needed
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));

    VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_));
    std::cout << "Rendering descriptor set allocated." << std::endl;

    return true;
}