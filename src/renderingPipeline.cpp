#include "renderingPipeline.h"
#include "vulkan_utils.h"

#include <vector>
#include <iostream>
#include <utility>

// Move Constructor/Assignment
RenderingPipeline::RenderingPipeline(RenderingPipeline&& other) noexcept :
    device_(other.device_),
    pipelineLayout_(other.pipelineLayout_),
    pipeline_(other.pipeline_),
    descriptorSetLayout_(other.descriptorSetLayout_),
    descriptorPool_(other.descriptorPool_),
    descriptorSets_(other.descriptorSets_),
    meshShader_(std::move(other.meshShader_)),
    taskShader_(std::move(other.taskShader_)),
    fragmentShader_(std::move(other.fragmentShader_))
{
    other.device_ = VK_NULL_HANDLE;
    other.pipelineLayout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
    other.descriptorSetLayout_ = VK_NULL_HANDLE;
    other.descriptorPool_ = VK_NULL_HANDLE;
    other.descriptorSets_.clear();
}

RenderingPipeline& RenderingPipeline::operator=(RenderingPipeline&& other) noexcept {
    if (this != &other) {
        releaseResources();
        device_ = other.device_;
        pipelineLayout_ = other.pipelineLayout_;
        pipeline_ = other.pipeline_;
        descriptorSetLayout_ = other.descriptorSetLayout_;
        descriptorPool_ = other.descriptorPool_;
        descriptorSets_ = other.descriptorSets_;
        meshShader_ = std::move(other.meshShader_);
        taskShader_ = std::move(other.taskShader_);
        fragmentShader_ = std::move(other.fragmentShader_);
        other.device_ = VK_NULL_HANDLE;
        other.pipelineLayout_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
        other.descriptorSetLayout_ = VK_NULL_HANDLE;
        other.descriptorPool_ = VK_NULL_HANDLE;
        other.descriptorSets_.clear();
    }
    return *this;
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
    if(taskShader_.module != VK_NULL_HANDLE) vkDestroyShaderModule(device_, taskShader_.module, nullptr);
    if(meshShader_.module != VK_NULL_HANDLE) vkDestroyShaderModule(device_, meshShader_.module, nullptr);
    if(fragmentShader_.module != VK_NULL_HANDLE) vkDestroyShaderModule(device_, fragmentShader_.module, nullptr);
    if (pipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    if (descriptorPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (descriptorSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);

    pipeline_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    descriptorPool_ = VK_NULL_HANDLE;
    descriptorSetLayout_ = VK_NULL_HANDLE;
    descriptorSets_.clear();
    device_ = VK_NULL_HANDLE;
}

void RenderingPipeline::createPipelineLayout() {
    // Define bindings for task, mesh, and fragment shaders
    std::vector<VkDescriptorSetLayoutBinding> bindings(5);

    // Binding 0: Scene Uniform Buffer (MVP, lighting, etc.)
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    // Binding 1: Meshlet Descriptor Buffer
    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };

    // Binding 2: Vertex Buffer
    bindings[2] = {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
    };

    // Binding 3: Index Buffer
    bindings[3] = {
        .binding = 3,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
    };

    // Binding 4: Meshlet Descriptor count Buffer
    bindings[4] = {
        .binding = 4,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));
}

void RenderingPipeline::createGraphicsPipeline(VkFormat colorFormat, VkFormat depthFormat, VkSampleCountFlagBits msaaSamples) {
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &colorFormat;
    pipelineRenderingInfo.depthAttachmentFormat = depthFormat;

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_TASK_BIT_EXT, taskShader_.module, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshShader_.module, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader_.module, "main", nullptr}
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE };
    VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr };
    VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f };
    VkPipelineMultisampleStateCreateInfo multisampling = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, msaaSamples, VK_FALSE, 0.0f, nullptr, VK_FALSE, VK_FALSE };
    VkPipelineDepthStencilStateCreateInfo depthStencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS, VK_FALSE, VK_FALSE, {}, {}, 0.0f, 1.0f };
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlending = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &colorBlendAttachment, {0.0f, 0.0f, 0.0f, 0.0f} };
    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, static_cast<uint32_t>(dynamicStates.size()), dynamicStates.data() };

    // The vertex input state is not used with mesh shaders
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};


    VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext = &pipelineRenderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo; // Pass the empty struct
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencilState;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_));
    std::cout << "Rendering pipeline (Task/Mesh/Fragment) created." << std::endl;
}


void RenderingPipeline::createDescriptorPoolAndSets(uint32_t maxFramesInFlight) {
    std::vector<VkDescriptorPoolSize> poolSizes;
    // Pool needs to be large enough for all sets
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxFramesInFlight});
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * maxFramesInFlight}); // 4 storage buffers per set

    VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxFramesInFlight; // Allocate enough for all in-flight frames
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));

    // **FIX:** Allocate one descriptor set for each frame in flight
    std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = maxFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();
    
    descriptorSets_.resize(maxFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets_.data()));
    std::cout << "Rendering descriptor sets allocated." << std::endl;
}


bool RenderingPipeline::setup(
    VkDevice device,
    VkFormat colorFormat,
    VkFormat depthFormat,
    VkSampleCountFlagBits msaaSamples,
    uint32_t maxFramesInFlight
) {
    if (device_ != VK_NULL_HANDLE) {
        std::cerr << "RenderingPipeline already initialized." << std::endl;
        return false;
    }
    device_ = device;

    // Load shaders
    assert(loadShader(taskShader_, device_, "/spirv/render.task.spv"));
    assert(loadShader(meshShader_, device_, "/spirv/render.mesh.spv"));
    assert(loadShader(fragmentShader_, device_, "/spirv/render.frag.spv"));
    std::cout << "Rendering shaders loaded." << std::endl;

    createPipelineLayout();
    createGraphicsPipeline(colorFormat, depthFormat, msaaSamples);
    createDescriptorPoolAndSets(maxFramesInFlight);

    return true;
}