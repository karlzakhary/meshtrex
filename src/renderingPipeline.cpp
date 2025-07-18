#include "renderingPipeline.h"

RenderingPipeline::~RenderingPipeline() {
    cleanup();
}

void RenderingPipeline::setup(VkDevice device, VkFormat colorFormat, VkFormat depthFormat, VkPipelineLayout layout) {
    device_ = device;

    // Load shaders
    assert(loadShader(meshShader_, device, "/spirv/render.mesh.spv"));
    assert(loadShader(taskShader_, device, "/spirv/render.task.spv"));
    assert(loadShader(fragShader_, device, "/spirv/render.frag.spv"));

    VkPipelineShaderStageCreateInfo stages[3] = {};
    // stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_TASK_BIT_NV, taskShader_.module, "main", nullptr};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_TASK_BIT_EXT, taskShader_.module, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshShader_.module, "main", nullptr};
    stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragShader_.module, "main", nullptr};

    VkGraphicsPipelineCreateInfo createInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    createInfo.stageCount = 3;
    createInfo.pStages = stages;

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
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT; // Re-enabled backface culling
    createInfo.pRasterizationState = &rasterizationState;

    VkPipelineMultisampleStateCreateInfo multisampleState = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    createInfo.pMultisampleState = &multisampleState;

    // Corrected Depth State for Reversed-Z
    VkPipelineDepthStencilStateCreateInfo depthStencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencilState.depthTestEnable = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_TRUE;
    depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER; // Use GREATER for reversed-Z (far plane at 0.0)
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

    createInfo.layout = layout;
    createInfo.renderPass = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline_));
}


void RenderingPipeline::cleanup() {
    if (device_ != VK_NULL_HANDLE) {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (meshShader_.module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, meshShader_.module, nullptr);
            meshShader_.module = VK_NULL_HANDLE;
        }
        if (taskShader_.module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, taskShader_.module, nullptr);
            taskShader_.module = VK_NULL_HANDLE;
        }
        if (fragShader_.module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, fragShader_.module, nullptr);
            fragShader_.module = VK_NULL_HANDLE;
        }
        device_ = VK_NULL_HANDLE;
    }
}