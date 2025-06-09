#include "extractionPipeline.h"
#include "vulkan_utils.h"   // Assumed helpers: VK_CHECK, loadShader
#include "shaders.h"        // Assumed definition of Shader struct
#include <vector>
#include <iostream>
#include <utility> // For std::move, std::swap

// Move Constructor
ExtractionPipeline::ExtractionPipeline(ExtractionPipeline&& other) noexcept :
    device_(other.device_),
    pipelineLayout_(other.pipelineLayout_),
    pipeline_(other.pipeline_),
    descriptorSetLayout_(other.descriptorSetLayout_),
    descriptorPool_(other.descriptorPool_),
    descriptorSet_(other.descriptorSet_),
    taskShader_(std::move(other.taskShader_)),
    meshShader_(std::move(other.meshShader_))
{
    // Nullify other's handles to prevent double deletion
    other.device_ = VK_NULL_HANDLE;
    other.pipelineLayout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
    other.descriptorSetLayout_ = VK_NULL_HANDLE;
    other.descriptorPool_ = VK_NULL_HANDLE;
    other.descriptorSet_ = VK_NULL_HANDLE;
}

// Move Assignment Operator
ExtractionPipeline& ExtractionPipeline::operator=(ExtractionPipeline&& other) noexcept {
    if (this != &other) {
        // Release existing resources first
        releaseResources();

        // Move resources from other
        device_ = other.device_;
        pipelineLayout_ = other.pipelineLayout_;
        pipeline_ = other.pipeline_;
        descriptorSetLayout_ = other.descriptorSetLayout_;
        descriptorPool_ = other.descriptorPool_;
        descriptorSet_ = other.descriptorSet_;
        taskShader_ = std::move(other.taskShader_);
        meshShader_ = std::move(other.meshShader_);

        // Nullify other's handles
        other.device_ = VK_NULL_HANDLE;
        other.pipelineLayout_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
        other.descriptorSetLayout_ = VK_NULL_HANDLE;
        other.descriptorPool_ = VK_NULL_HANDLE;
        other.descriptorSet_ = VK_NULL_HANDLE;
    }
    return *this;
}


// Destructor
ExtractionPipeline::~ExtractionPipeline() {
    releaseResources();
}

// Explicit cleanup
void ExtractionPipeline::cleanup() {
    releaseResources();
}

// Internal resource release helper
void ExtractionPipeline::releaseResources() {
    if (device_ == VK_NULL_HANDLE) {
        return; // Nothing to release
    }

    // Destroy in reverse order of creation
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    // Shader modules cleaned up by Shader struct's destructor (RAII)
    vkDestroyShaderModule(device_, taskShader_.module, nullptr);
    vkDestroyShaderModule(device_, meshShader_.module, nullptr);

    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    // Descriptor pool also destroys allocated sets
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE; // Set is implicitly invalid now
        descriptorSet_ = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }

    // Don't nullify device_ here as it might be owned elsewhere
    device_ = VK_NULL_HANDLE; // Indicate cleanup is done by nullifying device handle locally
}

void ExtractionPipeline::createPipelineLayout()
{
    // --- Create Descriptor Set Layout ---
    std::vector<VkDescriptorSetLayoutBinding> bindings(12);
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // UBO (PushConstants)

    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // Volume

    bindings[2] = {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // Active Block count

    bindings[3] = {
        .binding = 3,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // Block IDs

    bindings[4] = {
        .binding = 4,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // MC Triangle Table

    bindings[5] = {
        .binding = 5,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // MC Edge Table

    bindings[6] = {
        .binding = 6,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // output Vertex Buffer (Storage Buffer)

    bindings[7] = {
        .binding = 7,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // output Vertex count Buffer (Storage Buffer)

    bindings[8] = {
        .binding = 8,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // output Index Buffer (Storage Buffer)

    bindings[9] = {
        .binding = 9,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
    };  // output Index count Buffer (Storage Buffer)

    bindings[10] = {
        .binding = 10,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, // Task for atomic alloc, Mesh for write
    };  // Output Meshlet Descriptor Buffer (Storage Write)

    bindings[11] = {
        .binding = 11,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags =
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, // Task for atomic alloc, Mesh for write
    };  // Output Meshlet Descriptor counter Buffer (Storage Write)

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = 0,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr,
                                         &descriptorSetLayout_));

    // --- Create Pipeline Layout ---
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout_,
    };

    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr,
                                    &pipelineLayout_));
}
void ExtractionPipeline::createExtractionGraphicsPipeline(VkFormat colorFormat,
                                                          VkFormat depthFormat)
{
    // --- Define Rendering Info (needed for validation even with rasterizer
    // discard) ---
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    pipelineRenderingInfo.colorAttachmentCount =
        1;  // Match what vkCmdBeginRendering expects later
    pipelineRenderingInfo.pColorAttachmentFormats = &colorFormat;
    pipelineRenderingInfo.depthAttachmentFormat = depthFormat;
    pipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_TASK_BIT_EXT, taskShader_.module, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_MESH_BIT_EXT, meshShader_.module, "main", nullptr}};
    // Most states are irrelevant if rasterization is discarded, but must be
    // valid.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};  // Empty
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_FALSE};  // Irrelevant but required

    VkPipelineViewportStateCreateInfo viewportState = {
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        nullptr,
        0,
        1,
        nullptr,
        1,
        nullptr};  // Dynamic

    VkPipelineMultisampleStateCreateInfo multisampling = {
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        nullptr,
        0,
        VK_SAMPLE_COUNT_1_BIT,
        VK_FALSE,
        0.0f,
        nullptr,
        VK_FALSE,
        VK_FALSE};  // No multisampling

    VkPipelineDepthStencilStateCreateInfo depthStencilState = {
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};  // Depth/stencil
                                                                      // disabled
    depthStencilState.depthTestEnable = VK_FALSE;
    depthStencilState.depthWriteEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlendState = {
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};  // No color
                                                                    // blend
                                                                    // attachments
                                                                    // needed/used
    colorBlendState.attachmentCount = 0;  // No attachments actually written to
    colorBlendState.pAttachments = nullptr;

    VkPipelineRasterizationStateCreateInfo rasterizationState = {
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizationState.rasterizerDiscardEnable =
        VK_TRUE;  // *** KEY CHANGE: Disable rasterizer ***
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;  // Irrelevant
    rasterizationState.cullMode = VK_CULL_MODE_NONE;        // Irrelevant
    rasterizationState.frontFace =
        VK_FRONT_FACE_COUNTER_CLOCKWISE;  // Irrelevant
    rasterizationState.lineWidth = 1.0f;  // Irrelevant

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};  // Still required even if not used
    VkPipelineDynamicStateCreateInfo dynamicState = {
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount =
        static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &pipelineRenderingInfo;  // Link dynamic rendering info
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencilState;
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pRasterizationState =
        &rasterizationState;  // Use state with discard=TRUE
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1,
                                       &pipelineInfo, nullptr, &pipeline_));
    std::cout << "Extraction pipeline (Task/Mesh, Rasterizer Discard) created."
              << std::endl;
}

void ExtractionPipeline::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1});  // UBO
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1});   // Volume
    // Block count + IDs + MC Triangle Table + MC Number vertices + VB + VBC + IB + IBC + Meshlet Descriptor + Meshlet Descriptor Counter
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10});

    VkDescriptorPoolCreateInfo poolInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;  // Assuming one set needed
    VK_CHECK(
        vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
}

void ExtractionPipeline::allocateDescriptorSets()
{
    VkDescriptorSetAllocateInfo allocInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_));
}
// Setup method implementation
bool ExtractionPipeline::setup(
    VkDevice device,
    VkFormat colorFormat,
    VkFormat depthFormat
) {
    // Prevent double setup without cleanup
    if (device_ != VK_NULL_HANDLE) {
        std::cerr << "ExtractionPipeline::setup called on an already initialized object. Call cleanup() first." << std::endl;
        return false;
    }
    device_ = device;

    // --- Load Shaders ---
    std::string taskShaderPath = "/spirv/marching_cubes_pmb.task.spv";
    std::string meshShaderPath = "/spirv/marching_cubes_pmb.mesh.spv";

    assert(loadShader(taskShader_, device_, taskShaderPath.c_str()));
    assert(loadShader(meshShader_, device_, meshShaderPath.c_str()));
    std::cout << "Extraction shaders loaded." << std::endl;

    createPipelineLayout();
    createExtractionGraphicsPipeline(colorFormat, depthFormat);
    // --- Create Descriptor Pool & Allocate Set ---
    createDescriptorPool();
    allocateDescriptorSets();
    std::cout << "Extraction descriptor set allocated." << std::endl;

    return true;
}