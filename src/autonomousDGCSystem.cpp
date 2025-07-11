#include "autonomousDGCSystem.h"
#include "shaders.h"
#include "vulkan_utils.h"
#include "resources.h"
#include <iostream>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

AutonomousDGCSystem::AutonomousDGCSystem(VulkanContext& context, DeviceGeneratedCommands* dgcSystem)
    : context_(context), device_(context.getDevice()), dgcSystem_(dgcSystem),
      currentParameter_(0.0f), volumeFitsInMemory_(false), totalPageCount_(0),
      stateAnalysisPipeline_(VK_NULL_HANDLE), memoryAnalysisPipeline_(VK_NULL_HANDLE),
      prioritizationPipeline_(VK_NULL_HANDLE), commandGenerationPipeline_(VK_NULL_HANDLE),
      descriptorLayout_(VK_NULL_HANDLE), descriptorSet_(VK_NULL_HANDLE),
      pipelineLayout_(VK_NULL_HANDLE), descriptorPool_(VK_NULL_HANDLE) {
}

AutonomousDGCSystem::~AutonomousDGCSystem() {
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    }
    
    if (stateAnalysisPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, stateAnalysisPipeline_, nullptr);
    }
    if (memoryAnalysisPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, memoryAnalysisPipeline_, nullptr);
    }
    if (prioritizationPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, prioritizationPipeline_, nullptr);
    }
    if (commandGenerationPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, commandGenerationPipeline_, nullptr);
    }
    
    if (descriptorLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorLayout_, nullptr);
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    }
    
    destroyBuffer(extractionStateBuffer_, device_);
    destroyBuffer(memoryStateBuffer_, device_);
    destroyBuffer(pageValidityBuffer_, device_);
    destroyBuffer(commandQueueBuffer_, device_);
    destroyBuffer(pageResidencyBuffer_, device_);
    destroyBuffer(viewParametersBuffer_, device_);
    destroyBuffer(pagePriorityBuffer_, device_);
}

void AutonomousDGCSystem::initialize(const AutonomousDGCConfig& config) {
    config_ = config;
    
    // Calculate total pages
    uint32_t pagesX = (config.volumeDimX + config.pageSizeX - 1) / config.pageSizeX;
    uint32_t pagesY = (config.volumeDimY + config.pageSizeY - 1) / config.pageSizeY;
    uint32_t pagesZ = (config.volumeDimZ + config.pageSizeZ - 1) / config.pageSizeZ;
    totalPageCount_ = pagesX * pagesY * pagesZ;
    
    std::cout << "Initializing Autonomous DGC System" << std::endl;
    std::cout << "  Volume: " << config.volumeDimX << "x" << config.volumeDimY << "x" << config.volumeDimZ << std::endl;
    std::cout << "  Total pages: " << totalPageCount_ << std::endl;
    std::cout << "  Max resident: " << config.maxResidentPages << std::endl;
    
    // Determine memory fit status
    analyzeVolumeMemoryFit();
    
    // Create GPU buffers
    
    // Extraction state
    VkDeviceSize stateSize = sizeof(ExtractionState);
    createBuffer(extractionStateBuffer_, device_, context_.getMemoryProperties(),
                 stateSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Memory state
    VkDeviceSize memStateSize = sizeof(MemoryState);
    createBuffer(memoryStateBuffer_, device_, context_.getMemoryProperties(),
                 memStateSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Page validity (1 bit per page, but using uint32_t for simplicity)
    VkDeviceSize validitySize = sizeof(uint32_t) * totalPageCount_;
    createBuffer(pageValidityBuffer_, device_, context_.getMemoryProperties(),
                 validitySize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Command queue
    VkDeviceSize cmdSize = sizeof(AutonomousCommand) * config.maxCommandsPerFrame;
    createBuffer(commandQueueBuffer_, device_, context_.getMemoryProperties(),
                 cmdSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Page residency
    VkDeviceSize residencySize = sizeof(uint32_t) * totalPageCount_;
    createBuffer(pageResidencyBuffer_, device_, context_.getMemoryProperties(),
                 residencySize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // View parameters
    VkDeviceSize viewSize = sizeof(glm::mat4) * 2 + sizeof(glm::vec4) * 8;
    createBuffer(viewParametersBuffer_, device_, context_.getMemoryProperties(),
                 viewSize,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Page priority
    VkDeviceSize prioritySize = sizeof(float) * totalPageCount_;
    createBuffer(pagePriorityBuffer_, device_, context_.getMemoryProperties(),
                 prioritySize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Initialize GPU state
    ExtractionState initialExtraction = {};
    initialExtraction.currentParameter = currentParameter_;
    initialExtraction.previousParameter = currentParameter_;
    initialExtraction.framesSinceChange = 0;
    initialExtraction.parameterChanged = 0;
    initialExtraction.totalPagesToExtract = 0;
    initialExtraction.pagesExtractedSoFar = 0;
    initialExtraction.extractionComplete = 1;
    
    MemoryState initialMemory = {};
    initialMemory.totalPages = totalPageCount_;
    initialMemory.maxResidentPages = config.maxResidentPages;
    initialMemory.currentResidentPages = 0;
    initialMemory.entireVolumeFits = volumeFitsInMemory_ ? 1 : 0;
    initialMemory.memoryPressure = 0;
    initialMemory.visiblePageCount = 0;
    
    // Upload initial states
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 std::max(stateSize, memStateSize),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    
    // Upload extraction state
    memcpy(stagingBuffer.data, &initialExtraction, sizeof(ExtractionState));
    VkBufferCopy copyRegion = {0, 0, sizeof(ExtractionState)};
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, extractionStateBuffer_.buffer, 1, &copyRegion);
    
    // Upload memory state
    memcpy(stagingBuffer.data, &initialMemory, sizeof(MemoryState));
    copyRegion.size = sizeof(MemoryState);
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, memoryStateBuffer_.buffer, 1, &copyRegion);
    
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    destroyBuffer(stagingBuffer, device_);
    
    // Create pipelines and descriptor sets
    createDescriptorSets();
    createPipelines();
    
    std::cout << "Autonomous DGC System initialized" << std::endl;
    std::cout << "  Volume " << (volumeFitsInMemory_ ? "FITS" : "DOES NOT FIT") << " in memory" << std::endl;
    if (!volumeFitsInMemory_) {
        std::cout << "  View-based prioritization enabled for large volume" << std::endl;
    }
}

void AutonomousDGCSystem::setExtractionParameter(float parameter) {
    if (std::abs(parameter - currentParameter_) < 0.0001f) {
        return; // No significant change
    }
    
    std::cout << "Extraction parameter changed: " << currentParameter_ << " → " << parameter << std::endl;
    currentParameter_ = parameter;
    
    // Update GPU state
    updateExtractionState(parameter);
}

void AutonomousDGCSystem::updateExtractionState(float parameter) {
    ExtractionState newState = {};
    newState.currentParameter = parameter;
    newState.previousParameter = currentParameter_;
    newState.framesSinceChange = 0;
    newState.parameterChanged = 1;
    newState.totalPagesToExtract = totalPageCount_;
    newState.pagesExtractedSoFar = 0;
    newState.extractionComplete = 0;
    
    // Upload to GPU
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 sizeof(ExtractionState),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    memcpy(stagingBuffer.data, &newState, sizeof(ExtractionState));
    
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(ExtractionState)};
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, extractionStateBuffer_.buffer, 1, &copyRegion);
    
    // Invalidate all page geometry
    vkCmdFillBuffer(cmd, pageValidityBuffer_.buffer, 0, VK_WHOLE_SIZE, 0);
    
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    destroyBuffer(stagingBuffer, device_);
    
    std::cout << "Marked all " << totalPageCount_ << " pages for re-extraction" << std::endl;
}

void AutonomousDGCSystem::updateViewParameters(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& position) {
    // Only affects extraction if volume doesn't fit in memory
    if (volumeFitsInMemory_) {
        return;
    }
    
    // Update view buffer
    struct ViewData {
        glm::mat4 viewMatrix;
        glm::mat4 projMatrix;
        glm::vec4 frustumPlanes[6];
    } viewData;
    
    viewData.viewMatrix = view;
    viewData.projMatrix = proj;
    
    // Calculate frustum planes from view-projection matrix
    glm::mat4 viewProj = proj * view;
    
    // Extract frustum planes
    // Left plane
    viewData.frustumPlanes[0] = glm::vec4(
        viewProj[0][3] + viewProj[0][0],
        viewProj[1][3] + viewProj[1][0],
        viewProj[2][3] + viewProj[2][0],
        viewProj[3][3] + viewProj[3][0]
    );
    
    // Right plane
    viewData.frustumPlanes[1] = glm::vec4(
        viewProj[0][3] - viewProj[0][0],
        viewProj[1][3] - viewProj[1][0],
        viewProj[2][3] - viewProj[2][0],
        viewProj[3][3] - viewProj[3][0]
    );
    
    // Bottom plane
    viewData.frustumPlanes[2] = glm::vec4(
        viewProj[0][3] + viewProj[0][1],
        viewProj[1][3] + viewProj[1][1],
        viewProj[2][3] + viewProj[2][1],
        viewProj[3][3] + viewProj[3][1]
    );
    
    // Top plane
    viewData.frustumPlanes[3] = glm::vec4(
        viewProj[0][3] - viewProj[0][1],
        viewProj[1][3] - viewProj[1][1],
        viewProj[2][3] - viewProj[2][1],
        viewProj[3][3] - viewProj[3][1]
    );
    
    // Near plane
    viewData.frustumPlanes[4] = glm::vec4(
        viewProj[0][3] + viewProj[0][2],
        viewProj[1][3] + viewProj[1][2],
        viewProj[2][3] + viewProj[2][2],
        viewProj[3][3] + viewProj[3][2]
    );
    
    // Far plane
    viewData.frustumPlanes[5] = glm::vec4(
        viewProj[0][3] - viewProj[0][2],
        viewProj[1][3] - viewProj[1][2],
        viewProj[2][3] - viewProj[2][2],
        viewProj[3][3] - viewProj[3][2]
    );
    
    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float length = glm::length(glm::vec3(viewData.frustumPlanes[i]));
        if (length > 0.0f) {
            viewData.frustumPlanes[i] /= length;
        }
    }
    
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 sizeof(ViewData),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    memcpy(stagingBuffer.data, &viewData, sizeof(ViewData));
    
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(ViewData)};
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, viewParametersBuffer_.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    destroyBuffer(stagingBuffer, device_);
}

void AutonomousDGCSystem::executeAutonomousFrame(VkCommandBuffer cmd, uint32_t frameIndex) {
    recordAutonomousCommands(cmd, frameIndex);
}

void AutonomousDGCSystem::recordAutonomousCommands(VkCommandBuffer cmd, uint32_t frameIndex) {
    // Clear command queue
    vkCmdFillBuffer(cmd, commandQueueBuffer_.buffer, 0, sizeof(uint32_t), 0);
    
    VkMemoryBarrier2 clearBarrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    };
    VkDependencyInfo clearDependencyInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &clearBarrier
    };
    vkCmdPipelineBarrier2(cmd, &clearDependencyInfo);
    
    // Prepare push constants for all passes
    struct PushConstants {
        uint32_t volumeDimX;
        uint32_t volumeDimY;
        uint32_t volumeDimZ;
        uint32_t pageSizeX;
        uint32_t pageSizeY;
        uint32_t pageSizeZ;
        uint32_t maxCommandsPerFrame;
        uint32_t currentPass;
    } pushConstants = {
        config_.volumeDimX,
        config_.volumeDimY,
        config_.volumeDimZ,
        config_.pageSizeX,
        config_.pageSizeY,
        config_.pageSizeZ,
        config_.maxCommandsPerFrame,
        0  // Will be updated per pass
    };
    
    // Step 1: Analyze extraction state
    pushConstants.currentPass = 0;  // PASS_STATE_ANALYSIS
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, stateAnalysisPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, (totalPageCount_ + 63) / 64, 1, 1);
    
    VkMemoryBarrier2 syncBarrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    };
    VkDependencyInfo syncDependencyInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &syncBarrier
    };
    vkCmdPipelineBarrier2(cmd, &syncDependencyInfo);
    
    // Step 2: Analyze memory state
    pushConstants.currentPass = 1;  // PASS_MEMORY_ANALYSIS
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, memoryAnalysisPipeline_);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, 1, 1, 1);
    
    vkCmdPipelineBarrier2(cmd, &syncDependencyInfo);
    
    // Step 3: Prioritize pages
    pushConstants.currentPass = 2;  // PASS_PRIORITIZATION
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prioritizationPipeline_);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, (totalPageCount_ + 63) / 64, 1, 1);
    
    vkCmdPipelineBarrier2(cmd, &syncDependencyInfo);
    
    // Step 4: Generate commands
    pushConstants.currentPass = 3;  // PASS_COMMAND_GENERATION
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, commandGenerationPipeline_);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, 1, 1, 1);
    
    // Final barrier
    VkMemoryBarrier2 commandBarrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
    };
    VkDependencyInfo commandDependencyInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &commandBarrier
    };
    vkCmdPipelineBarrier2(cmd, &commandDependencyInfo);
}

void AutonomousDGCSystem::analyzeVolumeMemoryFit() {
    volumeFitsInMemory_ = (totalPageCount_ <= config_.maxResidentPages);
    
    if (volumeFitsInMemory_) {
        std::cout << "Volume fits entirely in memory (" << totalPageCount_ 
                  << " pages <= " << config_.maxResidentPages << " max)" << std::endl;
        std::cout << "View-based culling DISABLED for extraction" << std::endl;
    } else {
        std::cout << "Volume exceeds memory (" << totalPageCount_ 
                  << " pages > " << config_.maxResidentPages << " max)" << std::endl;
        std::cout << "View-based prioritization ENABLED" << std::endl;
    }
}

void AutonomousDGCSystem::createDescriptorSets() {
    // Create descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };
    
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorLayout_));
    
    // Create pipeline layout
    VkPushConstantRange pushRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t) * 8  // 8 uint32_t values: volumeDimX/Y/Z, pageSizeX/Y/Z, maxCommandsPerFrame, currentPass
    };
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange
    };
    
    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));
    
    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
    };
    
    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
    
    // Allocate and update descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorLayout_
    };
    
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_));
    
    // Update descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    
    VkDescriptorBufferInfo buffers[] = {
        {extractionStateBuffer_.buffer, 0, VK_WHOLE_SIZE},
        {memoryStateBuffer_.buffer, 0, VK_WHOLE_SIZE},
        {pageValidityBuffer_.buffer, 0, VK_WHOLE_SIZE},
        {commandQueueBuffer_.buffer, 0, VK_WHOLE_SIZE},
        {pageResidencyBuffer_.buffer, 0, VK_WHOLE_SIZE},
        {viewParametersBuffer_.buffer, 0, VK_WHOLE_SIZE},
        {pagePriorityBuffer_.buffer, 0, VK_WHOLE_SIZE}
    };
    
    for (uint32_t i = 0; i < 7; i++) {
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet_,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = (i == 5) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffers[i]
        });
    }
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void AutonomousDGCSystem::createPipelines() {
    // Load shader for autonomous extraction selection
    Shader autonomousExtractionShader_ {};
    assert(loadShader(autonomousExtractionShader_, device_, "/spirv/autonomousExtractionSelection.comp.spv"));
    
    VkPipelineShaderStageCreateInfo shaderStage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = autonomousExtractionShader_.module,
        .pName = "main"
    };
    
    VkComputePipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shaderStage,
        .layout = pipelineLayout_
    };
    
    // Create state analysis pipeline (Pass 0)
    VkSpecializationMapEntry specEntry = {
        .constantID = 0,
        .offset = 0,
        .size = sizeof(uint32_t)
    };
    
    uint32_t passType = 0; // PASS_STATE_ANALYSIS
    VkSpecializationInfo specInfo = {
        .mapEntryCount = 1,
        .pMapEntries = &specEntry,
        .dataSize = sizeof(uint32_t),
        .pData = &passType
    };
    
    shaderStage.pSpecializationInfo = &specInfo;
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &stateAnalysisPipeline_));
    
    // Create memory analysis pipeline (Pass 1)
    passType = 1; // PASS_MEMORY_ANALYSIS
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &memoryAnalysisPipeline_));
    
    // Create prioritization pipeline (Pass 2)
    passType = 2; // PASS_PRIORITIZATION
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &prioritizationPipeline_));
    
    // Create command generation pipeline (Pass 3)
    passType = 3; // PASS_COMMAND_GENERATION
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &commandGenerationPipeline_));
    
    // Cleanup shader module
    vkDestroyShaderModule(device_, autonomousExtractionShader_.module, nullptr);
    
    std::cout << "Autonomous DGC pipelines created successfully" << std::endl;
}

bool AutonomousDGCSystem::isExtractionComplete() const {
    // Read extraction state from GPU
    ExtractionState extractionState = {};
    
    // Create host-visible staging buffer for readback
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 sizeof(ExtractionState),
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy from device to staging
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(ExtractionState)};
    vkCmdCopyBuffer(cmd, extractionStateBuffer_.buffer, stagingBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    // Read back the data
    memcpy(&extractionState, stagingBuffer.data, sizeof(ExtractionState));
    
    // Cleanup
    destroyBuffer(stagingBuffer, device_);
    
    return extractionState.extractionComplete != 0;
}

float AutonomousDGCSystem::getExtractionProgress() const {
    // Read extraction state from GPU
    ExtractionState extractionState = {};
    
    // Create host-visible staging buffer for readback
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 sizeof(ExtractionState),
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy from device to staging
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(ExtractionState)};
    vkCmdCopyBuffer(cmd, extractionStateBuffer_.buffer, stagingBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    // Read back the data
    memcpy(&extractionState, stagingBuffer.data, sizeof(ExtractionState));
    
    // Cleanup
    destroyBuffer(stagingBuffer, device_);
    
    // Calculate progress as ratio of processed pages to total pages
    if (extractionState.totalPagesToExtract == 0) return 0.0f;
    return static_cast<float>(extractionState.pagesExtractedSoFar) / static_cast<float>(extractionState.totalPagesToExtract);
}

bool AutonomousDGCSystem::entireVolumeFitsInMemory() const {
    return volumeFitsInMemory_;
}

uint32_t AutonomousDGCSystem::getResidentPageCount() const {
    // Read memory state from GPU
    MemoryState memoryState = {};
    
    // Create host-visible staging buffer for readback
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 sizeof(MemoryState),
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy from device to staging
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(MemoryState)};
    vkCmdCopyBuffer(cmd, memoryStateBuffer_.buffer, stagingBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    // Read back the data
    memcpy(&memoryState, stagingBuffer.data, sizeof(MemoryState));
    
    // Cleanup
    destroyBuffer(stagingBuffer, device_);
    
    return memoryState.currentResidentPages;
}

void AutonomousDGCSystem::updateExtractionProgress(uint32_t pagesExtracted) {
    // Read current state
    ExtractionState currentState = {};
    
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 sizeof(ExtractionState),
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Read current state from GPU
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(ExtractionState)};
    vkCmdCopyBuffer(cmd, extractionStateBuffer_.buffer, stagingBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    memcpy(&currentState, stagingBuffer.data, sizeof(ExtractionState));
    
    // Update the state
    currentState.pagesExtractedSoFar += pagesExtracted;
    currentState.framesSinceChange++;
    
    // Check if extraction is complete
    if (currentState.pagesExtractedSoFar >= currentState.totalPagesToExtract) {
        currentState.extractionComplete = 1;
        std::cout << "Extraction complete! Processed " << currentState.pagesExtractedSoFar 
                  << " pages." << std::endl;
    }
    
    // Write updated state back to GPU
    memcpy(stagingBuffer.data, &currentState, sizeof(ExtractionState));
    
    cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, extractionStateBuffer_.buffer, 1, &copyRegion);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    destroyBuffer(stagingBuffer, device_);
}