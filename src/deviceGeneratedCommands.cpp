#include "common.h"
#include "streamingSystem.h"
#include "buffer.h"
#include "deviceGeneratedCommands.h"
#include "vulkan_utils.h" // For beginSingleTimeCommands/endSingleTimeCommands
#include "streamingShaderInterface.h" // For StreamingConstants
#include <iostream>
#include <algorithm>
#include <cstring>

DeviceGeneratedCommands::DeviceGeneratedCommands(VulkanContext& context) 
    : context(context), device(context.getDevice()) {
    queryDGCSupport();
}

DeviceGeneratedCommands::~DeviceGeneratedCommands() {
    if (indirectCommandsLayout != VK_NULL_HANDLE) {
        vkDestroyIndirectCommandsLayoutNV(device, indirectCommandsLayout, nullptr);
    }
    
    if (commandBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(commandBuffer, context.getDevice());
    }
    if (preprocessBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(preprocessBuffer, context.getDevice());
    }
    if (sequencesCountBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(sequencesCountBuffer, context.getDevice());
    }
}

void DeviceGeneratedCommands::initialize() {
    if (!dgcSupported) {
        std::cout << "Device Generated Commands not supported!" << std::endl;
        return;
    }
    
    createIndirectCommandsLayout();
    createPreprocessBuffer(1024);
    createSequencesCountBuffer();
    createCommandBuffer(1024);
}

void DeviceGeneratedCommands::queryDGCSupport() {
    // Check if the extension is available
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(context.getPhysicalDevice(), nullptr, &extensionCount, nullptr);
    
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(context.getPhysicalDevice(), nullptr, &extensionCount, extensions.data());
    
    for (const auto& extension : extensions) {
        if (strcmp(extension.extensionName, VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME) == 0) {
            dgcSupported = true;
        }
        if (strcmp(extension.extensionName, VK_NV_DEVICE_GENERATED_COMMANDS_COMPUTE_EXTENSION_NAME) == 0) {
            dgcComputeSupported = true;
        }
    }
    
    if (!dgcSupported) {
        std::cout << "VK_NV_device_generated_commands extension not available" << std::endl;
        return;
    }
    
    // Query features and properties
    dgcFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV;
    
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &dgcFeatures;
    
    vkGetPhysicalDeviceFeatures2(context.getPhysicalDevice(), &features2);
    
    dgcProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_NV;
    
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &dgcProperties;
    
    vkGetPhysicalDeviceProperties2(context.getPhysicalDevice(), &properties2);
    
    std::cout << "DGC Support: " << dgcFeatures.deviceGeneratedCommands << std::endl;
    std::cout << "DGC Compute Support: " << dgcComputeSupported << std::endl;
    std::cout << "DGC Max Sequences: " << dgcProperties.maxIndirectSequenceCount << std::endl;
    // std::cout << "DGC Max Commands per Sequence: " << dgcProperties.maxIndirectCommandsPerTokenSequence << std::endl;
}

void DeviceGeneratedCommands::createIndirectCommandsLayout() {
    if (!dgcComputeSupported) {
        std::cerr << "ERROR: Cannot create dispatch commands layout without VK_NV_device_generated_commands_compute extension!" << std::endl;
        throw std::runtime_error("VK_NV_device_generated_commands_compute extension is required for dispatch commands");
    }
    
    // Define tokens for our command stream
    std::vector<VkIndirectCommandsLayoutTokenNV> layoutTokens;
    
    // Token 0: Dispatch command
    VkIndirectCommandsLayoutTokenNV dispatchToken = {};
    dispatchToken.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_NV;
    dispatchToken.tokenType = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_NV;
    dispatchToken.stream = 0;
    dispatchToken.offset = 0;
    
    layoutTokens.push_back(dispatchToken);
    
    // Token 1: Push constants for page coordinate
    VkIndirectCommandsLayoutTokenNV pushConstantToken = {};
    pushConstantToken.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_NV;
    pushConstantToken.tokenType = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_NV;
    pushConstantToken.stream = 0;
    pushConstantToken.offset = sizeof(VkDispatchIndirectCommand);
    pushConstantToken.pushconstantPipelineLayout = VK_NULL_HANDLE; // Will be set during execution
    pushConstantToken.pushconstantShaderStageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantToken.pushconstantOffset = 0;
    pushConstantToken.pushconstantSize = sizeof(PageCoord);
    
    layoutTokens.push_back(pushConstantToken);
    
    // Define stream stride (size of all data for one command)
    uint32_t streamStride = sizeof(VkDispatchIndirectCommand) + sizeof(PageCoord);
    
    // Create the layout
    VkIndirectCommandsLayoutCreateInfoNV createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_NV;
    createInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    createInfo.flags = 0;
    createInfo.tokenCount = static_cast<uint32_t>(layoutTokens.size());
    createInfo.pTokens = layoutTokens.data();
    createInfo.streamCount = 1;
    createInfo.pStreamStrides = &streamStride;
    
    VK_CHECK(vkCreateIndirectCommandsLayoutNV(device, &createInfo, nullptr, &indirectCommandsLayout));
}

void DeviceGeneratedCommands::createPreprocessBuffer(uint32_t maxCommands) {
    VkGeneratedCommandsMemoryRequirementsInfoNV memReqInfo = {};
    memReqInfo.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_NV;
    memReqInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    memReqInfo.pipeline = VK_NULL_HANDLE;
    memReqInfo.indirectCommandsLayout = indirectCommandsLayout;
    memReqInfo.maxSequencesCount = maxCommands;
    
    VkMemoryRequirements2 memReq = {};
    memReq.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    
    vkGetGeneratedCommandsMemoryRequirementsNV(device, &memReqInfo, &memReq);
    
    createBuffer(preprocessBuffer, device, context.getMemoryProperties(),
                 memReq.memoryRequirements.size,
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void DeviceGeneratedCommands::createSequencesCountBuffer() {
    createBuffer(sequencesCountBuffer, device, context.getMemoryProperties(),
                 sizeof(uint32_t),
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void DeviceGeneratedCommands::createCommandBuffer(uint32_t maxCommands) {
    uint32_t commandSize = sizeof(VkDispatchIndirectCommand) + sizeof(PageCoord);
    
    createBuffer(commandBuffer, device, context.getMemoryProperties(),
                 commandSize * maxCommands,
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void DeviceGeneratedCommands::recordPreprocessCommands(VkCommandBuffer cmd, uint32_t maxSequenceCount) {
    if (!dgcSupported) return;
    
    uint32_t streamStride = sizeof(VkDispatchIndirectCommand) + sizeof(PageCoord);
    
    VkGeneratedCommandsInfoNV generatedCommandsInfo = {};
    generatedCommandsInfo.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_NV;
    generatedCommandsInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    generatedCommandsInfo.pipeline = VK_NULL_HANDLE;
    generatedCommandsInfo.indirectCommandsLayout = indirectCommandsLayout;
    generatedCommandsInfo.streamCount = 1;
    generatedCommandsInfo.pStreams = commandStreams.data();
    generatedCommandsInfo.sequencesCount = maxSequenceCount;
    generatedCommandsInfo.preprocessBuffer = preprocessBuffer.buffer;
    generatedCommandsInfo.preprocessOffset = 0;
    generatedCommandsInfo.preprocessSize = preprocessBuffer.size;
    generatedCommandsInfo.sequencesCountBuffer = sequencesCountBuffer.buffer;
    generatedCommandsInfo.sequencesCountOffset = 0;
    generatedCommandsInfo.sequencesIndexBuffer = VK_NULL_HANDLE;
    generatedCommandsInfo.sequencesIndexOffset = 0;
    
    vkCmdPreprocessGeneratedCommandsNV(cmd, &generatedCommandsInfo);
}

void DeviceGeneratedCommands::recordExecuteCommands(VkCommandBuffer cmd, VkPipeline pipeline, uint32_t maxSequenceCount) {
    if (!dgcSupported) return;
    
    uint32_t streamStride = sizeof(VkDispatchIndirectCommand) + sizeof(PageCoord);
    
    VkIndirectCommandsStreamNV stream = {};
    stream.buffer = commandBuffer.buffer;
    stream.offset = 0;
    
    commandStreams.clear();
    commandStreams.push_back(stream);
    
    VkGeneratedCommandsInfoNV generatedCommandsInfo = {};
    generatedCommandsInfo.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_NV;
    generatedCommandsInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    generatedCommandsInfo.pipeline = pipeline;
    generatedCommandsInfo.indirectCommandsLayout = indirectCommandsLayout;
    generatedCommandsInfo.streamCount = 1;
    generatedCommandsInfo.pStreams = commandStreams.data();
    generatedCommandsInfo.sequencesCount = maxSequenceCount;
    generatedCommandsInfo.preprocessBuffer = preprocessBuffer.buffer;
    generatedCommandsInfo.preprocessOffset = 0;
    generatedCommandsInfo.preprocessSize = preprocessBuffer.size;
    generatedCommandsInfo.sequencesCountBuffer = sequencesCountBuffer.buffer;
    generatedCommandsInfo.sequencesCountOffset = 0;
    generatedCommandsInfo.sequencesIndexBuffer = VK_NULL_HANDLE;
    generatedCommandsInfo.sequencesIndexOffset = 0;
    
    vkCmdExecuteGeneratedCommandsNV(cmd, VK_FALSE, &generatedCommandsInfo);
}

void DeviceGeneratedCommands::updateCommandStreams(const std::vector<IndirectDispatchCommand>& commands) {
    if (commands.empty()) return;
    
    struct CommandData {
        VkDispatchIndirectCommand dispatch;
        PageCoord pageCoord;
    };
    
    std::vector<CommandData> commandData;
    commandData.reserve(commands.size());
    
    for (const auto& cmd : commands) {
        CommandData data;
        data.dispatch.x = cmd.groupCountX;
        data.dispatch.y = cmd.groupCountY;
        data.dispatch.z = cmd.groupCountZ;
        data.pageCoord.x = cmd.pageCoordX;
        data.pageCoord.y = cmd.pageCoordY;
        data.pageCoord.z = cmd.pageCoordZ;
        data.pageCoord.mipLevel = cmd.mipLevel;
        
        commandData.push_back(data);
    }
    
    // Update command buffer using staging
    if (commandData.size() > 0) {
        VkDeviceSize dataSize = commandData.size() * sizeof(CommandData);
        
        Buffer stagingBuffer = {};
        createBuffer(stagingBuffer, device, context.getMemoryProperties(),
                     dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        memcpy(stagingBuffer.data, commandData.data(), dataSize);
        
        VkCommandBuffer cmd = beginSingleTimeCommands(device, context.getCommandPool());
        VkBufferCopy copyRegion = {0, 0, dataSize};
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, commandBuffer.buffer, 1, &copyRegion);
        endSingleTimeCommands(device, context.getCommandPool(), context.getQueue(), cmd);
        
        destroyBuffer(stagingBuffer, device);
    }
    
    // Update sequence count
    uint32_t sequenceCount = static_cast<uint32_t>(commands.size());
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device, context.getMemoryProperties(),
                 sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    memcpy(stagingBuffer.data, &sequenceCount, sizeof(uint32_t));
    
    VkCommandBuffer cmd = beginSingleTimeCommands(device, context.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(uint32_t)};
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, sequencesCountBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(device, context.getCommandPool(), context.getQueue(), cmd);
    
    destroyBuffer(stagingBuffer, device);
}

StreamingDGCManager::StreamingDGCManager(VulkanContext& context) 
    : context(context), dgc(context), 
      commandGenerationPipeline(VK_NULL_HANDLE),
      commandGenerationPipelineLayout(VK_NULL_HANDLE),
      commandGenerationDescriptorSetLayout(VK_NULL_HANDLE),
      descriptorPool(VK_NULL_HANDLE),
      commandGenerationDescriptorSet(VK_NULL_HANDLE) {
}

StreamingDGCManager::~StreamingDGCManager() {
    VkDevice device = context.getDevice();
    
    if (commandGenerationPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, commandGenerationPipeline, nullptr);
    }
    if (commandGenerationPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, commandGenerationPipelineLayout, nullptr);
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }
    if (commandGenerationDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, commandGenerationDescriptorSetLayout, nullptr);
    }
    
    if (gpuCommandBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(gpuCommandBuffer, context.getDevice());
    }
    if (streamingConstantsBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(streamingConstantsBuffer, context.getDevice());
    }
}

void StreamingDGCManager::initialize() {
    dgc.initialize();
    createCommandGenerationPipeline();
    createCommandGenerationDescriptors();
}

void StreamingDGCManager::beginFrame() {
    // Reset GPU command buffer counter
    uint32_t zero = 0;
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    memcpy(stagingBuffer.data, &zero, sizeof(uint32_t));
    
    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, sizeof(uint32_t)};
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, gpuCommandBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    
    destroyBuffer(stagingBuffer, context.getDevice());
}

void StreamingDGCManager::endFrame() {
}

void StreamingDGCManager::generateGPUCommands(VkCommandBuffer cmd, VkDescriptorSet streamingDescriptors, uint32_t passType) {
    // Note: Volume dimensions and isovalue should be provided by the caller
    // For now, using placeholder values - this should be updated when integrated
    updateStreamingConstants(passType, 0, 1024, 1024, 1024, 128.0f);
    
    // Bind command generation pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, commandGenerationPipeline);
    
    // Bind descriptor sets (streaming descriptors + command generation descriptors)
    VkDescriptorSet descriptorSets[] = {streamingDescriptors, commandGenerationDescriptorSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, commandGenerationPipelineLayout,
                           0, 2, descriptorSets, 0, nullptr);
    
    // Calculate dispatch size - one thread per potential page across all mip levels
    uint32_t totalPages = 0;
    
    // Read streaming constants from buffer to get volume dimensions
    StreamingConstants* constants = reinterpret_cast<StreamingConstants*>(streamingConstantsBuffer.data);
    
    // Base page sizes from streaming constants
    uint32_t pageSizeX = constants->pageSizeX;
    uint32_t pageSizeY = constants->pageSizeY;
    uint32_t pageSizeZ = constants->pageSizeZ;
    uint32_t volumeWidth = constants->volumeWidth;
    uint32_t volumeHeight = constants->volumeHeight;
    uint32_t volumeDepth = constants->volumeDepth;
    
    for (uint32_t mip = 0; mip <= 3; mip++) { // maxMipLevel = 3
        // For simplicity with non-cubic pages, use the smallest dimension for mip calculation
        uint32_t minPageSize = std::min({pageSizeX, pageSizeY, pageSizeZ});
        uint32_t mipPageSize = minPageSize >> mip;
        uint32_t pagesX = (volumeWidth + mipPageSize - 1) / mipPageSize;
        uint32_t pagesY = (volumeHeight + mipPageSize - 1) / mipPageSize;
        uint32_t pagesZ = (volumeDepth + mipPageSize - 1) / mipPageSize;
        totalPages += pagesX * pagesY * pagesZ;
    }
    
    uint32_t workGroupSize = 64;
    uint32_t dispatchGroups = (totalPages + workGroupSize - 1) / workGroupSize;
    
    // GPU generates commands based on page residency
    vkCmdDispatch(cmd, dispatchGroups, 1, 1);
    
    // Memory barrier - GPU command generation -> DGC execution
    VkMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    
    VkDependencyInfo dependencyInfo = {};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &barrier;
    
    vkCmdPipelineBarrier2(cmd, &dependencyInfo);
}

void StreamingDGCManager::executeGPUGeneratedCommands(VkCommandBuffer cmd, VkPipeline targetPipeline, VkPipelineLayout targetPipelineLayout, VkDescriptorSet streamingDescriptors) {
    if (!dgc.isSupported()) {
        return;
    }
    
    // Use DGC to execute GPU-generated commands
    // The GPU-generated command buffer is used directly by DGC
    dgc.recordPreprocessCommands(cmd, 1024); // Max commands
    
    VkMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    
    VkDependencyInfo dependencyInfo = {};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &barrier;
    
    vkCmdPipelineBarrier2(cmd, &dependencyInfo);
    
    // Bind target pipeline and execute GPU-generated commands
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipelineLayout,
                           0, 1, &streamingDescriptors, 0, nullptr);
    
    dgc.recordExecuteCommands(cmd, targetPipeline, 1024);
}

void StreamingDGCManager::createCommandGenerationPipeline() {
    // Create descriptor set layout for command generation
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    
    // Binding 0: Streaming constants
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    
    // Binding 1: Page table
    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    
    // Binding 2: GPU command buffer (output)
    bindings[2] = {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    VK_CHECK(vkCreateDescriptorSetLayout(context.getDevice(), &layoutInfo, nullptr, &commandGenerationDescriptorSetLayout));
    
    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &commandGenerationDescriptorSetLayout;
    
    VK_CHECK(vkCreatePipelineLayout(context.getDevice(), &pipelineLayoutInfo, nullptr, &commandGenerationPipelineLayout));
    
    // Create compute pipeline
    // commandGenerationPipeline = createComputePipeline(context, "shaders/gpuCommandGeneration.comp.spv", commandGenerationPipelineLayout);
}

void StreamingDGCManager::createCommandGenerationDescriptors() {
    VkDevice device = context.getDevice();
    
    // Create descriptor set layout for command generation
    VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }, // Page table
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }, // Streaming constants
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }, // GPU command buffer
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }  // Work queue header
    };
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings
    };
    
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &commandGenerationDescriptorSetLayout));
    
    // Create descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}
    };
    
    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = poolSizes
    };
    
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &commandGenerationDescriptorSetLayout
    };
    
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &commandGenerationDescriptorSet));
    
    // Create GPU command buffer for storing generated commands
    VkDeviceSize commandBufferSize = sizeof(IndirectDispatchCommand) * 1024; // Max 1024 commands
    createBuffer(gpuCommandBuffer, device, context.getMemoryProperties(),
                 commandBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Create streaming constants buffer
    VkDeviceSize constantsBufferSize = sizeof(StreamingConstants);
    createBuffer(streamingConstantsBuffer, device, context.getMemoryProperties(),
                 constantsBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void StreamingDGCManager::updateStreamingConstants(uint32_t passType, uint32_t frameIndex, 
                                                  uint32_t volumeWidth, uint32_t volumeHeight, uint32_t volumeDepth,
                                                  float isoValue) {
    StreamingConstants constants = {};
    constants.pageSizeX = 64;
    constants.pageSizeY = 32;
    constants.pageSizeZ = 32;
    constants.atlasSizeX = 1024;
    constants.atlasSizeY = 1024;
    constants.atlasSizeZ = 1024;
    constants.volumeWidth = volumeWidth;
    constants.volumeHeight = volumeHeight;
    constants.volumeDepth = volumeDepth;
    constants.maxMipLevel = 3;
    constants.isoValue = isoValue;
    constants.frameIndex = frameIndex;
    
    // Update streaming constants buffer
    if (streamingConstantsBuffer.data) {
        // If host visible, update directly
        memcpy(streamingConstantsBuffer.data, &constants, sizeof(constants));
    } else {
        // Use staging buffer for device local memory
        Buffer stagingBuffer = {};
        createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                     sizeof(StreamingConstants), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        memcpy(stagingBuffer.data, &constants, sizeof(StreamingConstants));
        
        VkCommandBuffer cmdBuf = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
        VkBufferCopy copyRegion = {0, 0, sizeof(StreamingConstants)};
        vkCmdCopyBuffer(cmdBuf, stagingBuffer.buffer, streamingConstantsBuffer.buffer, 1, &copyRegion);
        endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmdBuf);
        
        destroyBuffer(stagingBuffer, context.getDevice());
    }
}