#include "common.h"
#include "device.h"
#include "buffer.h"
#include "deviceGeneratedCommands.h"
#include "vulkan_utils.h" // For beginSingleTimeCommands/endSingleTimeCommands
#include <iostream>
#include <cstring>

DeviceGeneratedCommands::DeviceGeneratedCommands(VulkanContext& context) 
    : context(context), device(context.getDevice()) {
    dgcSupport = queryDGCSupport(context.getPhysicalDevice());
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
    if (!dgcSupport.dgcSupported) {
        std::cout << "Device Generated Commands not supported!" << std::endl;
        return;
    }
    
    //createIndirectCommandsLayout(); //TODO
    createPreprocessBuffer(1024);
    createSequencesCountBuffer();
    //createCommandBuffer(1024); //TODO
}


void DeviceGeneratedCommands::createIndirectCommandsLayout(uint32_t pushConstantsSize) {
    if (!dgcSupport.dgcComputeSupported) {
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
    pushConstantToken.pushconstantSize = pushConstantsSize;
    
    layoutTokens.push_back(pushConstantToken);
    
    // Define stream stride (size of all data for one command)
    uint32_t streamStride = sizeof(VkDispatchIndirectCommand) + pushConstantsSize;
    
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

void DeviceGeneratedCommands::createCommandBuffer(uint32_t maxCommands, uint32_t commandSize) {
    createBuffer(commandBuffer, device, context.getMemoryProperties(),
                 commandSize * maxCommands,
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void DeviceGeneratedCommands::recordPreprocessCommands(VkCommandBuffer cmd, uint32_t maxSequenceCount) {
    if (!dgcSupport.dgcSupported) return;
    
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
    if (!dgcSupport.dgcSupported) return;
    
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
    };
    
    std::vector<CommandData> commandData;
    commandData.reserve(commands.size());
    
    for (const auto& cmd : commands) {
        CommandData data;
        data.dispatch.x = cmd.groupCountX;
        data.dispatch.y = cmd.groupCountY;
        data.dispatch.z = cmd.groupCountZ;
        
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