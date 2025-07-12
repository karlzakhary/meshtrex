#pragma once

#include "common.h"
#include "vulkan_utils.h"
#include "device.h"
#include "vulkan_context.h"
#include "buffer.h"
#include "streamingShaderInterface.h"
#include <vector>

struct DGCToken {
    VkIndirectCommandsTokenTypeNV type;
    uint32_t offset;
    uint32_t stride;
};

struct DGCStreamInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    uint32_t stride;
};

struct IndirectDispatchCommand {
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
    uint32_t passType;
    uint32_t pageCoordX;
    uint32_t pageCoordY;
    uint32_t pageCoordZ;
    uint32_t mipLevel;
};

enum PassType {
    PASS_MIN_MAX_LEAF = 0,
    PASS_MIN_MAX_REDUCE = 1,
    PASS_ACTIVE_BLOCK_FILTER = 2,
    PASS_MESH_EXTRACTION = 3
};

class DeviceGeneratedCommands {
public:
    DeviceGeneratedCommands(VulkanContext& context);
    ~DeviceGeneratedCommands();
    
    void initialize();
    bool isSupported() const { return dgcSupport.dgcSupported; }
    
    void createIndirectCommandsLayout(uint32_t pushConstantsSize);
    void createPreprocessBuffer(uint32_t maxCommands);
    void createSequencesCountBuffer();
    
    void recordPreprocessCommands(VkCommandBuffer cmd, uint32_t maxSequenceCount);
    void recordExecuteCommands(VkCommandBuffer cmd, VkPipeline pipeline, uint32_t maxSequenceCount);
    
    void updateCommandStreams(const std::vector<IndirectDispatchCommand>& commands);
    
    VkBuffer getCommandBuffer() const { return commandBuffer.buffer; }
    VkBuffer getSequencesCountBuffer() const { return sequencesCountBuffer.buffer; }
    
private:
    VulkanContext& context;
    VkDevice device;
    
    DGCSupport dgcSupport;
    
    VkIndirectCommandsLayoutNV indirectCommandsLayout;
    
    Buffer commandBuffer;
    Buffer preprocessBuffer;
    Buffer sequencesCountBuffer;
    
    std::vector<VkIndirectCommandsStreamNV> commandStreams;
    std::vector<DGCToken> tokens;
    
    void createCommandBuffer(uint32_t maxCommands, uint32_t commandSize);
};

class StreamingDGCManager {
public:
    StreamingDGCManager(VulkanContext& context);
    ~StreamingDGCManager();
    
    void initialize();
    void beginFrame();
    void endFrame();
    
    void generateGPUCommands(VkCommandBuffer cmd, VkDescriptorSet streamingDescriptors, uint32_t passType);
    void executeGPUGeneratedCommands(VkCommandBuffer cmd, VkPipeline targetPipeline, VkPipelineLayout targetPipelineLayout, VkDescriptorSet streamingDescriptors);
    
    bool isSupported() const { return dgc.isSupported(); }
    
private:
    VulkanContext& context;
    DeviceGeneratedCommands dgc;
    
    VkPipeline commandGenerationPipeline;
    VkPipelineLayout commandGenerationPipelineLayout;
    VkDescriptorSetLayout commandGenerationDescriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet commandGenerationDescriptorSet;
    
    Buffer gpuCommandBuffer;
    Buffer streamingConstantsBuffer;
    
    void createCommandGenerationPipeline();
    void createCommandGenerationDescriptors();
    void updateStreamingConstants(uint32_t passType, uint32_t frameIndex, 
                                  uint32_t volumeWidth, uint32_t volumeHeight, uint32_t volumeDepth,
                                  float isoValue = 128.0f);
};