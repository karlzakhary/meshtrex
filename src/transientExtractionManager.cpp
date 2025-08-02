#include "transientExtractionManager.h"
#include "common.h"
#include "vulkan_context.h"
#include "transientExtractionPipeline.h"
#include "buffer.h"
#include "image.h"
#include "resources.h"
#include "gpuProfiler.h"
#include "vulkan_utils.h"
#include "mc_tables.h"
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <cstring>

// Static resources shared across calls (defined at file scope)
static struct {
    TransientExtractionPipeline pipeline;
    Buffer constantsUBO;
    Buffer mcTriTableBuffer;
    Buffer mcEdgeTableBuffer;
    VkSampler minMaxSampler = VK_NULL_HANDLE;
    bool initialized = false;
} g_transientResources;

// Helper to create UBO
static Buffer createConstantsUBO(VulkanContext& context, PushConstants& pushConstants) {
    Buffer constantsUBO = {};
    VkDeviceSize bufferSize = sizeof(PushConstants);

    createBuffer(constantsUBO, context.getDevice(), context.getMemoryProperties(),
                 bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (constantsUBO.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to create constants UBO buffer.");
    }
    if (constantsUBO.data == nullptr) {
         destroyBuffer(constantsUBO, context.getDevice());
         throw std::runtime_error("Failed to map constants UBO buffer.");
    }

    // Copy push constants to UBO
    memcpy(constantsUBO.data, &pushConstants, sizeof(PushConstants));

    return constantsUBO;
}

// Extract frustum planes from view-projection matrix
void extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4 frustumPlanes[6]) {
    // Extract frustum planes from view-projection matrix
    // Left plane
    frustumPlanes[0] = glm::vec4(
        viewProj[0][3] + viewProj[0][0],
        viewProj[1][3] + viewProj[1][0],
        viewProj[2][3] + viewProj[2][0],
        viewProj[3][3] + viewProj[3][0]
    );
    
    // Right plane
    frustumPlanes[1] = glm::vec4(
        viewProj[0][3] - viewProj[0][0],
        viewProj[1][3] - viewProj[1][0],
        viewProj[2][3] - viewProj[2][0],
        viewProj[3][3] - viewProj[3][0]
    );
    
    // Bottom plane
    frustumPlanes[2] = glm::vec4(
        viewProj[0][3] + viewProj[0][1],
        viewProj[1][3] + viewProj[1][1],
        viewProj[2][3] + viewProj[2][1],
        viewProj[3][3] + viewProj[3][1]
    );
    
    // Top plane
    frustumPlanes[3] = glm::vec4(
        viewProj[0][3] - viewProj[0][1],
        viewProj[1][3] - viewProj[1][1],
        viewProj[2][3] - viewProj[2][1],
        viewProj[3][3] - viewProj[3][1]
    );
    
    // Near plane
    frustumPlanes[4] = glm::vec4(
        viewProj[0][3] + viewProj[0][2],
        viewProj[1][3] + viewProj[1][2],
        viewProj[2][3] + viewProj[2][2],
        viewProj[3][3] + viewProj[3][2]
    );
    
    // Far plane
    frustumPlanes[5] = glm::vec4(
        viewProj[0][3] - viewProj[0][2],
        viewProj[1][3] - viewProj[1][2],
        viewProj[2][3] - viewProj[2][2],
        viewProj[3][3] - viewProj[3][2]
    );
    
    // Normalize planes
    for (int i = 0; i < 6; ++i) {
        float length = glm::length(glm::vec3(frustumPlanes[i]));
        frustumPlanes[i] /= length;
    }
}

// Helper to create marching cubes table buffers (reusing from extractionManager)
struct MCTableUploadResult {
    Buffer deviceBuffer;
    Buffer stagingBuffer;
};

static MCTableUploadResult createTriTableBuffer(VulkanContext& context, VkCommandBuffer cmd) {
    MCTableUploadResult result = {};
    const int* triTableData = &MarchingCubes::triTable[0][0];
    VkDeviceSize triTableSize = 256 * 16 * sizeof(int);

    createBuffer(result.stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 triTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    memcpy(result.stagingBuffer.data, triTableData, triTableSize);

    createBuffer(result.deviceBuffer, context.getDevice(), context.getMemoryProperties(),
                 triTableSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkBufferCopy copyRegion = {0, 0, triTableSize};
    vkCmdCopyBuffer(cmd, result.stagingBuffer.buffer, result.deviceBuffer.buffer, 1, &copyRegion);

    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        result.deviceBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    return result;
}

static MCTableUploadResult createEdgeTableBuffer(VulkanContext& context, VkCommandBuffer cmd) {
    MCTableUploadResult result = {};
    const int* edgeTableData = &MarchingCubes::edgeTable[0];
    VkDeviceSize edgeTableSize = 256 * sizeof(int);

    createBuffer(result.stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 edgeTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    memcpy(result.stagingBuffer.data, edgeTableData, edgeTableSize);

    createBuffer(result.deviceBuffer, context.getDevice(), context.getMemoryProperties(),
                 edgeTableSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkBufferCopy copyRegion = {0, 0, edgeTableSize};
    vkCmdCopyBuffer(cmd, result.stagingBuffer.buffer, result.deviceBuffer.buffer, 1, &copyRegion);

    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        result.deviceBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    return result;
}

void extractAndRenderTransient(
    VulkanContext& vulkanContext,
    MinMaxOutput& minMaxOutput,
    FilteringOutput& filterOutput,
    PushConstants& pushConstants,
    const TransientExtractionPushConstants& renderConstants,
    VkCommandBuffer cmd,
    VkFormat colorFormat,
    VkFormat depthFormat,
    GPUProfiler* profiler,
    bool pmb
) {
    VkDevice device = vulkanContext.getDevice();

    // Use global static resources
    static std::vector<Buffer> stagingBuffers;

    if (!g_transientResources.initialized) {
        // Initialize pipeline once with the actual formats
        if (!g_transientResources.pipeline.setup(device, colorFormat, depthFormat,
                                    pushConstants.blockDim.x, pushConstants.blockDim.y, 
                                    pushConstants.blockDim.z, pmb)) {
            throw std::runtime_error("Failed to setup transient extraction pipeline");
        }

        // Create UBO for push constants
        g_transientResources.constantsUBO = createConstantsUBO(vulkanContext, pushConstants);

        // Create MC tables using a separate command buffer for initialization
        VkCommandBuffer initCmd = beginSingleTimeCommands(device, vulkanContext.getCommandPool());
        
        auto triTableResult = createTriTableBuffer(vulkanContext, initCmd);
        g_transientResources.mcTriTableBuffer = triTableResult.deviceBuffer;
        stagingBuffers.push_back(triTableResult.stagingBuffer);

        auto edgeTableResult = createEdgeTableBuffer(vulkanContext, initCmd);
        g_transientResources.mcEdgeTableBuffer = edgeTableResult.deviceBuffer;
        stagingBuffers.push_back(edgeTableResult.stagingBuffer);
        
        // Submit the initialization commands
        endSingleTimeCommands(device, vulkanContext.getCommandPool(), vulkanContext.getQueue(), initCmd);
        
        // Clean up staging buffers after submission
        for (auto& staging : stagingBuffers) {
            destroyBuffer(staging, device);
        }
        stagingBuffers.clear();

        // Create sampler for min-max image
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.minLod = 0;
        samplerInfo.maxLod = 16; // Allow sampling from all mip levels
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &samplerInfo, nullptr, &g_transientResources.minMaxSampler);

        g_transientResources.initialized = true;
    }

    // Update descriptors
    VkDescriptorBufferInfo uboInfo = {g_transientResources.constantsUBO.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo volInfo = {VK_NULL_HANDLE, minMaxOutput.volumeImage.imageView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo minMaxInfo = {g_transientResources.minMaxSampler, minMaxOutput.minMaxImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo blockCountInfo = {filterOutput.activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo blockIdInfo = {filterOutput.compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo mcTriTableInfo = {g_transientResources.mcTriTableBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo mcEdgeTableInfo = {g_transientResources.mcEdgeTableBuffer.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes(7);
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_transientResources.pipeline.descriptorSet_, 0, 0, 1, 
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_transientResources.pipeline.descriptorSet_, 1, 0, 1, 
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &volInfo, nullptr, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_transientResources.pipeline.descriptorSet_, 2, 0, 1, 
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &minMaxInfo, nullptr, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_transientResources.pipeline.descriptorSet_, 3, 0, 1, 
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &blockCountInfo, nullptr};
    writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_transientResources.pipeline.descriptorSet_, 4, 0, 1, 
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &blockIdInfo, nullptr};
    writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_transientResources.pipeline.descriptorSet_, 5, 0, 1, 
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mcTriTableInfo, nullptr};
    writes[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_transientResources.pipeline.descriptorSet_, 6, 0, 1, 
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mcEdgeTableInfo, nullptr};

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Note: Pipeline barriers are handled in meshtrex.cpp before vkCmdBeginRendering
    // to avoid executing barriers inside a dynamic render pass

    // Bind pipeline and descriptors
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_transientResources.pipeline.pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_transientResources.pipeline.pipelineLayout_, 
                           0, 1, &g_transientResources.pipeline.descriptorSet_, 0, nullptr);

    // Push constants (view-proj matrix and frustum planes)
    vkCmdPushConstants(cmd, g_transientResources.pipeline.pipelineLayout_, 
                      VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                      0, sizeof(TransientExtractionPushConstants), &renderConstants);

    // Dispatch mesh tasks
    if (profiler) {
        profiler->beginProfileRegion(cmd, VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT, "Transient_Mesh_Extraction");
    }

    uint32_t taskCount = filterOutput.activeBlockCount;
    if (taskCount == 0) {
        // If CPU-side count not available, use maximum possible
        taskCount = pushConstants.blockGridDim.x * pushConstants.blockGridDim.y * pushConstants.blockGridDim.z;
    }
    
    vkCmdDrawMeshTasksEXT(cmd, taskCount, 1, 1);

    if (profiler) {
        profiler->endProfileRegion(cmd, VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT);
    }

    // Note: Pipeline, sampler and buffers are kept static for reuse across frames
}

// Cleanup function for static resources
void cleanupTransientExtractionResources(VkDevice device) {
    if (g_transientResources.initialized) {
        g_transientResources.pipeline.cleanup();
        destroyBuffer(g_transientResources.constantsUBO, device);
        destroyBuffer(g_transientResources.mcTriTableBuffer, device);
        destroyBuffer(g_transientResources.mcEdgeTableBuffer, device);
        if (g_transientResources.minMaxSampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, g_transientResources.minMaxSampler, nullptr);
            g_transientResources.minMaxSampler = VK_NULL_HANDLE;
        }
        g_transientResources.initialized = false;
    }
}