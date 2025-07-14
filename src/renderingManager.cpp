#include "renderingManager.h"
#include "extractionOutput.h"
#include "renderingPipeline.h"
#include "resources.h"
#include <vector>
#include <iostream>

void updateRenderingDescriptors(
    VkDevice device,
    RenderingPipeline& renderingPipelineState,
    uint32_t currentFrame,
    const Buffer& sceneUboBuffer,
    const ExtractionOutput& extractionOutput
) {
    VkDescriptorSet currentDescriptorSet = renderingPipelineState.descriptorSets_[currentFrame];
    if (currentDescriptorSet == VK_NULL_HANDLE) return;
    // Ensure the necessary buffers from the extraction stage exist.
    if (extractionOutput.meshletDescriptorBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.vertexBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.indexBuffer.buffer == VK_NULL_HANDLE) {
        std::cerr << "Warning: Attempting to update rendering descriptors with missing geometry buffers." << std::endl;
        return;
    }

    std::vector<VkWriteDescriptorSet> writes;

    // Binding 0: Scene UBO
    VkDescriptorBufferInfo uboInfo = {sceneUboBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSets_[currentFrame], 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr});

    // Binding 1: Meshlet Descriptor Buffer
    VkDescriptorBufferInfo meshletDescInfo = {extractionOutput.meshletDescriptorBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSets_[currentFrame], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &meshletDescInfo, nullptr});

    // Binding 2: Vertex Buffer
    VkDescriptorBufferInfo vbInfo = {extractionOutput.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSets_[currentFrame], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vbInfo, nullptr});

    // Binding 3: Index Buffer
    VkDescriptorBufferInfo ibInfo = {extractionOutput.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSets_[currentFrame], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ibInfo, nullptr});

    // Binding 4: Meshlet Desc count Buffer
    VkDescriptorBufferInfo meshletCountInfo = {extractionOutput.meshletDescriptorCountBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSets_[currentFrame], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &meshletCountInfo, nullptr});

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void recordRenderingCommands(
    VkCommandBuffer commandBuffer,
    RenderingPipeline& renderingPipelineState,
    uint32_t currentFrame,
    const ExtractionOutput& extractionOutput,
    VkExtent2D swapchainExtent,
    VkImageView colorAttachmentView,
    VkImageView depthAttachmentView
) {
    // Don't render if there's no geometry
    if (renderingPipelineState.pipeline_ == VK_NULL_HANDLE || extractionOutput.meshletCount == 0) {
        return;
    }

    // --- Begin Dynamic Rendering ---
    VkClearValue colorClear = {0.1f, 0.1f, 0.1f, 1.0f};
    VkClearValue depthClear = {1.0f, 0};

    VkRenderingAttachmentInfo colorAttachmentInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = colorAttachmentView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = colorClear,
    };

    VkRenderingAttachmentInfo depthAttachmentInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = depthAttachmentView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = depthClear,
    };

    VkRenderingInfo renderingInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, swapchainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo,
    };

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    // --- Bind Pipeline and Descriptors ---
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderingPipelineState.pipeline_);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderingPipelineState.pipelineLayout_, 0, 1, &renderingPipelineState.descriptorSets_[currentFrame], 0, nullptr);

    // --- Set Dynamic States ---
    VkViewport viewport = { 0.0f, 0.0f, (float)swapchainExtent.width, (float)swapchainExtent.height, 0.0f, 1.0f };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, swapchainExtent};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // --- Issue Draw Call using Mesh Tasks ---
    uint32_t taskWorkgroupSize = 32;
    uint32_t numTaskWorkgroups = (extractionOutput.meshletCount + taskWorkgroupSize - 1) / taskWorkgroupSize;
    
    if (numTaskWorkgroups > 0) {
        vkCmdDrawMeshTasksEXT(commandBuffer, numTaskWorkgroups, 1, 1);
    }

    // --- End Dynamic Rendering ---
    vkCmdEndRendering(commandBuffer);
}