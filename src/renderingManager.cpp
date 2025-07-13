#include <iostream>

#include "common.h"
#include "extractionOutput.h"
#include "extractionPipeline.h"
#include "minMaxOutput.h"
#include "filteringOutput.h"
#include "renderingPipeline.h"
#include "dgcRenderingPipeline.h"
#include "resources.h"

// Updates descriptors for Pipeline 2 (Extraction)
void updateExtractionDescriptors(
    VkDevice device,
    ExtractionPipeline& extractionPipelineState,
    const MinMaxOutput& minMaxOutput,
    const FilteringOutput& filteringResult, // Input: Active block list/count
    const Buffer& uboIsovalue,              // Input: UBO containing current isovalue etc.
    const Buffer& marchingCubesTriTable,    // Input: MC Table
    const ExtractionOutput& extractionOutput // Output: Target Vtx/Idx/Indirect buffers
) {
    if (extractionPipelineState.descriptorSet_ == VK_NULL_HANDLE) return;

    std::vector<VkWriteDescriptorSet> writes;
    // Binding 0: UBO (isovalue etc.)
    VkDescriptorBufferInfo uboInfo = {uboIsovalue.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipelineState.descriptorSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr});
    // Binding 1: Volume Image
    VkDescriptorImageInfo volInfo = {VK_NULL_HANDLE, minMaxOutput.volumeImage.imageView, VK_IMAGE_LAYOUT_GENERAL};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipelineState.descriptorSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &volInfo, nullptr, nullptr});
    // Binding 2: Block IDs
    VkDescriptorBufferInfo blockIdInfo = {filteringResult.compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipelineState.descriptorSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &blockIdInfo, nullptr});
    // Binding 3: MC Table
    VkDescriptorBufferInfo mcTableInfo = {marchingCubesTriTable.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipelineState.descriptorSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mcTableInfo, nullptr});
    // Binding 4: Output Vertex Buffer
    VkDescriptorBufferInfo vbInfo = {extractionOutput.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipelineState.descriptorSet_, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vbInfo, nullptr});
    // Binding 5: Output Index Buffer
    VkDescriptorBufferInfo ibInfo = {extractionOutput.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipelineState.descriptorSet_, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ibInfo, nullptr});
    // Binding 6: Output Indirect Draw Buffer (if used)
    // VkDescriptorBufferInfo drawInfo = {extractionOutput.indirectDrawBuffer.buffer, 0, VK_WHOLE_SIZE};
    // writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipelineState.descriptorSet_, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &drawInfo, nullptr});


    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

// Records commands for Pipeline 2 (Extraction) - Runs conditionally
// Runs *inside* a Begin/EndRendering scope, even though rasterization is disabled.
void recordExtractionDispatch(
    VkCommandBuffer commandBuffer,
    ExtractionPipeline& extractionPipelineState,
    const FilteringOutput& filteringResult,
    const Buffer& indirectDrawBuffer  // Add indirect draw buffer parameter
) {
     if (extractionPipelineState.pipeline_ == VK_NULL_HANDLE) return;
     // Note: We no longer check activeBlockCount on CPU - GPU handles empty cases via indirect draw

    // Need to be inside vkCmdBegin/EndRendering for vkCmdDrawMeshTasks...

    // Bind Pipeline 2 and its descriptors
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipelineState.pipeline_);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipelineState.pipelineLayout_, 0, 1, &extractionPipelineState.descriptorSet_, 0, nullptr);

    // Use indirect draw for GPU-driven dispatch
    vkCmdDrawMeshTasksIndirectEXT(commandBuffer, indirectDrawBuffer.buffer, 0, 1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
}

// Updates descriptors for Pipeline 3 (Rendering)
void updateRenderingDescriptors(
    VkDevice device,
    RenderingPipeline& renderingPipelineState,
    const Buffer& sceneUboBuffer, // UBO containing MVP, lighting etc.
    const ExtractionOutput& extractionOutput   // Contains MeshletDesc, VB, IB
    // Add texture bindings if needed
) {
    if (renderingPipelineState.descriptorSet_ == VK_NULL_HANDLE) return;

    // Check if necessary buffers from extraction exist
    if (extractionOutput.meshletDescriptorBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.vertexBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.indexBuffer.buffer == VK_NULL_HANDLE) {
        std::cerr << "Warning: Missing necessary geometry buffers in updateRenderingDescriptors." << std::endl;
        return;
        }

    std::vector<VkWriteDescriptorSet> writes;
    // Binding 0: MVP/Lighting UBO
    VkDescriptorBufferInfo uboInfo = {sceneUboBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr});

    // Binding 1: Meshlet Descriptor Buffer
    VkDescriptorBufferInfo meshletDescInfo = {extractionOutput.meshletDescriptorBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &meshletDescInfo, nullptr});

    // Binding 2: Vertex Buffer
    VkDescriptorBufferInfo vbInfo = {extractionOutput.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vbInfo, nullptr});

    // Binding 3: Index Buffer
    VkDescriptorBufferInfo ibInfo = {extractionOutput.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, renderingPipelineState.descriptorSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ibInfo, nullptr});

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


// Records commands for Pipeline 3 (Rendering) - Runs every frame
void recordRenderingCommands(
    VkCommandBuffer commandBuffer,
    VkDevice device,
    RenderingPipeline& renderingPipelineState,
    const ExtractionOutput& extractionOutput, // Input: Geometry buffers
    VkExtent2D swapchainExtent,
    VkImageView colorAttachmentView,
    VkImageView depthAttachmentView,
    VkImageLayout colorAttachmentLayout,
    VkImageLayout depthAttachmentLayout
) {
    // Validate pipeline
    if (renderingPipelineState.pipeline_ == VK_NULL_HANDLE) {
        std::cerr << "Warning: Rendering pipeline not initialized" << std::endl;
        return;
    }
    
    // Check if we have geometry to render
    if (extractionOutput.meshletDescriptorBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.meshletDescriptorCountBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.vertexBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.indexBuffer.buffer == VK_NULL_HANDLE ||
        extractionOutput.renderingIndirectDrawBuffer.buffer == VK_NULL_HANDLE) {
        // No geometry extracted yet, skip rendering
        return;
    }

    // --- Generate indirect draw commands for rendering ---
    // This must happen BEFORE vkCmdBeginRendering
    
    // Ensure meshlet count buffer is ready for compute shader
    VkBufferMemoryBarrier2 meshletCountBarrier = bufferBarrier(
        extractionOutput.meshletDescriptorCountBuffer.buffer,
        VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,  // From extraction mesh shader
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,     // That wrote to it
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,   // To DGC generation compute shader
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,      // That will read from it
        0,
        VK_WHOLE_SIZE
    );
    pipelineBarrier(commandBuffer, {}, 1, &meshletCountBarrier, 0, {});
    
    // Generate indirect draw commands based on meshlet count
    const uint32_t taskWorkgroupSize = 32; // Task shader processes 32 meshlets per workgroup
    generateRenderingIndirectCommands(commandBuffer, device,
                                    extractionOutput.meshletDescriptorCountBuffer,
                                    extractionOutput.renderingIndirectDrawBuffer,
                                    taskWorkgroupSize);
    
    // Ensure indirect commands are written before being consumed
    VkBufferMemoryBarrier2 indirectBarrier = bufferBarrier(
        extractionOutput.renderingIndirectDrawBuffer.buffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        0,
        VK_WHOLE_SIZE
    );
    pipelineBarrier(commandBuffer, {}, 1, &indirectBarrier, 0, {});
    
    // --- Begin Dynamic Rendering ---
    VkRenderingAttachmentInfo colorAttachmentInfo = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    colorAttachmentInfo.imageView = colorAttachmentView;
    colorAttachmentInfo.imageLayout = colorAttachmentLayout;
    colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentInfo.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};
    
    VkRenderingAttachmentInfo depthAttachmentInfo = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthAttachmentInfo.imageView = depthAttachmentView;
    depthAttachmentInfo.imageLayout = depthAttachmentLayout;
    depthAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachmentInfo.clearValue.depthStencil = {1.0f, 0};
    
    VkRenderingInfo renderingInfo = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    renderingInfo.renderArea = {{0, 0}, swapchainExtent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    // --- Bind Pipeline and descriptors ---
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderingPipelineState.pipeline_);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderingPipelineState.pipelineLayout_, 0, 1, &renderingPipelineState.descriptorSet_, 0, nullptr);

    // --- Set Dynamic States ---
    VkViewport viewport = { 0.0f, 0.0f, (float)swapchainExtent.width, (float)swapchainExtent.height, 0.0f, 1.0f };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, swapchainExtent};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    
    // Issue indirect draw call using GPU-generated commands
    vkCmdDrawMeshTasksIndirectEXT(commandBuffer,
                                  extractionOutput.renderingIndirectDrawBuffer.buffer,
                                  0, // offset
                                  1, // draw count
                                  sizeof(VkDrawMeshTasksIndirectCommandEXT)); // stride

    // --- End Dynamic Rendering ---
    vkCmdEndRendering(commandBuffer);
}