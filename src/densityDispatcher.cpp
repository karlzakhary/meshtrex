#include "densityDispatcher.h"
#include "vulkan_context.h"
#include "vulkan_utils.h"
#include "gpuProfiler.h"
#include <cstring>

void DensityDispatcher::ClassifiedBlocks::cleanup(VkDevice device) {
    // We only create the consolidated buffer now, not the individual buffers
    if (consolidatedBuffer.buffer != VK_NULL_HANDLE) {
        destroyBuffer(consolidatedBuffer, device);
    }
}

DensityDispatcher::ClassifiedBlocks DensityDispatcher::classifyAndUpload(
    const std::vector<DensityAnalyzer::BlockDensity>& densities,
    const VulkanContext& context) 
{
    ClassifiedBlocks classified;
    
    // Sort blocks by classification
    for (const auto& block : densities) {
        switch (block.classification) {
            case DensityAnalyzer::EMPTY:
                classified.emptyBlocks.push_back(block.blockID);
                break;
            case DensityAnalyzer::SPARSE:
                classified.sparseBlocks.push_back(block.blockID);
                break;
            case DensityAnalyzer::MEDIUM:
                classified.mediumBlocks.push_back(block.blockID);
                break;
            case DensityAnalyzer::DENSE:
                classified.denseBlocks.push_back(block.blockID);
                break;
        }
    }
    
    // Update counts
    classified.sparseCount = classified.sparseBlocks.size();
    classified.mediumCount = classified.mediumBlocks.size();
    classified.denseCount = classified.denseBlocks.size();
    
    // Debug output
    std::cout << "Block classification results:" << std::endl;
    std::cout << "  Empty blocks: " << classified.emptyBlocks.size() << std::endl;
    std::cout << "  Sparse blocks: " << classified.sparseCount << std::endl;
    std::cout << "  Medium blocks: " << classified.mediumCount << std::endl;
    std::cout << "  Dense blocks: " << classified.denseCount << std::endl;
    std::cout << "  Total active (non-empty): " << (classified.sparseCount + classified.mediumCount + classified.denseCount) << std::endl;
    
    // Create consolidated buffer with all blocks sorted by density
    std::vector<uint32_t> consolidatedBlocks;
    consolidatedBlocks.reserve(classified.sparseCount + classified.mediumCount + classified.denseCount);
    
    // Add sparse blocks
    classified.sparseOffset = 0;
    consolidatedBlocks.insert(consolidatedBlocks.end(), 
                             classified.sparseBlocks.begin(), 
                             classified.sparseBlocks.end());
    
    // Add medium blocks
    classified.mediumOffset = classified.sparseCount;
    consolidatedBlocks.insert(consolidatedBlocks.end(), 
                             classified.mediumBlocks.begin(), 
                             classified.mediumBlocks.end());
    
    // Add dense blocks
    classified.denseOffset = classified.sparseCount + classified.mediumCount;
    consolidatedBlocks.insert(consolidatedBlocks.end(), 
                             classified.denseBlocks.begin(), 
                             classified.denseBlocks.end());
    
    // Upload consolidated buffer
    if (!consolidatedBlocks.empty()) {
        uploadBlocks(consolidatedBlocks, classified.consolidatedBuffer, context);
    }
    
    return classified;
}

void DensityDispatcher::uploadBlocks(
    const std::vector<uint32_t>& blocks,
    Buffer& buffer,
    const VulkanContext& context) 
{
    size_t bufferSize = blocks.size() * sizeof(uint32_t);
    
    // Create staging buffer
    Buffer stagingBuffer;
    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Copy data to staging buffer
    memcpy(stagingBuffer.data, blocks.data(), bufferSize);
    
    // Create device buffer
    createBuffer(buffer, context.getDevice(), context.getMemoryProperties(),
                 bufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Copy from staging to device
    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    
    VkBufferCopy copyRegion{};
    copyRegion.size = bufferSize;
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, buffer.buffer, 1, &copyRegion);
    
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    
    // Clean up staging buffer
    destroyBuffer(stagingBuffer, context.getDevice());
}

void DensityDispatcher::recordDensityBasedExtraction(
    VkCommandBuffer cmd,
    const ClassifiedBlocks& blocks,
    const VulkanContext& context,
    VkPipeline sparsePipeline,
    VkPipeline mediumPipeline,
    VkPipeline densePipeline,
    VkPipelineLayout pipelineLayout,
    const DensityPushConstants& pushConstants,
    VkDescriptorSet descriptorSet,
    GPUProfiler* profiler,
    VkDescriptorSet sparseDescriptorSet,
    VkDescriptorSet denseDescriptorSet) 
{
    // Begin a dummy render pass (required for mesh shaders even with rasterizer discard)
    VkRenderingInfo renderingInfo = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = {{0, 0}, {1, 1}};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.pColorAttachments = nullptr;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Pre-allocate ranges based on block counts and expected output
    // Theoretical max per block: 64 cells × 12 vertices = 768 verts, 64 cells × 5 triangles × 3 = 960 indices
    // Allocate based on expected density characteristics
    const uint32_t VERTICES_PER_SPARSE_BLOCK = 384;   // 50% of max (sparse blocks have fewer active cells)
    const uint32_t VERTICES_PER_MEDIUM_BLOCK = 768;   // 100% of max (medium blocks can be quite full)
    const uint32_t VERTICES_PER_DENSE_BLOCK = 1024;   // 133% of max (safety margin for complex surfaces)
    
    const uint32_t INDICES_PER_SPARSE_BLOCK = 480;    // 50% of max (160 triangles × 3)
    const uint32_t INDICES_PER_MEDIUM_BLOCK = 960;    // 100% of max (320 triangles × 3)
    const uint32_t INDICES_PER_DENSE_BLOCK = 1280;    // 133% of max (426 triangles × 3)
    
    const uint32_t MESHLETS_PER_SPARSE_BLOCK = 1;     // One meshlet per block for sparse
    const uint32_t MESHLETS_PER_MEDIUM_BLOCK = 1;     // One meshlet per block for medium
    const uint32_t MESHLETS_PER_DENSE_BLOCK = 8;      // Dense blocks can have multiple meshlets
    
    // Calculate offsets for each density class
    uint32_t sparseVertexOffset = 0;
    uint32_t sparseIndexOffset = 0;
    uint32_t sparseMeshletOffset = 0;
    
    uint32_t mediumVertexOffset = blocks.sparseCount * VERTICES_PER_SPARSE_BLOCK;
    uint32_t mediumIndexOffset = blocks.sparseCount * INDICES_PER_SPARSE_BLOCK;
    uint32_t mediumMeshletOffset = blocks.sparseCount * MESHLETS_PER_SPARSE_BLOCK;
    
    uint32_t denseVertexOffset = mediumVertexOffset + blocks.mediumCount * VERTICES_PER_MEDIUM_BLOCK;
    uint32_t denseIndexOffset = mediumIndexOffset + blocks.mediumCount * INDICES_PER_MEDIUM_BLOCK;
    uint32_t denseMeshletOffset = mediumMeshletOffset + blocks.mediumCount * MESHLETS_PER_MEDIUM_BLOCK;
    
    // Dispatch 1: Sparse blocks (2 blocks per workgroup for better efficiency)
    if (blocks.sparseCount > 0) {
        if (profiler) {
            profiler->beginProfileRegion(cmd, VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT, "DensityExtraction_Sparse");
        }
        
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sparsePipeline);
        
        // Bind sparse-specific descriptor set if provided
        VkDescriptorSet sparseDesc = sparseDescriptorSet ? sparseDescriptorSet : descriptorSet;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipelineLayout, 0, 1, &sparseDesc, 0, nullptr);
        
        // Push constants with sparse-specific offsets
        DensityPushConstants sparsePC = pushConstants;
        sparsePC.globalVertexOffset = sparseVertexOffset;
        sparsePC.globalIndexOffset = sparseIndexOffset;
        sparsePC.globalMeshletOffset = sparseMeshletOffset;
        sparsePC.densityClass = 0; // Sparse
        sparsePC.activeBlockCount = blocks.sparseCount;
        sparsePC.blockOffset = blocks.sparseOffset; // Offset in consolidated buffer
        
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                          0, sizeof(DensityPushConstants), &sparsePC);
        
        // Dispatch with grouped workgroups (2 blocks per workgroup as defined in sparse shader)
        uint32_t sparseWorkgroups = (blocks.sparseCount + 1) / 2;
        vkCmdDrawMeshTasksEXT(cmd, sparseWorkgroups, 1, 1);
        
        if (profiler) {
            profiler->endProfileRegion(cmd, VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT);
        }
    }
    
    // Dispatch 2: Medium blocks (standard 1:1 mapping)
    if (blocks.mediumCount > 0) {
        if (profiler) {
            profiler->beginProfileRegion(cmd, VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT, "DensityExtraction_Medium");
        }
        
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mediumPipeline);
        
        // Bind medium descriptor set (uses main descriptor set)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        
        // Push constants with medium-specific offsets
        DensityPushConstants mediumPC = pushConstants;
        mediumPC.globalVertexOffset = mediumVertexOffset;
        mediumPC.globalIndexOffset = mediumIndexOffset;
        mediumPC.globalMeshletOffset = mediumMeshletOffset;
        mediumPC.densityClass = 1; // Medium
        mediumPC.activeBlockCount = blocks.mediumCount;
        mediumPC.blockOffset = blocks.mediumOffset; // Offset in consolidated buffer
        
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                          0, sizeof(DensityPushConstants), &mediumPC);
        
        vkCmdDrawMeshTasksEXT(cmd, blocks.mediumCount, 1, 1);
        
        if (profiler) {
            profiler->endProfileRegion(cmd, VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT);
        }
    }
    
    // Dispatch 3: Dense blocks (optimized for high complexity)
    if (blocks.denseCount > 0) {
        if (profiler) {
            profiler->beginProfileRegion(cmd, VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT, "DensityExtraction_Dense");
        }
        
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, densePipeline);
        
        // Bind dense-specific descriptor set if provided
        VkDescriptorSet denseDesc = denseDescriptorSet ? denseDescriptorSet : descriptorSet;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipelineLayout, 0, 1, &denseDesc, 0, nullptr);
        
        // Push constants with dense-specific offsets
        DensityPushConstants densePC = pushConstants;
        densePC.globalVertexOffset = denseVertexOffset;
        densePC.globalIndexOffset = denseIndexOffset;
        densePC.globalMeshletOffset = denseMeshletOffset;
        densePC.densityClass = 2; // Dense
        densePC.activeBlockCount = blocks.denseCount;
        densePC.blockOffset = blocks.denseOffset; // Offset in consolidated buffer
        
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                          0, sizeof(DensityPushConstants), &densePC);
        
        vkCmdDrawMeshTasksEXT(cmd, blocks.denseCount, 1, 1);
        
        if (profiler) {
            profiler->endProfileRegion(cmd, VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT);
        }
    }
    
    // End render pass
    vkCmdEndRendering(cmd);
    
    printf("Density dispatch: Sparse=%u workgroups (from %u blocks), Medium=%u, Dense=%u\n",
           (blocks.sparseCount + 1) / 2, blocks.sparseCount, blocks.mediumCount, blocks.denseCount);
    
    // Debug: Print expected allocation ranges
    uint32_t totalExpectedVertices = blocks.sparseCount * VERTICES_PER_SPARSE_BLOCK +
                                    blocks.mediumCount * VERTICES_PER_MEDIUM_BLOCK +
                                    blocks.denseCount * VERTICES_PER_DENSE_BLOCK;
    uint32_t totalExpectedIndices = blocks.sparseCount * INDICES_PER_SPARSE_BLOCK +
                                   blocks.mediumCount * INDICES_PER_MEDIUM_BLOCK +
                                   blocks.denseCount * INDICES_PER_DENSE_BLOCK;
    printf("Expected allocation: %u vertices, %u indices (%u triangles)\n",
           totalExpectedVertices, totalExpectedIndices, totalExpectedIndices / 3);
}