#include "persistentGeometryExtraction.h"
#include "extractionPipeline.h"
#include "vulkan_utils.h"
#include "buffer.h"
#include "mc_tables.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>

// Structure for global allocation counters (matches what shaders expect)
struct GlobalExtractionCounters {
    uint32_t globalVertexOffset;
    uint32_t globalIndexOffset;
    uint32_t globalMeshletOffset;
    uint32_t maxVertices;
    uint32_t maxIndices;
    uint32_t maxMeshlets;
    uint32_t frameIndex;
    uint32_t padding;
};

// Structure for actual geometry counts read back from GPU
struct ActualGeometryCounts {
    uint32_t verticesGenerated;
    uint32_t indicesGenerated;
    uint32_t meshletsGenerated;
};

PersistentGeometryBuffers::PersistentGeometryBuffers(VulkanContext& context) 
    : context_(context), device_(context.getDevice()),
      currentVertexOffset(0), currentIndexOffset(0), currentMeshletOffset(0), currentFrame(0) {
}

PersistentGeometryBuffers::~PersistentGeometryBuffers() {
    if (device_ != VK_NULL_HANDLE) {
        destroyBuffer(globalVertexBuffer, device_);
        destroyBuffer(globalIndexBuffer, device_);
        destroyBuffer(globalMeshletBuffer, device_);
        destroyBuffer(globalCountersBuffer, device_);
        destroyBuffer(vertexCounterBuffer, device_);
        destroyBuffer(indexCounterBuffer, device_);
        destroyBuffer(meshletCounterBuffer, device_);
    }
}

void PersistentGeometryBuffers::initialize(uint32_t maxVertices, uint32_t maxIndices, uint32_t maxMeshlets) {
    this->maxVertices = maxVertices;
    this->maxIndices = maxIndices;
    this->maxMeshlets = maxMeshlets;

    std::cout << "Initializing Persistent Geometry Buffers:" << std::endl;
    std::cout << "  Max Vertices: " << maxVertices << " (" << (maxVertices * 48 / 1024 / 1024) << " MB)" << std::endl;
    std::cout << "  Max Indices: " << maxIndices << " (" << (maxIndices * 4 / 1024 / 1024) << " MB)" << std::endl;
    std::cout << "  Max Meshlets: " << maxMeshlets << " (" << (maxMeshlets * 16 / 1024 / 1024) << " MB)" << std::endl;

    // Create global vertex buffer (48 bytes per vertex: position + normal)
    VkDeviceSize vertexBufferSize = static_cast<VkDeviceSize>(maxVertices) * 48;
    createBuffer(globalVertexBuffer, device_, context_.getMemoryProperties(),
                 vertexBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create global index buffer
    VkDeviceSize indexBufferSize = static_cast<VkDeviceSize>(maxIndices) * sizeof(uint32_t);
    createBuffer(globalIndexBuffer, device_, context_.getMemoryProperties(),
                 indexBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create global meshlet buffer
    VkDeviceSize meshletBufferSize = static_cast<VkDeviceSize>(maxMeshlets) * 16;
    createBuffer(globalMeshletBuffer, device_, context_.getMemoryProperties(),
                 meshletBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create global counters buffer
    VkDeviceSize countersBufferSize = sizeof(GlobalExtractionCounters);
    createBuffer(globalCountersBuffer, device_, context_.getMemoryProperties(),
                 countersBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create atomic counter buffers (single uint each)
    createBuffer(vertexCounterBuffer, device_, context_.getMemoryProperties(),
                 sizeof(uint32_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                 
    createBuffer(indexCounterBuffer, device_, context_.getMemoryProperties(),
                 sizeof(uint32_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                 
    createBuffer(meshletCounterBuffer, device_, context_.getMemoryProperties(),
                 sizeof(uint32_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (globalVertexBuffer.buffer == VK_NULL_HANDLE ||
        globalIndexBuffer.buffer == VK_NULL_HANDLE ||
        globalMeshletBuffer.buffer == VK_NULL_HANDLE ||
        globalCountersBuffer.buffer == VK_NULL_HANDLE ||
        vertexCounterBuffer.buffer == VK_NULL_HANDLE ||
        indexCounterBuffer.buffer == VK_NULL_HANDLE ||
        meshletCounterBuffer.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to create persistent geometry buffers");
    }

    resetAllocations();
    
    // Initialize atomic counters to zero
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    vkCmdFillBuffer(cmd, vertexCounterBuffer.buffer, 0, sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, indexCounterBuffer.buffer, 0, sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, meshletCounterBuffer.buffer, 0, sizeof(uint32_t), 0);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    
    std::cout << "Persistent Geometry Buffers initialized successfully" << std::endl;
}

void PersistentGeometryBuffers::beginFrame(uint32_t frameIndex) {
    currentFrame = frameIndex;
    updateGlobalCounters();
}

void PersistentGeometryBuffers::endFrame() {
    std::cout << "Frame " << currentFrame << " persistent geometry: "
              << currentVertexOffset << " vertices, "
              << currentIndexOffset << " indices, "
              << currentMeshletOffset << " meshlets" << std::endl;
}

PersistentExtractionResult PersistentGeometryBuffers::extractPageToGlobalBuffers(
    const PageCoord& pageCoord,
    const StreamingExtractionConstants& constants,
    const FilteringOutput& filteringOutput,
    VkImageView volumeAtlasView,
    const Buffer& pageTableBuffer,
    uint32_t volumeSizeX,
    uint32_t volumeSizeY,
    uint32_t volumeSizeZ) {

    PersistentExtractionResult result = {};
    result.sourcePageCoord = pageCoord;
    result.isoValue = constants.isoValue;
    result.frameExtracted = currentFrame;

    if (filteringOutput.activeBlockCount == 0) {
        std::cout << "No active blocks for page (" << pageCoord.x << "," << pageCoord.y << "," << pageCoord.z << "), skipping extraction" << std::endl;
        result.success = true; // Not a failure, just nothing to extract
        return result;
    }

    // Estimate geometry requirements (conservative estimates)
    uint32_t cellsPerBlock = constants.blockSize * constants.blockSize * constants.blockSize;
    
    if (constants.blockSize == 0) {
        std::cerr << "ERROR: blockSize is 0! This will cause incorrect estimates." << std::endl;
        cellsPerBlock = 64; // Default to 4³
    }
    
    uint32_t estimatedVertices = filteringOutput.activeBlockCount * cellsPerBlock * 12; // Max 12 verts per cell
    uint32_t estimatedIndices = filteringOutput.activeBlockCount * cellsPerBlock * 15;  // Max 5 triangles * 3 indices
    uint32_t estimatedMeshlets = filteringOutput.activeBlockCount * cellsPerBlock / 64; // ~64 triangles per meshlet

    std::cout << "Extracting page (" << pageCoord.x << "," << pageCoord.y << "," << pageCoord.z 
              << ") with " << filteringOutput.activeBlockCount << " active blocks" << std::endl;
    std::cout << "  Estimated: " << estimatedVertices << " vertices, " 
              << estimatedIndices << " indices, " << estimatedMeshlets << " meshlets" << std::endl;

    // Allocate space in global buffers
    uint32_t globalVertexOffset, globalIndexOffset, globalMeshletOffset;
    if (!allocateSpace(estimatedVertices, estimatedIndices, estimatedMeshlets,
                      globalVertexOffset, globalIndexOffset, globalMeshletOffset)) {
        std::cerr << "Failed to allocate space in global buffers for page extraction" << std::endl;
        result.success = false;
        return result;
    }

    result.globalVertexOffset = globalVertexOffset;
    result.globalIndexOffset = globalIndexOffset;
    result.globalMeshletOffset = globalMeshletOffset;

    try {
        // Execute PMB extraction directly to global buffers
        executePMBExtractionToGlobalBuffers(pageCoord, constants, filteringOutput,
                                           volumeAtlasView, pageTableBuffer,
                                           globalVertexOffset, globalIndexOffset, globalMeshletOffset,
                                           volumeSizeX, volumeSizeY, volumeSizeZ);

        // Read back actual counts from GPU after extraction
        ActualGeometryCounts actualCounts = readbackActualGeometryCounts(filteringOutput);
        
        // Calculate counts for just this page (atomic counters are cumulative)
        result.verticesGenerated = actualCounts.verticesGenerated - lastVertexCounter;
        result.indicesGenerated = actualCounts.indicesGenerated - lastIndexCounter;
        result.meshletsGenerated = actualCounts.meshletsGenerated - lastMeshletCounter;
        
        // Update last counter values for next page
        lastVertexCounter = actualCounts.verticesGenerated;
        lastIndexCounter = actualCounts.indicesGenerated;
        lastMeshletCounter = actualCounts.meshletsGenerated;
        result.success = true;

        std::cout << "Successfully extracted page geometry to global buffers at offsets: "
                  << "V[" << globalVertexOffset << "], I[" << globalIndexOffset << "], M[" << globalMeshletOffset << "]" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during persistent extraction: " << e.what() << std::endl;
        result.success = false;
        
        // Rollback allocation
        currentVertexOffset = globalVertexOffset;
        currentIndexOffset = globalIndexOffset;
        currentMeshletOffset = globalMeshletOffset;
    }

    return result;
}

void PersistentGeometryBuffers::resetAllocations() {
    currentVertexOffset = 0;
    currentIndexOffset = 0;
    currentMeshletOffset = 0;
    lastVertexCounter = 0;
    lastIndexCounter = 0;
    lastMeshletCounter = 0;
    updateGlobalCounters();
    std::cout << "Reset persistent geometry allocations" << std::endl;
}

void PersistentGeometryBuffers::updateCounters(uint32_t verticesAdded, uint32_t indicesAdded, uint32_t meshletsAdded) {
    currentVertexOffset += verticesAdded;
    currentIndexOffset += indicesAdded;
    currentMeshletOffset += meshletsAdded;
    updateGlobalCounters();
}

bool PersistentGeometryBuffers::allocateSpace(uint32_t verticesNeeded, uint32_t indicesNeeded, uint32_t meshletsNeeded,
                                            uint32_t& vertexOffset, uint32_t& indexOffset, uint32_t& meshletOffset) {
    // Check if we have space
    if (currentVertexOffset + verticesNeeded > maxVertices) {
        std::cerr << "Vertex buffer full: need " << verticesNeeded << ", have " << (maxVertices - currentVertexOffset) << std::endl;
        return false;
    }
    if (currentIndexOffset + indicesNeeded > maxIndices) {
        std::cerr << "Index buffer full: need " << indicesNeeded << ", have " << (maxIndices - currentIndexOffset) << std::endl;
        return false;
    }
    if (currentMeshletOffset + meshletsNeeded > maxMeshlets) {
        std::cerr << "Meshlet buffer full: need " << meshletsNeeded << ", have " << (maxMeshlets - currentMeshletOffset) << std::endl;
        return false;
    }

    // Allocate space
    vertexOffset = currentVertexOffset;
    indexOffset = currentIndexOffset;
    meshletOffset = currentMeshletOffset;

    currentVertexOffset += verticesNeeded;
    currentIndexOffset += indicesNeeded;
    currentMeshletOffset += meshletsNeeded;

    return true;
}

void PersistentGeometryBuffers::updateGlobalCounters() {
    GlobalExtractionCounters counters = {};
    counters.globalVertexOffset = currentVertexOffset;
    counters.globalIndexOffset = currentIndexOffset;
    counters.globalMeshletOffset = currentMeshletOffset;
    counters.maxVertices = maxVertices;
    counters.maxIndices = maxIndices;
    counters.maxMeshlets = maxMeshlets;
    counters.frameIndex = currentFrame;
    counters.padding = 0;

    Buffer stagingBuffer;
    createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                 sizeof(GlobalExtractionCounters),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    memcpy(stagingBuffer.data, &counters, sizeof(GlobalExtractionCounters));

    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy region = {0, 0, sizeof(GlobalExtractionCounters)};
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, globalCountersBuffer.buffer, 1, &region);
    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);

    destroyBuffer(stagingBuffer, device_);
}

void PersistentGeometryBuffers::executePMBExtractionToGlobalBuffers(
    const PageCoord& pageCoord,
    const StreamingExtractionConstants& constants,
    const FilteringOutput& filteringOutput,
    VkImageView volumeAtlasView,
    const Buffer& pageTableBuffer,
    uint32_t globalVertexOffset,
    uint32_t globalIndexOffset,
    uint32_t globalMeshletOffset,
    uint32_t volumeSizeX,
    uint32_t volumeSizeY,
    uint32_t volumeSizeZ) {

    // Create extraction pipeline with streaming shaders
    ExtractionPipeline extractionPipeline;
    const char* taskShaderPath = "/spirv/streaming_marching_cubes_pmb.task.spv";
    const char* meshShaderPath = "/spirv/streaming_marching_cubes_debug.mesh.spv";
    
    if (!extractionPipeline.setup(device_, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_D32_SFLOAT,
                                 taskShaderPath, meshShaderPath)) {
        std::cerr << "Failed to create streaming extraction pipeline" << std::endl;
        return;
    }
    
    // Create Marching Cubes tables (factored from streamingExtractionManager.cpp)
    Buffer mcTriTableBuffer = createMCTriTableBuffer();
    Buffer mcEdgeTableBuffer = createMCEdgeTableBuffer();
    
    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());

    // Update descriptor set for extraction pipeline that uses GLOBAL buffers
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;
    
    bufferInfos.reserve(12);
    imageInfos.reserve(1);
    writes.reserve(12);

    // For streaming pipeline, binding 0 is the page table (STORAGE_BUFFER)
    bufferInfos.push_back({pageTableBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    // Volume atlas
    imageInfos.push_back({VK_NULL_HANDLE, volumeAtlasView, VK_IMAGE_LAYOUT_GENERAL});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &imageInfos.back()
    });

    // Active block count and IDs
    bufferInfos.push_back({filteringOutput.activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    bufferInfos.push_back({filteringOutput.compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 3,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    // MC tables
    bufferInfos.push_back({mcTriTableBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 4,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    bufferInfos.push_back({mcEdgeTableBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 5,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    // GLOBAL output buffers (key difference from per-page extraction)
    bufferInfos.push_back({globalVertexBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 6,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    // Vertex count buffer (binding 7) - uses dedicated atomic counter
    bufferInfos.push_back({vertexCounterBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 7,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    bufferInfos.push_back({globalIndexBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 8,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    // Index count buffer (binding 9) - uses dedicated atomic counter
    bufferInfos.push_back({indexCounterBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 9,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    bufferInfos.push_back({globalMeshletBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 10,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    // Meshlet count buffer (binding 11) - uses dedicated atomic counter
    bufferInfos.push_back({meshletCounterBuffer.buffer, 0, VK_WHOLE_SIZE});
    writes.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = extractionPipeline.descriptorSet_,
        .dstBinding = 11,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos.back()
    });

    // Update the extraction pipeline's descriptor set
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Create PMB extraction push constants that match the shader layout
    struct PMBExtractionPushConstants {
        glm::uvec3 pageCoord;      // Page coordinate for this pass
        uint32_t mipLevel;         // Mip level
        float isoValue;            // Isovalue for filtering
        uint32_t blockSize;        // Block size (typically 4)
        uint32_t pageSizeX;        // Page size X (64)
        uint32_t pageSizeY;        // Page size Y (32)
        uint32_t pageSizeZ;        // Page size Z (32)
        // Global buffer offsets for persistent geometry extraction
        uint32_t globalVertexOffset;
        uint32_t globalIndexOffset;
        uint32_t globalMeshletOffset;
        uint32_t volumeSizeX;      // Full volume dimension X (256)
        uint32_t volumeSizeY;      // Full volume dimension Y (256)
        uint32_t volumeSizeZ;      // Full volume dimension Z (256)
        uint32_t padding[2];
    };
    
    PMBExtractionPushConstants pmbConstants = {};
    pmbConstants.pageCoord = glm::uvec3(pageCoord.x, pageCoord.y, pageCoord.z);
    pmbConstants.mipLevel = 0; // Always 0 for now
    pmbConstants.isoValue = constants.isoValue;
    pmbConstants.blockSize = 4; // Always 4 for 4x4x4 blocks
    pmbConstants.pageSizeX = constants.pageSizeX;
    pmbConstants.pageSizeY = constants.pageSizeY;
    pmbConstants.pageSizeZ = constants.pageSizeZ;
    pmbConstants.globalVertexOffset = globalVertexOffset;
    pmbConstants.globalIndexOffset = globalIndexOffset;
    pmbConstants.globalMeshletOffset = globalMeshletOffset;
    // Pass full volume dimensions for correct vertex normalization
    pmbConstants.volumeSizeX = volumeSizeX;
    pmbConstants.volumeSizeY = volumeSizeY;
    pmbConstants.volumeSizeZ = volumeSizeZ;
    pmbConstants.padding[0] = 0;
    pmbConstants.padding[1] = 0;

    // Push the PMB constants that include global buffer offsets
    vkCmdPushConstants(cmd, extractionPipeline.pipelineLayout_, 
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                       0, sizeof(PMBExtractionPushConstants), &pmbConstants);

    // Begin dynamic rendering
    VkRenderingInfo renderingInfo = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    renderingInfo.layerCount = 1;
    renderingInfo.renderArea = {{0, 0}, {1, 1}};
    renderingInfo.colorAttachmentCount = 0;
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // Bind pipeline and descriptors
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipeline.pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipeline.pipelineLayout_,
                           0, 1, &extractionPipeline.descriptorSet_, 0, nullptr);
    
    // For streaming pipeline, page table is already bound at set 0, binding 0
    
    // Set dynamic state
    VkViewport viewport = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
    VkRect2D scissor = {{0, 0}, {1, 1}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    
    // Dispatch mesh tasks
    uint32_t taskCount = filteringOutput.activeBlockCount;
    std::cout << "PersistentGeometryBuffers: Dispatching " << taskCount 
              << " mesh tasks for page (" << pageCoord.x << "," << pageCoord.y << "," << pageCoord.z << ")" << std::endl;
    
    if (taskCount == 0) {
        std::cout << "WARNING: No active blocks to extract for this page!" << std::endl;
    }
    
    vkCmdDrawMeshTasksEXT(cmd, taskCount, 1, 1);
    
    vkCmdEndRendering(cmd);

    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    VK_CHECK(vkDeviceWaitIdle(device_));

    // Cleanup temporary resources
    destroyBuffer(mcTriTableBuffer, device_);
    destroyBuffer(mcEdgeTableBuffer, device_);
    // constantsUBO no longer used - streaming pipeline uses push constants
}

PersistentGeometryBuffers::ActualGeometryCounts PersistentGeometryBuffers::readbackActualGeometryCounts(const FilteringOutput& filteringOutput) {
    ActualGeometryCounts counts = {0, 0, 0};
    
    try {
        // Read back the atomic counters from GPU
        uint32_t vertexCount = 0, indexCount = 0, meshletCount = 0;
        
        // Create staging buffers for readback
        Buffer stagingBuffer = {};
        createBuffer(stagingBuffer, device_, context_.getMemoryProperties(),
                     sizeof(uint32_t) * 3, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        if (stagingBuffer.buffer == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to create staging buffer for counter readback");
        }

        VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
        
        // Copy the three atomic counters
        VkBufferCopy copyRegions[3] = {
            {0, 0, sizeof(uint32_t)},                     // vertex counter
            {0, sizeof(uint32_t), sizeof(uint32_t)},      // index counter
            {0, sizeof(uint32_t) * 2, sizeof(uint32_t)}   // meshlet counter
        };
        
        vkCmdCopyBuffer(cmd, vertexCounterBuffer.buffer, stagingBuffer.buffer, 1, &copyRegions[0]);
        vkCmdCopyBuffer(cmd, indexCounterBuffer.buffer, stagingBuffer.buffer, 1, &copyRegions[1]);
        vkCmdCopyBuffer(cmd, meshletCounterBuffer.buffer, stagingBuffer.buffer, 1, &copyRegions[2]);
        
        endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
        VK_CHECK(vkDeviceWaitIdle(device_));

        if (stagingBuffer.data) {
            uint32_t* counters = reinterpret_cast<uint32_t*>(stagingBuffer.data);
            vertexCount = counters[0];
            indexCount = counters[1];
            meshletCount = counters[2];
            
            counts.verticesGenerated = vertexCount;
            counts.indicesGenerated = indexCount;
            counts.meshletsGenerated = meshletCount;
            
            std::cout << "GPU Readback - Atomic counter values: "
                      << vertexCount << " vertices, "
                      << indexCount << " indices, "
                      << meshletCount << " meshlets" << std::endl;
        } else {
            std::cerr << "Failed to map staging buffer for counter readback" << std::endl;
        }
        
        destroyBuffer(stagingBuffer, device_);

    } catch (const std::exception& e) {
        std::cerr << "Error during geometry count readback: " << e.what() << std::endl;
    }
    
    return counts;
}

Buffer PersistentGeometryBuffers::createMCTriTableBuffer() {
    Buffer triTableBuffer = {};
    const int* triTableData = &MarchingCubes::triTable[0][0];
    VkDeviceSize triTableSize = 256 * 16 * sizeof(int);
    Buffer triTableStagingBuffer = {};

    createBuffer(triTableStagingBuffer, device_, context_.getMemoryProperties(),
                 triTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (triTableStagingBuffer.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to create staging buffer for MC TriTable.");
    }
    if (triTableStagingBuffer.data == nullptr) {
        destroyBuffer(triTableStagingBuffer, device_);
        throw std::runtime_error("Failed to map staging buffer for MC TriTable.");
    }

    memcpy(triTableStagingBuffer.data, triTableData, triTableSize);

    createBuffer(triTableBuffer, device_, context_.getMemoryProperties(),
                 triTableSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (triTableBuffer.buffer == VK_NULL_HANDLE) {
        destroyBuffer(triTableStagingBuffer, device_);
        throw std::runtime_error("Failed to create device buffer for MC TriTable.");
    }

    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, triTableSize};
    vkCmdCopyBuffer(cmd, triTableStagingBuffer.buffer, triTableBuffer.buffer, 1, &copyRegion);

    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        triTableBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    destroyBuffer(triTableStagingBuffer, device_);

    return triTableBuffer;
}

Buffer PersistentGeometryBuffers::createMCEdgeTableBuffer() {
    Buffer edgeTableBuffer = {};
    const int* edgeTableData = &MarchingCubes::edgeTable[0];
    VkDeviceSize edgeTableSize = 256 * sizeof(int);
    Buffer edgeTableStagingBuffer = {};

    createBuffer(edgeTableStagingBuffer, device_, context_.getMemoryProperties(),
                 edgeTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (edgeTableStagingBuffer.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to create staging buffer for MC EdgeTable.");
    }
    if (edgeTableStagingBuffer.data == nullptr) {
        destroyBuffer(edgeTableStagingBuffer, device_);
        throw std::runtime_error("Failed to map staging buffer for MC EdgeTable.");
    }

    memcpy(edgeTableStagingBuffer.data, edgeTableData, edgeTableSize);

    createBuffer(edgeTableBuffer, device_, context_.getMemoryProperties(),
                 edgeTableSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (edgeTableBuffer.buffer == VK_NULL_HANDLE) {
        destroyBuffer(edgeTableStagingBuffer, device_);
        throw std::runtime_error("Failed to create device buffer for MC EdgeTable.");
    }

    VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, edgeTableSize};
    vkCmdCopyBuffer(cmd, edgeTableStagingBuffer.buffer, edgeTableBuffer.buffer, 1, &copyRegion);

    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        edgeTableBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
    destroyBuffer(edgeTableStagingBuffer, device_);

    return edgeTableBuffer;
}

bool PersistentGeometryBuffers::exportToOBJ(const std::string& filePath) const {
    const uint32_t vertexCount = currentVertexOffset;
    const uint32_t indexCount = currentIndexOffset;
    
    if (vertexCount == 0 || indexCount == 0) {
        std::cerr << "No geometry to export (vertex count: " << vertexCount 
                  << ", index count: " << indexCount << ")" << std::endl;
        return false;
    }
    
    std::cout << "Exporting OBJ file with " << vertexCount << " vertices and " 
              << (indexCount / 3) << " triangles to " << filePath << std::endl;
    
    try {
        // Read vertices from GPU
        Buffer vertexReadback = {};
        VkDeviceSize vertexDataSize = vertexCount * sizeof(glm::vec4) * 2; // pos + normal
        createBuffer(vertexReadback, device_, context_.getMemoryProperties(),
                    vertexDataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VkCommandBuffer cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
        
        // Add barrier to ensure all writes are complete
        VkBufferMemoryBarrier2 vertexBarrier = bufferBarrier(
            globalVertexBuffer.buffer,
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            0, vertexDataSize);
        pipelineBarrier(cmd, {}, 1, &vertexBarrier, 0, {});
        
        VkBufferCopy copyRegion = {0, 0, vertexDataSize};
        vkCmdCopyBuffer(cmd, globalVertexBuffer.buffer, vertexReadback.buffer, 1, &copyRegion);
        endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
        
        // Read indices from GPU
        Buffer indexReadback = {};
        VkDeviceSize indexDataSize = indexCount * sizeof(uint32_t);
        createBuffer(indexReadback, device_, context_.getMemoryProperties(),
                    indexDataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        cmd = beginSingleTimeCommands(device_, context_.getCommandPool());
        
        // Add barrier for index buffer
        VkBufferMemoryBarrier2 indexBarrier = bufferBarrier(
            globalIndexBuffer.buffer,
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            0, indexDataSize);
        pipelineBarrier(cmd, {}, 1, &indexBarrier, 0, {});
        
        copyRegion = {0, 0, indexDataSize};
        vkCmdCopyBuffer(cmd, globalIndexBuffer.buffer, indexReadback.buffer, 1, &copyRegion);
        endSingleTimeCommands(device_, context_.getCommandPool(), context_.getQueue(), cmd);
        
        // Wait for transfers to complete
        VK_CHECK(vkDeviceWaitIdle(device_));
        
        // Write OBJ file
        std::ofstream objFile(filePath);
        if (!objFile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filePath << std::endl;
            destroyBuffer(vertexReadback, device_);
            destroyBuffer(indexReadback, device_);
            return false;
        }
        
        // Parse vertex data (interleaved position and normal)
        glm::vec4* vertexData = reinterpret_cast<glm::vec4*>(vertexReadback.data);
        
        // Write vertices
        for (uint32_t i = 0; i < vertexCount; i++) {
            glm::vec4 position = vertexData[i * 2];
            objFile << "v " << position.x << " " << position.y << " " << position.z << "\n";
        }
        
        // Write normals
        for (uint32_t i = 0; i < vertexCount; i++) {
            glm::vec4 normal = vertexData[i * 2 + 1];
            objFile << "vn " << normal.x << " " << normal.y << " " << normal.z << "\n";
        }
        
        // Write faces (OBJ uses 1-based indexing)
        uint32_t* indexData = reinterpret_cast<uint32_t*>(indexReadback.data);
        for (uint32_t i = 0; i < indexCount; i += 3) {
            objFile << "f " << (indexData[i] + 1) << "//" << (indexData[i] + 1) << " "
                   << (indexData[i+1] + 1) << "//" << (indexData[i+1] + 1) << " "
                   << (indexData[i+2] + 1) << "//" << (indexData[i+2] + 1) << "\n";
        }
        
        objFile.close();
        
        // Cleanup
        destroyBuffer(vertexReadback, device_);
        destroyBuffer(indexReadback, device_);
        
        std::cout << "Successfully exported geometry to " << filePath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error exporting OBJ: " << e.what() << std::endl;
        return false;
    }
}