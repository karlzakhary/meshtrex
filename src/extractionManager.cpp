#include "extractionManager.h"
#include "common.h"
#include "marchingCubesUtils.h"

#include "vulkan_context.h"
#include "filteringOutput.h"
#include "extractionOutput.h"
#include "extractionPipeline.h"
#include "buffer.h"
#include "image.h"
#include "resources.h"
#include "gpuProfiler.h"
#include "vulkan_utils.h"

#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include "densityAnalyzer.h"
#include "densityDispatcher.h"
#include "densityUtils.h"
#include "shaders.h"
#include <utility> // For std::move if needed later
#include "volume.h"

// Removed old MC table creation functions - now using MarchingCubesUtils

// Helper to create UBO - revised to take necessary values directly
Buffer createConstantsUBO(VulkanContext& context, PushConstants& pushConstants) {
    Buffer constantsUBO = {};
    VkDeviceSize bufferSize = sizeof(PushConstants);

    createBuffer(constantsUBO, context.getDevice(), context.getMemoryProperties(),
                 bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // Add DST for potential future updates via staging
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // Keep HOST_VISIBLE for easy update

    if (constantsUBO.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to create constants UBO buffer.");
    }
    if (constantsUBO.data == nullptr) {
         destroyBuffer(constantsUBO, context.getDevice());
         throw std::runtime_error("Failed to map constants UBO buffer.");
    }


    memcpy(constantsUBO.data, &pushConstants, bufferSize);

    // If not using HOST_COHERENT, need vkFlushMappedMemoryRanges here

    std::cout << "Constants UBO created and updated." << std::endl;
    return constantsUBO;
}


// --- Main Extraction Function Implementation ---

ExtractionOutput extractMeshletDescriptors(VulkanContext& vulkanContext, MinMaxOutput& minMaxOutput, FilteringOutput& filterOutput, PushConstants& pushConstants,
                                          VkCommandBuffer externalCmd, GPUProfiler* profiler, bool pmb) {
    std::cout << "\n--- Starting Meshlet Extraction ---" << std::endl;
    if (filterOutput.activeBlockCount == 0) {
        std::cout << "No active blocks found. Skipping meshlet extraction." << std::endl;
        return {};
    }

    VkDevice device = vulkanContext.getDevice();
    ExtractionPipeline extractionPipeline;
    ExtractionOutput extractionOutput = {};
    extractionOutput.device = device; // Store device handle for RAII cleanup

    Buffer constantsUBO = {};
    MarchingCubesTables mcTables = {};
    
    bool ownCommandBuffer = (externalCmd == VK_NULL_HANDLE);

    try {
        // 1. Setup Extraction Pipeline State
        // Ensure formats passed are compatible with device/swapchain if validation complains
        if (!extractionPipeline.setup(device, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_D32_SFLOAT,
                                     pushConstants.blockDim.x, pushConstants.blockDim.y, pushConstants.blockDim.z, pmb)) {
            throw std::runtime_error("Failed to setup Extraction Pipeline.");
        }

        // 2. Create Output Buffers (Sizing is critical and heuristic)
        const VkDeviceSize counterSize = sizeof(uint32_t);

        //For basic mesh shader
        const uint32_t CELLS_PER_BLOCK_FROM_SHADER = pushConstants.blockDim.x * pushConstants.blockDim.y * pushConstants.blockDim.z; // Match shader's #define
        const uint32_t MAX_VERTS_PER_CELL_FROM_SHADER = 12;    // Match shader's #define
        const uint32_t MAX_PRIMS_PER_CELL_FROM_SHADER = 5;     // Match shader's #define

        const VkDeviceSize MAX_TOTAL_VERTICES_BYTES =
            static_cast<VkDeviceSize>(filterOutput.activeBlockCount) *
            CELLS_PER_BLOCK_FROM_SHADER *
            MAX_VERTS_PER_CELL_FROM_SHADER *
            sizeof(VertexData);
        std::cout << " Max vertices: " << MAX_TOTAL_VERTICES_BYTES / sizeof(VertexData) << std::endl;
        const VkDeviceSize MAX_TOTAL_INDICES_BYTES =
            static_cast<VkDeviceSize>(filterOutput.activeBlockCount) *
            CELLS_PER_BLOCK_FROM_SHADER *
            MAX_PRIMS_PER_CELL_FROM_SHADER * 3 * // 3 indices per primitive
            sizeof(uint32_t);

        const VkDeviceSize MAX_MESHLET_DESCRIPTORS_BYTES =
            static_cast<VkDeviceSize>(filterOutput.activeBlockCount) *
                CELLS_PER_BLOCK_FROM_SHADER *
                    sizeof(MeshletDescriptor)
        ;

        std::cout << "Requesting output buffer sizes (incl. counter) based on " << filterOutput.activeBlockCount << " active blocks:" << std::endl;
        std::cout << "  - Vertex Buffer Size:       " << MAX_TOTAL_VERTICES_BYTES << " bytes" << std::endl;
        std::cout << "  - Index Buffer Size:        " << MAX_TOTAL_INDICES_BYTES << " bytes" << std::endl;
        std::cout << "  - Descriptor Buffer Size:   " << MAX_MESHLET_DESCRIPTORS_BYTES << " bytes" << std::endl;

        createBuffer(extractionOutput.vertexBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_TOTAL_VERTICES_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (extractionOutput.vertexBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create vertexBuffer."); }

        createBuffer(extractionOutput.vertexCountBuffer, device, vulkanContext.getMemoryProperties(),
                     counterSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (extractionOutput.vertexCountBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create vertexCountBuffer."); }

        createBuffer(extractionOutput.indexBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_TOTAL_INDICES_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (extractionOutput.indexBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create indexBuffer."); }

        createBuffer(extractionOutput.indexCountBuffer, device, vulkanContext.getMemoryProperties(),
             counterSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (extractionOutput.indexCountBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create indexCountBuffer."); }

        createBuffer(extractionOutput.meshletDescriptorBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_MESHLET_DESCRIPTORS_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (extractionOutput.meshletDescriptorBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create meshletDescriptorBuffer."); }

        createBuffer(extractionOutput.meshletDescriptorCountBuffer, device, vulkanContext.getMemoryProperties(),
                     counterSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (extractionOutput.meshletDescriptorCountBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create meshletDescriptorCountBuffer."); }


        // 3. Create UBO, MC Triangle Table, and number of vertices buffers
        // Pass necessary values from pushConstants to UBO helper
        constantsUBO = createConstantsUBO(vulkanContext, pushConstants);
        if (constantsUBO.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create constants UBO."); }
        
        // Create marching cubes tables using the utility
        MarchingCubesUtils::createMarchingCubesTables(mcTables, device, vulkanContext, false);
        if (mcTables.triTableBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create MC triangle table buffer."); }
        if (mcTables.numVerticesBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create MC numVertices table buffer."); }

        // 4. Update Descriptors
        if (extractionPipeline.descriptorSet_ == VK_NULL_HANDLE) { throw std::runtime_error("Extraction pipeline descriptor set is null."); }
        std::vector<VkWriteDescriptorSet> writes;
        VkDescriptorBufferInfo uboInfo = {constantsUBO.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo volInfo = {VK_NULL_HANDLE, minMaxOutput.volumeImage.imageView, VK_IMAGE_LAYOUT_GENERAL}; // Assuming GENERAL layout from filtering
        VkDescriptorBufferInfo blockCountInfo = {filterOutput.activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo blockIdInfo = {filterOutput.compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE};
        // MC tables will use buffer views for texel buffers
        VkDescriptorBufferInfo vbInfo = {extractionOutput.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo vertexCountInfo = {extractionOutput.vertexCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo ibInfo = {extractionOutput.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo indexCountInfo = {extractionOutput.indexCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo descInfo = {extractionOutput.meshletDescriptorBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo descCountInfo = {extractionOutput.meshletDescriptorCountBuffer.buffer,0,  VK_WHOLE_SIZE};

        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            nullptr,
            &uboInfo,
            nullptr
        }); // Binding 0: UBO
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            1,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            &volInfo,
            nullptr,
            nullptr
        }); // Binding 1: Volume Image
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            2,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &blockCountInfo,
            nullptr
        }); // Binding 2: Active Block counts
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            3,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &blockIdInfo,
            nullptr
        }); // Binding 3: Active Block IDs
        VkDescriptorBufferInfo triTableInfo{};
        triTableInfo.buffer = mcTables.triTableBuffer.buffer;
        triTableInfo.offset = 0;
        triTableInfo.range = 256 * 16 * sizeof(uint8_t);
        
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            4,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &triTableInfo,
            nullptr
        }); // Binding 4: MC Triangle Table
        VkDescriptorBufferInfo numVerticesInfo{};
        numVerticesInfo.buffer = mcTables.numVerticesBuffer.buffer;
        numVerticesInfo.offset = 0;
        numVerticesInfo.range = 256 * sizeof(uint8_t);
        
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            5,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &numVerticesInfo,
            nullptr
        }); // Binding 5: MC NumVertices Table (edge table is now hardcoded in shader)
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            6,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &vbInfo,
            nullptr
        }); // Binding 6: Output Vertices
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            7,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &vertexCountInfo,
            nullptr
        }); // Binding 7: Output Vertex count
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            8,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &ibInfo,
            nullptr
        }); // Binding 8: Output Indices
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            9,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &indexCountInfo,
            nullptr
        }); // Binding 9: Output Indices count
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            10,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &descInfo,
            nullptr
        }); // Binding 10: Output Meshlet descriptors
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            11,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &descCountInfo,
            nullptr
        }); // Binding 11: Output meshlet descriptors count
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // 5. Record Command Buffer
        VkCommandBuffer cmd;
        
        if (ownCommandBuffer) {
            cmd = beginSingleTimeCommands(device, vulkanContext.getCommandPool());
        } else {
            cmd = externalCmd;
        }

        // --- Initialize Atomic Counters ---
        std::vector<VkBufferMemoryBarrier2> fillToComputeBarriers;
        const VkPipelineStageFlags2 ATOMIC_SHADER_STAGES = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
        const VkAccessFlags2 ATOMIC_ACCESS = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        vkCmdFillBuffer(cmd, extractionOutput.vertexCountBuffer.buffer, 0, counterSize, 0);
        fillToComputeBarriers.push_back(bufferBarrier(extractionOutput.vertexBuffer.buffer,VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ATOMIC_SHADER_STAGES, ATOMIC_ACCESS,0, counterSize ));
        vkCmdFillBuffer(cmd, extractionOutput.indexCountBuffer.buffer, 0, counterSize, 0);
        fillToComputeBarriers.push_back(bufferBarrier(extractionOutput.indexBuffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ATOMIC_SHADER_STAGES, ATOMIC_ACCESS, 0, counterSize ));
        vkCmdFillBuffer(cmd, extractionOutput.meshletDescriptorCountBuffer.buffer, 0, counterSize, 0);
        fillToComputeBarriers.push_back(bufferBarrier(extractionOutput.meshletDescriptorBuffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ATOMIC_SHADER_STAGES, ATOMIC_ACCESS, 0, counterSize ));
        pipelineBarrier(cmd, {}, fillToComputeBarriers.size(), fillToComputeBarriers.data(), 0, {});

        // --- Barriers Before Extraction ---
        std::vector<VkBufferMemoryBarrier2> preBufferBarriers;
        std::vector<VkImageMemoryBarrier2> preImageBarriers;
        const VkPipelineStageFlags2 EXTRACTION_SHADER_STAGES = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        const VkAccessFlags2 EXTRACTION_WRITE_ACCESS = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        const VkAccessFlags2 EXTRACTION_READ_ACCESS = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT;
        const VkAccessFlags2 EXTRACTION_ATOMIC_ACCESS = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        // Inputs Readable
        preBufferBarriers.push_back(bufferBarrier(
            filterOutput.activeBlockCountBuffer.buffer,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            0,
            VK_WHOLE_SIZE
        ));
        preBufferBarriers.push_back(bufferBarrier(
            filterOutput.compactedBlockIdBuffer.buffer,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            0,
            VK_WHOLE_SIZE
        ));
        // *** Assuming volume image is already in GENERAL layout from filtering pass ***
        preImageBarriers.push_back(imageBarrier(
            minMaxOutput.volumeImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            EXTRACTION_SHADER_STAGES,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT
        ));
        // MC tables don't need barriers as they're created with proper access already

        // *** Added Barrier for UBO ***
        preBufferBarriers.push_back(bufferBarrier(constantsUBO.buffer, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT, EXTRACTION_SHADER_STAGES, VK_ACCESS_2_UNIFORM_READ_BIT, 0, VK_WHOLE_SIZE));

        // Outputs Writable (Simplified barrier after fill barrier)
        preBufferBarriers.push_back(bufferBarrier(
            extractionOutput.vertexBuffer.buffer,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            EXTRACTION_SHADER_STAGES,
            EXTRACTION_WRITE_ACCESS | EXTRACTION_ATOMIC_ACCESS,
            0,
            VK_WHOLE_SIZE
        ));
        preBufferBarriers.push_back(bufferBarrier(
            extractionOutput.indexBuffer.buffer,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            EXTRACTION_SHADER_STAGES,
            EXTRACTION_WRITE_ACCESS | EXTRACTION_ATOMIC_ACCESS,
            0,
            VK_WHOLE_SIZE
        ));
        preBufferBarriers.push_back(bufferBarrier(
            extractionOutput.meshletDescriptorBuffer.buffer,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            EXTRACTION_SHADER_STAGES,
            EXTRACTION_WRITE_ACCESS | EXTRACTION_ATOMIC_ACCESS,
            0,
            VK_WHOLE_SIZE
        ));

        pipelineBarrier(cmd, {}, preBufferBarriers.size(), preBufferBarriers.data(), preImageBarriers.size(), preImageBarriers.data());

        // --- Begin Dynamic Rendering ---
        VkRenderingInfo renderingInfo = { VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.layerCount = 1;
        renderingInfo.renderArea = {{0, 0}, {1, 1}};
        renderingInfo.colorAttachmentCount = 0;
        renderingInfo.pColorAttachments = nullptr;
        renderingInfo.pDepthAttachment = nullptr;
        renderingInfo.pStencilAttachment = nullptr;
        vkCmdBeginRendering(cmd, &renderingInfo);

        // --- Bind Pipeline & Descriptors ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipeline.pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipeline.pipelineLayout_, 0, 1, &extractionPipeline.descriptorSet_, 0, nullptr);

        // --- Set Dynamic State ---
        VkViewport viewport = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
        VkRect2D scissor = {{0, 0}, {1, 1}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // --- Dispatch mesh tasks using direct draw ---
        if (profiler) {
            profiler->beginProfileRegion(cmd, VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT, "Mesh_Extraction");
        }
        
        uint32_t taskCount = filterOutput.activeBlockCount;
        std::cout << "Dispatching " << taskCount << " mesh tasks..." << std::endl;
        vkCmdDrawMeshTasksEXT(cmd, taskCount, 1, 1);
        
        if (profiler) {
            profiler->endProfileRegion(cmd, VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT);
        }

        // --- End Dynamic Rendering ---
        vkCmdEndRendering(cmd);

        // --- Barriers After Extraction ---
        std::vector<VkBufferMemoryBarrier2> postBufferBarriers;
        VkPipelineStageFlags2 postSrcStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        // *** Corrected: Use STORAGE_WRITE for srcAccessMask ***
        VkAccessFlags2 postSrcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkPipelineStageFlags2 postDstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        VkAccessFlags2 postDstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;

        postBufferBarriers.push_back(bufferBarrier(extractionOutput.vertexBuffer.buffer, postSrcStageMask, postSrcAccessMask, postDstStageMask, postDstAccessMask, 0, VK_WHOLE_SIZE));
        postBufferBarriers.push_back(bufferBarrier(extractionOutput.indexBuffer.buffer, postSrcStageMask, postSrcAccessMask, postDstStageMask, postDstAccessMask, 0, VK_WHOLE_SIZE));
        postBufferBarriers.push_back(bufferBarrier(extractionOutput.meshletDescriptorBuffer.buffer, postSrcStageMask, postSrcAccessMask, postDstStageMask, postDstAccessMask, 0, VK_WHOLE_SIZE));

        pipelineBarrier(cmd, {}, postBufferBarriers.size(), postBufferBarriers.data(), 0, {});

        // 6. Submit and Wait
        if (ownCommandBuffer) {
            endSingleTimeCommands(device, vulkanContext.getCommandPool(), vulkanContext.getQueue(), cmd);
        }
        extractionOutput.meshletCount = filterOutput.activeBlockCount; // Still an upper bound

    } catch (const std::exception& e) {
        std::cerr << "Error during meshlet extraction: " << e.what() << std::endl;
        // Cleanup is handled by RAII destructors for pipeline/output
        // Need to manually clean up UBO/MC Table if created before throw
        destroyBuffer(constantsUBO, device); // Safe to call even if null
        MarchingCubesUtils::destroyMarchingCubesTables(mcTables, device);
        throw;
    }

    // Cleanup temporary UBO and MC table buffer
    // RAII handles extractionPipeline and extractionOutput cleanup
    if (ownCommandBuffer) {
        // Only destroy buffers if we own the command buffer (already submitted)
        destroyBuffer(constantsUBO, device);
        MarchingCubesUtils::destroyMarchingCubesTables(mcTables, device);
    } else {
        
        // Store temporary buffers for later cleanup
        extractionOutput.tempResources.device = device;
        extractionOutput.tempResources.addBuffer(constantsUBO);
        extractionOutput.tempResources.addBuffer(mcTables.triTableBuffer);
        extractionOutput.tempResources.addBuffer(mcTables.numVerticesBuffer);
        
        // Also store pipeline resources to prevent premature destruction
        extractionOutput.tempResources.addPipeline(extractionPipeline.pipeline_);
        extractionOutput.tempResources.addPipelineLayout(extractionPipeline.pipelineLayout_);
        extractionOutput.tempResources.addDescriptorSetLayout(extractionPipeline.descriptorSetLayout_);
        extractionOutput.tempResources.addDescriptorPool(extractionPipeline.getDescriptorPool());
        extractionOutput.tempResources.addShaderModule(extractionPipeline.getTaskShaderModule());
        extractionOutput.tempResources.addShaderModule(extractionPipeline.getMeshShaderModule());
        // Note: Descriptor sets are freed when the pool is destroyed
        
        // Transfer ownership to prevent double-free
        extractionPipeline.transferResourceOwnership();
    }

    std::cout << "--- Finished Meshlet Extraction ---" << std::endl;
    return extractionOutput;
}

// New density-based extraction implementation

ExtractionOutput extractMeshletDescriptorsWithDensity(
    VulkanContext& vulkanContext, 
    MinMaxOutput& minMaxOutput, 
    FilteringOutput& filterOutput, 
    PushConstants& pushConstants,
    const Volume& volume,
    bool useDensityDispatch,
    VkCommandBuffer externalCmd, 
    GPUProfiler* profiler, 
    bool pmb) 
{
    // If not using density dispatch, fall back to original implementation
    if (!useDensityDispatch) {
        return extractMeshletDescriptors(vulkanContext, minMaxOutput, filterOutput, 
                                       pushConstants, externalCmd, profiler, pmb);
    }
    
    std::cout << "\n--- Starting Density-Based Meshlet Extraction ---" << std::endl;
    
    if (filterOutput.activeBlockCount == 0) {
        std::cout << "No active blocks found. Skipping meshlet extraction." << std::endl;
        return {};
    }
    
    // Track CPU timing for density analysis
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    
    // 1. Read back active blocks from GPU
    std::cout << "Reading back " << filterOutput.activeBlockCount << " active blocks from GPU..." << std::endl;
    std::vector<uint32_t> activeBlocks = DensityUtils::readbackActiveBlocks(
        vulkanContext, 
        filterOutput.compactedBlockIdBuffer, 
        filterOutput.activeBlockCount
    );
    
    // 2. Perform CPU-based density analysis
    std::cout << "Analyzing block densities..." << std::endl;
    DensityAnalyzer analyzer;
    DensityAnalyzer::VolumeParams volumeParams = {
        glm::uvec3(pushConstants.volumeDim.x, pushConstants.volumeDim.y, pushConstants.volumeDim.z),
        glm::uvec3(pushConstants.blockDim.x, pushConstants.blockDim.y, pushConstants.blockDim.z),
        glm::uvec3(pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z)
    };
    
    auto densities = analyzer.analyzeActiveBlocks(
        volume.volume_data.data(),
        volumeParams,
        activeBlocks,
        pushConstants.isovalue
    );
    
    // 3. Classify and upload blocks to GPU
    std::cout << "Uploading classified blocks to GPU..." << std::endl;
    DensityDispatcher dispatcher;
    auto classified = dispatcher.classifyAndUpload(densities, vulkanContext);
    
    auto cpuEndTime = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuEndTime - cpuStartTime).count() / 1000.0f;
    std::cout << "CPU Density Analysis Time: " << cpuDuration << " ms" << std::endl;
    
    // Skip if no non-empty blocks
    if (classified.sparseCount == 0 && classified.mediumCount == 0 && classified.denseCount == 0) {
        std::cout << "All blocks are empty. Skipping extraction." << std::endl;
        classified.cleanup(vulkanContext.getDevice());
        return {};
    }
    
    // 4. Set up extraction pipeline (similar to original)
    VkDevice device = vulkanContext.getDevice();
    ExtractionOutput extractionOutput = {};
    extractionOutput.device = device;
    
    // Create density-specific pipelines
    ExtractionPipeline mediumPipeline;
    ExtractionPipeline sparsePipeline;
    ExtractionPipeline densePipeline;
    
    // Sparse pipeline with custom shaders
    if (!sparsePipeline.setupWithShaders(
            device,
            VK_FORMAT_R8G8B8A8_UNORM,  // Color format (unused for extraction)
            VK_FORMAT_D32_SFLOAT,      // Depth format (unused for extraction)
            pushConstants.blockDim.x, pushConstants.blockDim.y, pushConstants.blockDim.z, pmb,
            "/spirv/marching_cubes_sparse.task.spv",
            "/spirv/marching_cubes_sparse.mesh.spv")) {
        std::cerr << "Failed to create sparse pipeline" << std::endl;
        classified.cleanup(device);
        return {};
    }

    // Medium pipeline with custom shaders
    if (!mediumPipeline.setupWithShaders(
            device,
            VK_FORMAT_R8G8B8A8_UNORM,  // Color format (unused for extraction)
            VK_FORMAT_D32_SFLOAT,      // Depth format (unused for extraction)
            pushConstants.blockDim.x, pushConstants.blockDim.y, pushConstants.blockDim.z, pmb,
            "/spirv/marching_cubes_medium.task.spv",
            "/spirv/marching_cubes_medium.mesh.spv")) {
        std::cerr << "Failed to create medium pipeline" << std::endl;
        classified.cleanup(device);
        return {};
    }
    
    // Dense pipeline with custom shaders
    if (!densePipeline.setupWithShaders(
            device,
            VK_FORMAT_R8G8B8A8_UNORM,  // Color format (unused for extraction)
            VK_FORMAT_D32_SFLOAT,      // Depth format (unused for extraction)
            pushConstants.blockDim.x, pushConstants.blockDim.y, pushConstants.blockDim.z, pmb,
            "/spirv/marching_cubes_dense.task.spv", 
            "/spirv/marching_cubes_dense.mesh.spv")) {
        std::cerr << "Failed to create dense pipeline" << std::endl;
        sparsePipeline.cleanup();
        classified.cleanup(device);
        return {};
    }
    
    Buffer constantsUBO = {};
    MarchingCubesTables mcTables = {};
    
    bool ownCommandBuffer = (externalCmd == VK_NULL_HANDLE);
    
    try {
        // Create output buffers (same sizing logic as original)
        const VkDeviceSize counterSize = sizeof(uint32_t);
        const uint32_t CELLS_PER_BLOCK = pushConstants.blockDim.x * pushConstants.blockDim.y * pushConstants.blockDim.z;
        const uint32_t MAX_VERTS_PER_CELL = 12;
        const uint32_t MAX_PRIMS_PER_CELL = 5;
        
        const VkDeviceSize MAX_TOTAL_VERTICES_BYTES = 
            static_cast<VkDeviceSize>(filterOutput.activeBlockCount) * CELLS_PER_BLOCK * MAX_VERTS_PER_CELL * sizeof(VertexData);
        const VkDeviceSize MAX_TOTAL_INDICES_BYTES = 
            static_cast<VkDeviceSize>(filterOutput.activeBlockCount) * CELLS_PER_BLOCK * MAX_PRIMS_PER_CELL * 3 * sizeof(uint32_t);
        const VkDeviceSize MAX_MESHLET_DESCRIPTORS_BYTES = 
            static_cast<VkDeviceSize>(filterOutput.activeBlockCount) * CELLS_PER_BLOCK * sizeof(MeshletDescriptor);
        
        // Create all buffers (same as original)
        createBuffer(extractionOutput.vertexBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_TOTAL_VERTICES_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        createBuffer(extractionOutput.vertexCountBuffer, device, vulkanContext.getMemoryProperties(),
                     counterSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        createBuffer(extractionOutput.indexBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_TOTAL_INDICES_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        createBuffer(extractionOutput.indexCountBuffer, device, vulkanContext.getMemoryProperties(),
                     counterSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        createBuffer(extractionOutput.meshletDescriptorBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_MESHLET_DESCRIPTORS_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        createBuffer(extractionOutput.meshletDescriptorCountBuffer, device, vulkanContext.getMemoryProperties(),
                     counterSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        // Create UBO and MC tables
        constantsUBO = createConstantsUBO(vulkanContext, pushConstants);
        MarchingCubesUtils::createMarchingCubesTables(mcTables, device, vulkanContext, false);
        
        // Record command buffer
        VkCommandBuffer cmd = ownCommandBuffer ? 
            beginSingleTimeCommands(device, vulkanContext.getCommandPool()) : externalCmd;
        
        // Initialize atomic counters
        vkCmdFillBuffer(cmd, extractionOutput.vertexCountBuffer.buffer, 0, counterSize, 0);
        vkCmdFillBuffer(cmd, extractionOutput.indexCountBuffer.buffer, 0, counterSize, 0);
        vkCmdFillBuffer(cmd, extractionOutput.meshletDescriptorCountBuffer.buffer, 0, counterSize, 0);
        
        // Barriers (simplified for brevity - in real implementation, use same barriers as original)
        VkMemoryBarrier2 fillBarrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        };
        VkDependencyInfo depInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &fillBarrier
        };
        vkCmdPipelineBarrier2(cmd, &depInfo);
        
        // Create DensityPushConstants from original PushConstants
        DensityPushConstants densityPC = {};
        densityPC.volumeDim = pushConstants.volumeDim;
        densityPC.blockDim = pushConstants.blockDim;
        densityPC.blockGridDim = pushConstants.blockGridDim;
        densityPC.voxelSize = glm::vec4(volume.voxel_size); // Default voxel size
        densityPC.origin = glm::vec4(0.0f);    // Default origin
        densityPC.isovalue = pushConstants.isovalue;
        densityPC.activeBlockCount = filterOutput.activeBlockCount;
        // Offsets will be set by the dispatcher
        densityPC.globalVertexOffset = 0;
        densityPC.globalIndexOffset = 0;
        densityPC.globalMeshletOffset = 0;
        densityPC.densityClass = 0;
        densityPC.blockOffset = 0;
        
        // Update all required descriptor bindings
        std::vector<VkWriteDescriptorSet> writes;
        VkDescriptorBufferInfo uboInfo = {constantsUBO.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo volInfo = {VK_NULL_HANDLE, minMaxOutput.volumeImage.imageView, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo blockCountInfo = {filterOutput.activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo consolidatedBufferInfo = {classified.consolidatedBuffer.buffer, 0, VK_WHOLE_SIZE};
        // MC tables will use buffer views for texel buffers
        VkDescriptorBufferInfo vbInfo = {extractionOutput.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo vertexCountInfo = {extractionOutput.vertexCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo ibInfo = {extractionOutput.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo indexCountInfo = {extractionOutput.indexCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo meshletDescInfo = {extractionOutput.meshletDescriptorBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo descCountInfo = {extractionOutput.meshletDescriptorCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        
        // Binding 0: UBO
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            nullptr, &uboInfo, nullptr});
        
        // Binding 1: Volume Image
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            &volInfo, nullptr, nullptr});
        
        // Binding 2: Active Block Count
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &blockCountInfo, nullptr});
        
        // Binding 3: Consolidated Block IDs (used by all density pipelines with offsets)
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &consolidatedBufferInfo, nullptr});
        
        // TODO: Update density-based extraction to use new MC table utils
        // For now, commented out as density-based approach is being ignored
        /*
        // Binding 4: MC Triangle Table
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &mcTriTableInfo, nullptr});
        
        // Binding 5: MC Edge Table
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &mcEdgeTableInfo, nullptr});
        */
        
        // Binding 6: Output Vertex Buffer
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &vbInfo, nullptr});
        
        // Binding 7: Output Vertex Count
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &vertexCountInfo, nullptr});
        
        // Binding 8: Output Index Buffer
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &ibInfo, nullptr});
        
        // Binding 9: Output Index Count
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &indexCountInfo, nullptr});
        
        // Binding 10: Output Meshlet Descriptors
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &meshletDescInfo, nullptr});
        
        // Binding 11: Output Meshlet Descriptor Count
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
            mediumPipeline.descriptorSet_, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr, &descCountInfo, nullptr});
        
        // Update descriptors for medium pipeline
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        
        // Also update descriptors for sparse pipeline
        for (auto& write : writes) {
            write.dstSet = sparsePipeline.descriptorSet_;
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        
        // And for dense pipeline
        for (auto& write : writes) {
            write.dstSet = densePipeline.descriptorSet_;
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        
        // Record density-based extraction with profiling
        if (profiler) {
            // profiler->beginProfileRegion(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, "DensityBasedExtraction_Total");
        }
        
        dispatcher.recordDensityBasedExtraction(
            cmd,
            classified,
            vulkanContext,
            sparsePipeline.pipeline_,      // Sparse pipeline
            mediumPipeline.pipeline_,      // Medium pipeline
            densePipeline.pipeline_,       // Dense pipeline
            mediumPipeline.pipelineLayout_,
            densityPC,
            mediumPipeline.descriptorSet_,
            profiler,
            sparsePipeline.descriptorSet_, // Pass sparse descriptor set
            densePipeline.descriptorSet_   // Pass dense descriptor set
        );
        
        if (profiler) {
            // profiler->endProfileRegion(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        }
        
        // Submit if we own the command buffer
        if (ownCommandBuffer) {
            endSingleTimeCommands(device, vulkanContext.getCommandPool(), vulkanContext.getQueue(), cmd);
        }
        
        extractionOutput.meshletCount = filterOutput.activeBlockCount; // Upper bound
        
    } catch (const std::exception& e) {
        std::cerr << "Error during density-based extraction: " << e.what() << std::endl;
        // Cleanup
        classified.cleanup(device);
        destroyBuffer(constantsUBO, device);
        MarchingCubesUtils::destroyMarchingCubesTables(mcTables, device);
        throw;
    }
    
    // Cleanup
    if (ownCommandBuffer) {
        // If we own the command buffer, it's already been submitted, so cleanup now
        classified.cleanup(device);
        destroyBuffer(constantsUBO, device);
        MarchingCubesUtils::destroyMarchingCubesTables(mcTables, device);
    } else {
        // Store for later cleanup (same as original)
        extractionOutput.tempResources.device = device;
        // Add the consolidated buffer to temporary resources
        extractionOutput.tempResources.addBuffer(classified.consolidatedBuffer);
        extractionOutput.tempResources.addBuffer(constantsUBO);
        // TODO: Update to use new MC table utils
        //extractionOutput.tempResources.addBuffer(mcTriTableBuffer);
        //extractionOutput.tempResources.addBuffer(mcEdgeTableBuffer);
        
        // Store pipeline resources
        extractionOutput.tempResources.addPipeline(mediumPipeline.pipeline_);
        extractionOutput.tempResources.addPipelineLayout(mediumPipeline.pipelineLayout_);
        extractionOutput.tempResources.addDescriptorSetLayout(mediumPipeline.descriptorSetLayout_);
        extractionOutput.tempResources.addDescriptorPool(mediumPipeline.getDescriptorPool());
        extractionOutput.tempResources.addShaderModule(mediumPipeline.getTaskShaderModule());
        extractionOutput.tempResources.addShaderModule(mediumPipeline.getMeshShaderModule());
        
        extractionOutput.tempResources.addPipeline(sparsePipeline.pipeline_);
        extractionOutput.tempResources.addShaderModule(sparsePipeline.getTaskShaderModule());
        extractionOutput.tempResources.addShaderModule(sparsePipeline.getMeshShaderModule());
        
        extractionOutput.tempResources.addPipeline(densePipeline.pipeline_);
        extractionOutput.tempResources.addShaderModule(densePipeline.getTaskShaderModule());
        extractionOutput.tempResources.addShaderModule(densePipeline.getMeshShaderModule());
        
        // Transfer ownership
        mediumPipeline.transferResourceOwnership();
        sparsePipeline.transferResourceOwnership();
        densePipeline.transferResourceOwnership();
    }
    
    std::cout << "--- Finished Density-Based Meshlet Extraction ---" << std::endl;
    return extractionOutput;
}