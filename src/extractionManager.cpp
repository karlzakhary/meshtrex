#include "extractionManager.h"
#include "common.h"

#include "vulkan_context.h"
#include "filteringOutput.h"
#include "extractionOutput.h"
#include "extractionPipeline.h"
#include "buffer.h"
#include "image.h"
#include "resources.h"
#include "gpuProfiler.h"
#include "vulkan_utils.h"
#include "mc_tables.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <utility> // For std::move if needed later

// --- Helper Function to Create Marching Cubes Table Buffer (Revised) ---
struct MCTableUploadResult {
    Buffer deviceBuffer;
    Buffer stagingBuffer;  // Needs cleanup after command buffer submission
};

MCTableUploadResult createTriTableBuffer(VulkanContext& context, VkCommandBuffer externalCmd = VK_NULL_HANDLE) {
    MCTableUploadResult result = {};
    const int* triTableData = &MarchingCubes::triTable[0][0];

    VkDeviceSize triTableSize = 256 * 16 * sizeof(int);

    createBuffer(result.stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 triTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
     // Check if staging buffer creation succeeded
     if (result.stagingBuffer.buffer == VK_NULL_HANDLE) {
         throw std::runtime_error("Failed to create staging buffer for MC TriTable.");
     }
     if (result.stagingBuffer.data == nullptr) {
          destroyBuffer(result.stagingBuffer, context.getDevice()); // Clean up before throwing
          throw std::runtime_error("Failed to map staging buffer for MC TriTable.");
     }

    // Copy table data to staging buffer
    memcpy(result.stagingBuffer.data, triTableData, triTableSize); // Use correct pointer

    // Create device-local buffer for the table
    createBuffer(result.deviceBuffer, context.getDevice(), context.getMemoryProperties(),
                 triTableSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
     // Check if device buffer creation succeeded
      if (result.deviceBuffer.buffer == VK_NULL_HANDLE) {
          destroyBuffer(result.stagingBuffer, context.getDevice()); // Clean up staging buffer
          throw std::runtime_error("Failed to create device buffer for MC TriTable.");
      }

    // Use external command buffer if provided, otherwise create our own
    VkCommandBuffer cmd;
    bool ownCommandBuffer = (externalCmd == VK_NULL_HANDLE);
    
    if (ownCommandBuffer) {
        cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    } else {
        cmd = externalCmd;
    }
    
    VkBufferCopy copyRegion = {0, 0, triTableSize};
    vkCmdCopyBuffer(cmd, result.stagingBuffer.buffer, result.deviceBuffer.buffer, 1, &copyRegion);

    // Barrier to ensure transfer completes before shader access
    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        result.deviceBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, // Task & Mesh read
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    if (ownCommandBuffer) {
        endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
        // Cleanup staging buffer immediately since we submitted the command
        destroyBuffer(result.stagingBuffer, context.getDevice());
        result.stagingBuffer = {}; // Clear to indicate it's been cleaned up
    }
    // If using external command buffer, staging buffer cleanup is caller's responsibility

    return result;
}

MCTableUploadResult createEdgeTableBuffer(VulkanContext& context, VkCommandBuffer externalCmd = VK_NULL_HANDLE) {
    MCTableUploadResult result = {};
    const int* edgeTableData = &MarchingCubes::edgeTable[0];

    VkDeviceSize edgeTableSize = 256 * sizeof(int);

    createBuffer(result.stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 edgeTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
     // Check if staging buffer creation succeeded
     if (result.stagingBuffer.buffer == VK_NULL_HANDLE) {
         throw std::runtime_error("Failed to create staging buffer for MC EdgeTable.");
     }
     if (result.stagingBuffer.data == nullptr) {
          destroyBuffer(result.stagingBuffer, context.getDevice()); // Clean up before throwing
          throw std::runtime_error("Failed to map staging buffer for MC EdgeTable.");
     }

    // Copy table data to staging buffer
    memcpy(result.stagingBuffer.data, edgeTableData, edgeTableSize); // Use correct pointer

    // Create device-local buffer for the table
    createBuffer(result.deviceBuffer, context.getDevice(), context.getMemoryProperties(),
                 edgeTableSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
     // Check if device buffer creation succeeded
      if (result.deviceBuffer.buffer == VK_NULL_HANDLE) {
          destroyBuffer(result.stagingBuffer, context.getDevice()); // Clean up staging buffer
          throw std::runtime_error("Failed to create device buffer for MC EdgeTable.");
      }

    // Use external command buffer if provided, otherwise create our own
    VkCommandBuffer cmd;
    bool ownCommandBuffer = (externalCmd == VK_NULL_HANDLE);
    
    if (ownCommandBuffer) {
        cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    } else {
        cmd = externalCmd;
    }
    
    VkBufferCopy copyRegion = {0, 0, edgeTableSize};
    vkCmdCopyBuffer(cmd, result.stagingBuffer.buffer, result.deviceBuffer.buffer, 1, &copyRegion);

    // Barrier to ensure transfer completes before shader access
    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        result.deviceBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, // Task & Mesh read
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    if (ownCommandBuffer) {
        endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
        // Cleanup staging buffer immediately since we submitted the command
        destroyBuffer(result.stagingBuffer, context.getDevice());
        result.stagingBuffer = {}; // Clear to indicate it's been cleaned up
    }
    // If using external command buffer, staging buffer cleanup is caller's responsibility

    return result;
}

MCTableUploadResult createNumVerticesBuffer(VulkanContext& context, VkCommandBuffer externalCmd = VK_NULL_HANDLE) {
    MCTableUploadResult result = {};
    const int* numVerticesData = &MarchingCubes::numVerticesTable[0];

    VkDeviceSize numVerticesSize = 256 * sizeof(int);

    createBuffer(result.stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 numVerticesSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
     // Check if staging buffer creation succeeded
     if (result.stagingBuffer.buffer == VK_NULL_HANDLE) {
         throw std::runtime_error("Failed to create staging buffer for MC NumVertices.");
     }
     if (result.stagingBuffer.data == nullptr) {
          destroyBuffer(result.stagingBuffer, context.getDevice()); // Clean up before throwing
          throw std::runtime_error("Failed to map staging buffer for MC NumVertices.");
     }

    // Copy table data to staging buffer
    memcpy(result.stagingBuffer.data, numVerticesData, numVerticesSize); // Use correct pointer

    // Create device-local buffer for the table
    createBuffer(result.deviceBuffer, context.getDevice(), context.getMemoryProperties(),
                 numVerticesSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
     // Check if device buffer creation succeeded
      if (result.deviceBuffer.buffer == VK_NULL_HANDLE) {
          destroyBuffer(result.stagingBuffer, context.getDevice()); // Clean up staging buffer
          throw std::runtime_error("Failed to create device buffer for MC NumVertices.");
      }

    // Use external command buffer if provided, otherwise create our own
    VkCommandBuffer cmd;
    bool ownCommandBuffer = (externalCmd == VK_NULL_HANDLE);
    
    if (ownCommandBuffer) {
        cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    } else {
        cmd = externalCmd;
    }
    
    VkBufferCopy copyRegion = {0, 0, numVerticesSize};
    vkCmdCopyBuffer(cmd, result.stagingBuffer.buffer, result.deviceBuffer.buffer, 1, &copyRegion);

    // Barrier to ensure transfer completes before shader access
    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        result.deviceBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, // Task & Mesh read
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    if (ownCommandBuffer) {
        endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
        // Cleanup staging buffer immediately since we submitted the command
        destroyBuffer(result.stagingBuffer, context.getDevice());
        result.stagingBuffer = {}; // Clear to indicate it's been cleaned up
    }
    // If using external command buffer, staging buffer cleanup is caller's responsibility

    std::cout << "Marching Cubes NumVertices buffer created and uploaded." << std::endl;
    return result;
}

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
                                          VkCommandBuffer externalCmd, GPUProfiler* profiler) {
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
    Buffer mcTriTableBuffer = {};
    Buffer mcEdgeTableBuffer = {};
    MCTableUploadResult triTableResult = {};
    MCTableUploadResult edgeTableResult = {};
    
    bool ownCommandBuffer = (externalCmd == VK_NULL_HANDLE);

    try {
        // 1. Setup Extraction Pipeline State
        // Ensure formats passed are compatible with device/swapchain if validation complains
        if (!extractionPipeline.setup(device, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_D32_SFLOAT,
                                     pushConstants.blockDim.x, pushConstants.blockDim.y, pushConstants.blockDim.z)) {
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
        
        // Pass the external command buffer to avoid premature staging buffer cleanup
        triTableResult = createTriTableBuffer(vulkanContext, externalCmd);
        mcTriTableBuffer = triTableResult.deviceBuffer;
        if (mcTriTableBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create MC triangle table buffer."); }
        
        edgeTableResult = createEdgeTableBuffer(vulkanContext, externalCmd);
        mcEdgeTableBuffer = edgeTableResult.deviceBuffer;
        if (mcEdgeTableBuffer.buffer == VK_NULL_HANDLE) { throw std::runtime_error("Failed to create MC edge table buffer."); }

        // 4. Update Descriptors
        if (extractionPipeline.descriptorSet_ == VK_NULL_HANDLE) { throw std::runtime_error("Extraction pipeline descriptor set is null."); }
        std::vector<VkWriteDescriptorSet> writes;
        VkDescriptorBufferInfo uboInfo = {constantsUBO.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo volInfo = {VK_NULL_HANDLE, minMaxOutput.volumeImage.imageView, VK_IMAGE_LAYOUT_GENERAL}; // Assuming GENERAL layout from filtering
        VkDescriptorBufferInfo blockCountInfo = {filterOutput.activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo blockIdInfo = {filterOutput.compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo mcTriTableInfo = {mcTriTableBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo mcEdgeTableInfo = {mcEdgeTableBuffer.buffer, 0, VK_WHOLE_SIZE};
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
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            4,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &mcTriTableInfo,
            nullptr
        }); // Binding 4: MC Triangle Table
        writes.push_back({
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            extractionPipeline.descriptorSet_,
            5,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &mcEdgeTableInfo,
            nullptr
        }); // Binding 5: MC Edge Table
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
        preBufferBarriers.push_back(bufferBarrier(mcTriTableBuffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, EXTRACTION_SHADER_STAGES, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, 0, VK_WHOLE_SIZE));
        preBufferBarriers.push_back(bufferBarrier(mcEdgeTableBuffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, EXTRACTION_SHADER_STAGES, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, 0, VK_WHOLE_SIZE));

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
        destroyBuffer(mcTriTableBuffer, device); // Safe to call even if null
        destroyBuffer(mcEdgeTableBuffer, device);
        throw;
    }

    // Cleanup temporary UBO and MC table buffer
    // RAII handles extractionPipeline and extractionOutput cleanup
    if (ownCommandBuffer) {
        // Only destroy buffers if we own the command buffer (already submitted)
        destroyBuffer(constantsUBO, device);
        destroyBuffer(mcTriTableBuffer, device);
        destroyBuffer(mcEdgeTableBuffer, device);
    } else {
        
        // Store temporary buffers for later cleanup
        extractionOutput.tempResources.device = device;
        extractionOutput.tempResources.addBuffer(constantsUBO);
        extractionOutput.tempResources.addBuffer(mcTriTableBuffer);
        extractionOutput.tempResources.addBuffer(mcEdgeTableBuffer);
        
        // Also add the staging buffers from MC table uploads if they exist
        if (triTableResult.stagingBuffer.buffer != VK_NULL_HANDLE) {
            extractionOutput.tempResources.addBuffer(triTableResult.stagingBuffer);
        }
        if (edgeTableResult.stagingBuffer.buffer != VK_NULL_HANDLE) {
            extractionOutput.tempResources.addBuffer(edgeTableResult.stagingBuffer);
        }
        
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