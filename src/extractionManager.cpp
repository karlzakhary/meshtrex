#include "extractionManager.h"
#include "common.h"

#include "vulkan_context.h"
#include "filteringOutput.h"
#include "extractionOutput.h"
#include "extractionPipeline.h"
#include "buffer.h"
#include "image.h"
#include "resources.h"
#include "vulkan_utils.h"
#include "mc_tables.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstring>

// --- Helper Function to Create Marching Cubes Table Buffer (Example) ---
// You'll need to provide the actual table data.
Buffer createMCTableBuffer(VulkanContext& context) {
    Buffer mcTableBuffer = {};
    // Use the triTable defined in mc_tables.h
    const int* tableData = *MarchingCubes::triTable;
    // Each of the 256 cases has 16 integer entries (vertex indices or -1)
    VkDeviceSize bufferSize = 256 * 16 * sizeof(int);
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(),
                 bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy table data to staging buffer
    memcpy(stagingBuffer.data, tableData, bufferSize);

    // Create device-local buffer for the table
    createBuffer(mcTableBuffer, context.getDevice(), context.getMemoryProperties(),
                 bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Copy from staging to device-local
    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    VkBufferCopy copyRegion = {0, 0, bufferSize};
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, mcTableBuffer.buffer, 1, &copyRegion);

    // Barrier to ensure transfer completes before shader access
    VkBufferMemoryBarrier2 transferCompleteBarrier = bufferBarrier(
        mcTableBuffer.buffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, // Assuming next use is Mesh Shader read
        0, VK_WHOLE_SIZE);
    pipelineBarrier(cmd, {}, 1, &transferCompleteBarrier, 0, {});

    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);

    // Cleanup staging buffer
    destroyBuffer(stagingBuffer, context.getDevice());

    std::cout << "Marching Cubes triangle table buffer created and uploaded." << std::endl;
    return mcTableBuffer;
}

struct ExtractionConstants {
    alignas(16) glm::uvec4 volumeDim;
    alignas(16) glm::uvec4 blockGridDim;
    alignas(4) float isovalue;
};

Buffer createConstantsUBO(VulkanContext& context, const FilteringOutput& filterOutput, PushConstants& pushConstants) {
    Buffer constantsUBO = {};
    VkDeviceSize bufferSize = sizeof(ExtractionConstants);

    createBuffer(constantsUBO, context.getDevice(), context.getMemoryProperties(),
                 bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Map and copy data
    ExtractionConstants constantsData = {};
    constantsData.volumeDim = pushConstants.volumeDim;
    constantsData.blockGridDim = pushConstants.blockGridDim;
    constantsData.isovalue = pushConstants.isovalue;

    memcpy(constantsUBO.data, &constantsData, bufferSize);

    std::cout << "Constants UBO created and updated." << std::endl;
    return constantsUBO;
}


// --- Main Extraction Function Implementation ---

// Returns populated ExtractionOutput struct. Throws on critical error.
ExtractionOutput extractMeshletDescriptors(VulkanContext& vulkanContext, FilteringOutput& filterOutput, PushConstants& pushConstants) {
    std::cout << "\n--- Starting Meshlet Extraction ---" << std::endl;
    if (filterOutput.activeBlockCount == 0) {
        std::cout << "No active blocks found. Skipping meshlet extraction." << std::endl;
        return {}; // Return empty output
    }

    VkDevice device = vulkanContext.getDevice();
    ExtractionPipeline extractionPipeline; // RAII object for pipeline state
    ExtractionOutput extractionOutput = {}; // RAII object for output buffers
    extractionOutput.device = device; // Store device for cleanup in destructor

    Buffer constantsUBO = {}; // Keep UBO handle for cleanup
    Buffer mcTableBuffer = {}; // Keep MC table handle for cleanup

    try {
        // 1. Setup Extraction Pipeline State
        if (!extractionPipeline.setup(device, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED)) {
            throw std::runtime_error("Failed to setup Extraction Pipeline.");
        }

        // 2. Create Output Buffers (Sizing is critical and heuristic)
        // Estimate max vertices: active_blocks * max_vertices_per_block (e.g., from paper ~128)
        // Estimate max primitives: active_blocks * max_primitives_per_block (e.g., ~248)
        // Estimate max indices: max_primitives * 3
        // Estimate max descriptors: active_blocks (potentially fewer if tasks merge)
        const VkDeviceSize counterSize = sizeof(uint32_t);
        constexpr uint32_t MAX_VERTS_PER_MESHLET_ESTIMATE = 128; // From paper/common practice
        constexpr uint32_t MAX_PRIMS_PER_MESHLET_ESTIMATE = 256; // Higher end estimate
        const VkDeviceSize MAX_TOTAL_VERTICES = static_cast<VkDeviceSize>(filterOutput.activeBlockCount) * MAX_VERTS_PER_MESHLET_ESTIMATE * sizeof(float) * 3; // Simplified: vec3 position
        const VkDeviceSize MAX_TOTAL_INDICES = static_cast<VkDeviceSize>(filterOutput.activeBlockCount) * MAX_PRIMS_PER_MESHLET_ESTIMATE * 3 * sizeof(uint32_t); // Assuming uint32 indices for meshlets
        const VkDeviceSize MAX_MESHLET_DESCRIPTORS = static_cast<VkDeviceSize>(filterOutput.activeBlockCount) * sizeof(MeshletDescriptor);

        std::cout << "Estimating output buffer sizes based on " << filterOutput.activeBlockCount << " active blocks:" << std::endl;
        std::cout << "  - Max Vertices: " << MAX_TOTAL_VERTICES << " bytes" << std::endl;
        std::cout << "  - Max Indices: " << MAX_TOTAL_INDICES << " bytes" << std::endl;
        std::cout << "  - Max Descriptors: " << MAX_MESHLET_DESCRIPTORS << " bytes" << std::endl;

        // Create buffers with STORAGE usage for shader writes
        createBuffer(extractionOutput.vertexBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_TOTAL_VERTICES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // Add VERTEX if needed directly by renderer
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        createBuffer(extractionOutput.indexBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_TOTAL_INDICES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // Add INDEX if needed directly by renderer
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        createBuffer(extractionOutput.meshletDescriptorBuffer, device, vulkanContext.getMemoryProperties(),
                     MAX_MESHLET_DESCRIPTORS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // 3. Create and Update Descriptors
        // Create UBO for constants (isovalue etc.) - Assuming Binding 0
        constantsUBO = createConstantsUBO(vulkanContext, filterOutput, pushConstants);
        mcTableBuffer = createMCTableBuffer(vulkanContext);


        // --- Update Descriptor Set ---
        if (extractionPipeline.descriptorSet_ == VK_NULL_HANDLE) {
            throw std::runtime_error("Extraction pipeline descriptor set is null.");
        }
        std::vector<VkWriteDescriptorSet> writes;
        VkDescriptorBufferInfo uboInfo = {constantsUBO.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo volInfo = {VK_NULL_HANDLE, filterOutput.volumeImage.imageView, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo blockIdInfo = {filterOutput.compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo mcTableInfo = {mcTableBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo vbInfo = {extractionOutput.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo ibInfo = {extractionOutput.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo descInfo = {extractionOutput.meshletDescriptorBuffer.buffer, 0, VK_WHOLE_SIZE};

        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipeline.descriptorSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipeline.descriptorSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &volInfo, nullptr, nullptr});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipeline.descriptorSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &blockIdInfo, nullptr});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipeline.descriptorSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mcTableInfo, nullptr});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipeline.descriptorSet_, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vbInfo, nullptr});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipeline.descriptorSet_, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ibInfo, nullptr});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, extractionPipeline.descriptorSet_, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &descInfo, nullptr});

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        std::cout << "Extraction pipeline descriptor set updated." << std::endl;

        // 4. Record Command Buffer
        VkCommandBuffer cmd = beginSingleTimeCommands(device, vulkanContext.getCommandPool());

        // --- Barriers Before Extraction ---
        std::vector<VkBufferMemoryBarrier2> fillToComputeBarriers;

        // Zero out the first uint (counter) in each output buffer
        vkCmdFillBuffer(cmd, extractionOutput.vertexBuffer.buffer, 0, counterSize, 0);
        fillToComputeBarriers.push_back(bufferBarrier(
            extractionOutput.vertexBuffer.buffer,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            0, counterSize // Barrier only for the counter part
        ));

        vkCmdFillBuffer(cmd, extractionOutput.indexBuffer.buffer, 0, counterSize, 0);
        fillToComputeBarriers.push_back(bufferBarrier(
           extractionOutput.indexBuffer.buffer,
           VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
           VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
           0, counterSize
       ));

        vkCmdFillBuffer(cmd, extractionOutput.meshletDescriptorBuffer.buffer, 0, counterSize, 0);
        fillToComputeBarriers.push_back(bufferBarrier(
           extractionOutput.meshletDescriptorBuffer.buffer,
           VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
           VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
           0, counterSize
       ));

        // Ensure fill operations complete before shaders access the counters
        pipelineBarrier(cmd, {}, fillToComputeBarriers.size(), fillToComputeBarriers.data(), 0, {});
        std::cout << "Atomic counters initialized." << std::endl;

        std::vector<VkBufferMemoryBarrier2> preBufferBarriers;
        std::vector<VkImageMemoryBarrier2> preImageBarriers;

        // Input Buffers/Images need to be readable by compute/task/mesh stages
        preBufferBarriers.push_back(bufferBarrier(
            filterOutput.compactedBlockIdBuffer.buffer,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, // Assuming last op was copy/transfer read in filteringManager
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            0, VK_WHOLE_SIZE));
        preImageBarriers.push_back(imageBarrier(
            filterOutput.volumeImage.image,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, // Assuming last use was compute read
             VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, // Keep general layout
            VK_IMAGE_ASPECT_COLOR_BIT));
        preBufferBarriers.push_back(bufferBarrier(
                mcTableBuffer.buffer,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, // Assuming last op was transfer write
                VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                0, VK_WHOLE_SIZE));

        // Output Buffers need to be writable by mesh shader
        preBufferBarriers.push_back(bufferBarrier(
            extractionOutput.vertexBuffer.buffer,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            0, VK_WHOLE_SIZE));
        preBufferBarriers.push_back(bufferBarrier(
            extractionOutput.indexBuffer.buffer,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            0, VK_WHOLE_SIZE));
        preBufferBarriers.push_back(bufferBarrier(
            extractionOutput.meshletDescriptorBuffer.buffer,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            0, VK_WHOLE_SIZE));


        pipelineBarrier(cmd, {}, preBufferBarriers.size(), preBufferBarriers.data(), preImageBarriers.size(), preImageBarriers.data());

        // --- Bind Pipeline & Descriptors ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipeline.pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, extractionPipeline.pipelineLayout_, 0, 1, &extractionPipeline.descriptorSet_, 0, nullptr);

        // --- Dispatch ---
        // Dispatch one task workgroup per active block found in the previous stage
        uint32_t taskCount = filterOutput.activeBlockCount;
        std::cout << "Dispatching " << taskCount << " mesh tasks..." << std::endl;
        vkCmdDrawMeshTasksEXT(cmd, taskCount, 1, 1); // One task per active block ID

        // --- Barriers After Extraction ---
        // Ensure mesh shader writes are finished before subsequent reads (e.g., rendering or copy)
        std::vector<VkBufferMemoryBarrier2> postBufferBarriers;
        VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        VkAccessFlags2 dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
        postBufferBarriers.push_back(bufferBarrier(extractionOutput.vertexBuffer.buffer, VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, dstStageMask, dstAccessMask, 0, VK_WHOLE_SIZE));
        postBufferBarriers.push_back(bufferBarrier(extractionOutput.indexBuffer.buffer, VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, dstStageMask, dstAccessMask, 0, VK_WHOLE_SIZE));
        postBufferBarriers.push_back(bufferBarrier(extractionOutput.meshletDescriptorBuffer.buffer, VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, dstStageMask, dstAccessMask, 0, VK_WHOLE_SIZE));

        pipelineBarrier(cmd, {}, postBufferBarriers.size(), postBufferBarriers.data(), 0, {});



         pipelineBarrier(cmd, {}, postBufferBarriers.size(), postBufferBarriers.data(), 0, {});


        // 5. Submit and Wait
        endSingleTimeCommands(device, vulkanContext.getCommandPool(), vulkanContext.getQueue(), cmd);
        VK_CHECK(vkDeviceWaitIdle(device)); // Ensure GPU is finished

        std::cout << "Meshlet extraction command buffer submitted and executed." << std::endl;

        // Optionally read back meshlet count if needed (requires another buffer + atomic in shader)
        // extractionOutput.meshletCount = readBackCountBuffer(...);

    } catch (const std::exception& e) {
        std::cerr << "Error during meshlet extraction: " << e.what() << std::endl;
        // Cleanup partially created resources
        extractionPipeline.cleanup(); // Uses RAII
        extractionOutput.cleanup();   // Uses RAII
        destroyBuffer(constantsUBO, device); // Manual cleanup for UBO
        destroyBuffer(mcTableBuffer, device); // Manual cleanup for MC table
        throw; // Re-throw the exception
    }

     // --- Cleanup Temporary Resources ---
     // UBO and MC Table Buffer are temporary for this stage
     destroyBuffer(constantsUBO, device);
     destroyBuffer(mcTableBuffer, device);

    // extractionPipeline goes out of scope and cleans itself up via RAII destructor

    std::cout << "--- Finished Meshlet Extraction ---" << std::endl;

    // extractionOutput now holds the results (buffers). Its destructor will clean them up when it goes out of scope.
    // If you need to transfer ownership, use std::move.
    return extractionOutput;
}