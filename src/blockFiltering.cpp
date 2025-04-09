#include "common.h"
#include "blockFiltering.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <iostream>
#include <vector>
#include <stdexcept>

#include "buffer.h"
#include "device.h"
#include "mesh.h"
#include "resources.h"
#include "shaders.h"
#include "vulkan_utils.h"
#include "volume.h"

struct alignas(16) PushConstants {
    glm::uvec4 volumeDim;    // Offset 0, Size 16. Shader uses .xyz
    glm::uvec4 blockDim;     // Offset 16, Size 16. Shader uses .xyz
    glm::uvec4 blockGridDim; // Offset 32, Size 16. Shader uses .xyz
    float      isovalue;     // Offset 48, Size 4.
};

std::tuple<VkPipelineLayout, VkDescriptorSetLayout> createComputeMinMaxPipelineLayout(
    VkDevice device)
{
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants)
    };

    VkDescriptorSetLayoutBinding bindings[2] = {};

    // Binding 0: Input Volume Image (Read-Only in shader)
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, // Shader reads via imageLoad
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    // Binding 1: Output MinMax Image (Write-Only in shader)
    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = 2,
        .pBindings = bindings
    };

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1,
    pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout layout = nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &layout));

    return std::make_tuple(layout, descriptorSetLayout);
}

std::tuple<VkPipelineLayout, VkDescriptorSetLayout> createComputeOccupiedBlockFilteringPipelineLayout(
    VkDevice device)
{
    // This layout needs bindings for:
    // binding 0: input minMaxImage (Storage Image)
    // binding 1: output compactedBlockIdBuffer (Storage Buffer)
    // binding 2: output activeBlockCountBuffer (Storage Buffer)
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; // MinMax Input
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Compacted IDs Output
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Atomic Counter Output
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

    VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    return std::make_tuple(pipelineLayout, descriptorSetLayout);
}


void pushMinMaxDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
                        VkImageView inputVolumeView, // Input Volume Image
                        VkImageView outputMinMaxView) // Output MinMax Image
{
    VkDescriptorImageInfo inputImageInfo = {};
    inputImageInfo.sampler = VK_NULL_HANDLE; // Not used for storage image
    inputImageInfo.imageView = inputVolumeView;
    inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // Shader reads via imageLoad

    VkDescriptorImageInfo outputImageInfo = {};
    outputImageInfo.sampler = VK_NULL_HANDLE; // Not used for storage image
    outputImageInfo.imageView = outputMinMaxView;
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // Shader writes via imageStore

    VkWriteDescriptorSet writes[2] = {};

    // Binding 0: Input Volume Image
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &inputImageInfo;

    // Binding 1: Output MinMax Image
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &outputImageInfo;

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout, 0, 2, writes);
}

void pushOccupiedBlockFilteringDescriptorSets(
    VkCommandBuffer cmd,
    VkImageView minMaxImage,
    const PushConstants& pushData,   // Contains dimensions, isovalue etc.
    Buffer& compactedBlockIdBuffer,  // Output: Buffer to store active block IDs
    Buffer& activeBlockCountBuffer,
    VkPipelineLayout pipelineLayout)
{
    VkDescriptorImageInfo minMaxImageInfo = {};
    minMaxImageInfo.imageView = minMaxImage;
    minMaxImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // Must match layout from previous pass

    VkDescriptorBufferInfo compactedIdBufferInfo = {compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo countBufferInfo = {activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &minMaxImageInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &compactedIdBufferInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &countBufferInfo;

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 3, writes);

    // Push Constants (including isovalue)
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushData);
}

// --- Function to set up and run the filtering pass ---
// Returns the total count of active blocks found.
uint32_t runOccupiedBlockFiltering(
    char** argv,
    VkDevice device,
    VkQueue computeQueue,
    VkCommandPool commandPool,
    VkPhysicalDeviceMemoryProperties memoryProperties,
    const Image& minMaxImage,        // Input: Result from the previous MinMax pass
    const PushConstants& pushData,   // Contains dimensions, isovalue etc.
    Buffer& compactedBlockIdBuffer,  // Output: Buffer to store active block IDs
    Buffer& activeBlockCountBuffer) // Output: Buffer to store the count
{
    std::cout << "\nSetting up Occupied Block Filtering pass..." << std::endl;

    // --- Create Output Buffers ---
    uint32_t totalBlocks = pushData.blockGridDim.x * pushData.blockGridDim.y * pushData.blockGridDim.z;

    // 1. Compacted Block ID Buffer (worst case size: all blocks active)
    VkDeviceSize compactedBufferSize = totalBlocks * sizeof(uint32_t); // Store uint IDs
    createBuffer(compactedBlockIdBuffer, device, memoryProperties,
                 compactedBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // Allow clearing
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // 2. Active Block Count Buffer (single uint)
    VkDeviceSize countBufferSize = sizeof(uint32_t);
    createBuffer(activeBlockCountBuffer, device, memoryProperties,
                 countBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // Allow clearing & readback
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); // Can be device local initially

    // --- Initialize Active Block Count Buffer to 0 ---
    VkCommandBuffer initCmd = beginSingleTimeCommands(device, commandPool);
    vkCmdFillBuffer(initCmd, activeBlockCountBuffer.buffer, 0, countBufferSize, 0);
    // Add barrier if needed before compute shader uses it
     VkBufferMemoryBarrier2 countBufBarrier = {};
     countBufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
     countBufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT; // Or TRANSFER_BIT if using vkCmdUpdateBuffer/Copy
     countBufBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
     countBufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
     countBufBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT; // Atomic access
     countBufBarrier.buffer = activeBlockCountBuffer.buffer;
     countBufBarrier.offset = 0;
     countBufBarrier.size = VK_WHOLE_SIZE;
     VkDependencyInfo countDepInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
     countDepInfo.bufferMemoryBarrierCount = 1;
     countDepInfo.pBufferMemoryBarriers = &countBufBarrier;
     vkCmdPipelineBarrier2(initCmd, &countDepInfo);
    endSingleTimeCommands(device, commandPool, computeQueue, initCmd);
    std::cout << "Output buffers created and count initialized." << std::endl;
    auto [pipelineLayout, setLayout] = createComputeOccupiedBlockFilteringPipelineLayout(device);

    // --- Load Shader & Create Pipeline ---
    Shader filterCS{};
    assert(loadShader(filterCS, device, argv[0], "spirv/occupiedBlockFiltering.comp.spv"));
    VkPipeline computePipeline = createComputePipeline(device, nullptr, filterCS, pipelineLayout);
    std::cout << "Filtering pipeline created." << std::endl;


    // --- Dispatch ---
    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

    pushOccupiedBlockFilteringDescriptorSets(
        cmd,
        minMaxImage.imageView,
        pushData,
        compactedBlockIdBuffer,
        activeBlockCountBuffer,
        pipelineLayout
    );
    // Calculate dispatch size (1D)
    uint32_t localSizeX = 128; // Must match shader's local_size_x
    uint32_t groupCountX = (totalBlocks + localSizeX - 1) / localSizeX;
    std::cout << "Dispatching " << groupCountX << " workgroups (" << totalBlocks << " total blocks)..." << std::endl;
    vkCmdDispatch(cmd, groupCountX, 1, 1);

    // Barrier: Ensure shader writes (especially to count buffer) are finished before potential readback
    VkBufferMemoryBarrier2 countReadBarrier = {};
    countReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    countReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    countReadBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT; // Shader writes/atomic ops
    countReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; // Prepare for potential copy/readback
    countReadBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    countReadBarrier.buffer = activeBlockCountBuffer.buffer;
    countReadBarrier.offset = 0;
    countReadBarrier.size = VK_WHOLE_SIZE;
    // Add similar barrier for compactedBlockIdBuffer if reading it back immediately

    VkDependencyInfo readDepInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    readDepInfo.bufferMemoryBarrierCount = 1;
    readDepInfo.pBufferMemoryBarriers = &countReadBarrier;
    vkCmdPipelineBarrier2(cmd, &readDepInfo);


    endSingleTimeCommands(device, commandPool, computeQueue, cmd);
    VK_CHECK(vkDeviceWaitIdle(device)); // Ensure completion before reading count

    // --- Read back the count ---
    uint32_t activeCount = 0;
    // Create a host-visible buffer
    Buffer countReadbackBuffer = {};
    createBuffer(countReadbackBuffer, device, memoryProperties, countBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Copy device buffer to host buffer
    cmd = beginSingleTimeCommands(device, commandPool);
    VkBufferCopy region = {0, 0, countBufferSize};
    vkCmdCopyBuffer(cmd, activeBlockCountBuffer.buffer, countReadbackBuffer.buffer, 1, &region);
    endSingleTimeCommands(device, commandPool, computeQueue, cmd);
    VK_CHECK(vkDeviceWaitIdle(device));
    // Read from mapped pointer
    memcpy(&activeCount, countReadbackBuffer.data, sizeof(uint32_t));
    destroyBuffer(countReadbackBuffer, device); // Clean up staging buffer

    std::cout << "Occupied Block Filtering finished. Active blocks found: " << activeCount << std::endl;

    // --- Cleanup ---
    // Destroy pipeline, layout, shader module, descriptor set layout
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyShaderModule(device, filterCS.module, nullptr);
    // Buffers (compactedBlockIdBuffer, activeBlockCountBuffer) are kept as output

    return activeCount;
}

void uploadVolumeImage(VkCommandBuffer commandBuffer,
                       VkImage volumeImage, Buffer stagingBuffer, VkExtent3D extent) {
    transitionImage(commandBuffer, volumeImage,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
    );
    // copy1DBufferTo3DImage should copy from stagingBuffer.buffer to volumeImage
    copy1DBufferTo3DImage(stagingBuffer, commandBuffer, volumeImage, extent.width, extent.height, extent.depth);
    // Transition to GENERAL for shader access (read/write)
    transitionImage(commandBuffer, volumeImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );
}


// NEW Function: Reads back the MinMax data from a VkImage via a staging buffer
std::vector<MinMaxResult> mapMinMaxImage(VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
                                         VkCommandPool commandPool, VkQueue queue,
                                         const Image& minMaxImage,
                                         VkExtent3D gridExtent,
                                         VkDeviceSize minMaxTotalBytes)
{
    std::cout << "\nReading back GPU MinMax Image results..." << std::endl;

    // 1. Create a host-visible staging buffer for readback
    Buffer readbackBuffer = {};
    createBuffer(readbackBuffer, device, memoryProperties, minMaxTotalBytes,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, // Destination for image copy
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // 2. Copy data from GPU image to staging buffer
    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);

    // Transition image layout for copying
    transitionImage(cmd, minMaxImage.image,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

    // Setup copy command
    copy3DImageTo1DBuffer(readbackBuffer, cmd, minMaxImage.image, gridExtent);

    // Transition image layout back to GENERAL (or whatever is needed next)
    transitionImage(cmd, minMaxImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

    endSingleTimeCommands(device, commandPool, queue, cmd); // Submits and waits

    // 3. Access the mapped pointer of the readback buffer
    void* mappedData = readbackBuffer.data;
    if (mappedData == nullptr) {
        destroyBuffer(readbackBuffer, device);
        throw std::runtime_error("Readback buffer is not mapped!");
    }

    // 4. Copy data from mapped buffer to CPU vector
    size_t numElements = minMaxTotalBytes / sizeof(MinMaxResult);
    // ... (Add validation checks for size/count as before if desired) ...
    size_t expectedCount = static_cast<size_t>(gridExtent.width) * gridExtent.height * gridExtent.depth;
     if (numElements != expectedCount) {
          std::cerr << "Warning: Number of elements in buffer (" << numElements
                   << ") does not match expected block count (" << expectedCount << ")" << std::endl;
          numElements = std::min(numElements, expectedCount);
     }

    std::vector<MinMaxResult> results(numElements);

    // No vkInvalidateMappedMemoryRanges needed due to HOST_COHERENT
    memcpy(results.data(), mappedData, numElements * sizeof(MinMaxResult));

    destroyBuffer(readbackBuffer, device);
    std::cout << "GPU image results readback complete." << std::endl;

    uint32_t totalBlocks = numElements;
    std::cout << "GPU Results (first few blocks):" << std::endl;
    for (uint32_t i = 0; i < std::min(totalBlocks, 10u); ++i) {
        std::cout << "Block[" << i << "] Min: " << results[i].minVal
                  << " Max: " << results[i].maxVal << std::endl;
    }
     if (totalBlocks > 10u) {
        std::cout << "... (" << totalBlocks << " total blocks read)" << std::endl;
     }

    return results;
}

// Helper function to read back the simple test output buffer
// Very similar to mapMinMaxImage's readback part, but reads a buffer.
std::vector<MinMaxResult> mapTestResultBuffer(VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
                                             VkCommandPool commandPool, VkQueue queue,
                                             const Buffer& testOutputBuffer, VkDeviceSize bufferSize, size_t expectedElements)
{
    std::cout << "\nReading back test shader output buffer..." << std::endl;
    Buffer readbackBuffer = {};
    createBuffer(readbackBuffer, device, memoryProperties, bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);
    VkBufferCopy region = {0, 0, bufferSize};
    vkCmdCopyBuffer(cmd, testOutputBuffer.buffer, readbackBuffer.buffer, 1, &region);
    endSingleTimeCommands(device, commandPool, queue, cmd);
    VK_CHECK(vkDeviceWaitIdle(device));

    void* mappedData = readbackBuffer.data;
    if (mappedData == nullptr) {
        destroyBuffer(readbackBuffer, device);
        throw std::runtime_error("Test readback buffer is not mapped!");
    }

    size_t numElements = bufferSize / sizeof(MinMaxResult);
    if (numElements != expectedElements) {
         std::cerr << "Warning: Test buffer element count (" << numElements
                   << ") does not match expected block count (" << expectedElements << ")" << std::endl;
         numElements = std::min(numElements, expectedElements);
    }

    std::vector<MinMaxResult> results(numElements);
    memcpy(results.data(), mappedData, numElements * sizeof(MinMaxResult));

    // destroyBuffer(readbackBuffer, device);
    std::cout << "Test buffer readback complete." << std::endl;
    return results;
}


/**
 * @brief Runs a test compute shader to verify reading from the minMaxImage.
 *
 * @param device Logical Vulkan device.
 * @param computeQueue Queue for compute and transfer.
 * @param commandPool Command pool for recording commands.
 * @param memoryProperties Physical device memory properties.
 * @param minMaxImage The VkImage containing min/max data (output from first pass).
 * @param pushData Push constants containing grid dimensions.
 * @return true if the test shader's output matches a direct readback of the image, false otherwise.
 */
bool testMinMaxReadback(
    char** argv,
    VkDevice device,
    VkQueue computeQueue,
    VkCommandPool commandPool,
    VkPhysicalDeviceMemoryProperties memoryProperties,
    const Image& minMaxImage,
    const PushConstants& pushData) // Only blockGridDim is strictly needed here
{
    std::cout << "\n--- Starting MinMax Image Read Test ---" << std::endl;
    bool success = false;

    VkPipeline testPipeline = VK_NULL_HANDLE;
    VkPipelineLayout testPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout testSetLayout = VK_NULL_HANDLE;
    Shader testCS = {};
    Buffer testOutputBuffer = {}; // Buffer to store results from test shader

    try {
        // --- Calculate Sizes ---
        VkExtent3D gridExtent = {pushData.blockGridDim.x, pushData.blockGridDim.y, pushData.blockGridDim.z};
        uint32_t totalBlocks = gridExtent.width * gridExtent.height * gridExtent.depth;
        VkDeviceSize outputBufferSize = totalBlocks * sizeof(MinMaxResult); // uvec2 size

        // --- Create Test Output Buffer ---
        createBuffer(testOutputBuffer, device, memoryProperties, outputBufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // Shader writes, we read back
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // --- Create Test Pipeline Layout & Descriptor Set Layout ---
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0; // Input: minMaxImage
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1; // Output: testOutputBuffer
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &testSetLayout));

        VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &testSetLayout;
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &testPipelineLayout));

        // --- Load Test Shader & Create Test Pipeline ---
        // Ensure TestMinMaxRead.comp is compiled to SPIR-V
        assert(loadShader(testCS, device, /* path_prefix? */ argv[0], "spirv/testReadingMinMax.comp.spv"));
        testPipeline = createComputePipeline(device, nullptr, testCS, testPipelineLayout);
        assert(testPipeline != VK_NULL_HANDLE);
        std::cout << "Test pipeline created." << std::endl;

        // --- Dispatch Test Shader ---
        VkCommandBuffer cmd = beginSingleTimeCommands(device, commandPool);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, testPipeline);

        // Push Descriptors
        VkDescriptorImageInfo minMaxImageInfo = {VK_NULL_HANDLE, minMaxImage.imageView, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo outputBufferInfo = {testOutputBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &minMaxImageInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &outputBufferInfo, nullptr};
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, testPipelineLayout, 0, 2, writes);

        // Push Constants (only blockGridDim needed by test shader)
        vkCmdPushConstants(cmd, testPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushData);

        // Calculate dispatch size
        uint32_t localSizeX = 128; // Must match test shader's local_size_x
        uint32_t groupCountX = (totalBlocks + localSizeX - 1) / localSizeX;
        vkCmdDispatch(cmd, groupCountX, 1, 1);

        // Barrier: Ensure test shader writes to testOutputBuffer are done before readback
        VkBufferMemoryBarrier2 testOutputBarrier = {};
        testOutputBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        testOutputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        testOutputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        testOutputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; // Prepare for copy in readback
        testOutputBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        testOutputBarrier.buffer = testOutputBuffer.buffer;
        testOutputBarrier.size = VK_WHOLE_SIZE;
        VkDependencyInfo testDepInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        testDepInfo.bufferMemoryBarrierCount = 1;
        testDepInfo.pBufferMemoryBarriers = &testOutputBarrier;
        vkCmdPipelineBarrier2(cmd, &testDepInfo);

        endSingleTimeCommands(device, commandPool, computeQueue, cmd);
        VK_CHECK(vkDeviceWaitIdle(device)); // Wait for test shader to finish
        std::cout << "Test shader dispatched and finished." << std::endl;
        vkResetCommandPool(device, commandPool, 0);
        // 2. Read back results directly from the minMaxImage (using existing function)
        std::vector<MinMaxResult> directImageResults = mapMinMaxImage(
            device, memoryProperties, commandPool, computeQueue,
            minMaxImage, gridExtent, outputBufferSize // outputBufferSize is same as minMaxTotalBytes
        );

        // --- Verification ---
        // 1. Read back results from the test shader's output buffer
        std::vector<MinMaxResult> testShaderResults = mapTestResultBuffer(
            device, memoryProperties, commandPool, computeQueue,
            testOutputBuffer, outputBufferSize, totalBlocks
        );


        // 3. Compare
        std::cout << "Comparing test shader output vs direct image readback..." << std::endl;
        if (testShaderResults.size() != directImageResults.size()) {
            std::cerr << "Error: Size mismatch during verification! Test shader output size: "
                      << testShaderResults.size() << ", Direct image readback size: "
                      << directImageResults.size() << std::endl;
        } else if (testShaderResults.empty()) {
             std::cout << "Verification skipped: Result vectors are empty." << std::endl;
             success = true; // Or handle as error?
        }
         else {
            int mismatches = 0;
            for (size_t i = 0; i < testShaderResults.size(); ++i) {
                if (testShaderResults[i].minVal != directImageResults[i].minVal ||
                    testShaderResults[i].maxVal != directImageResults[i].maxVal) {
                    mismatches++;
                    if (mismatches <= 10) { // Print first few mismatches
                         std::cerr << "Verification Mismatch at Index " << i << ":"
                                   << " TestShader={" << testShaderResults[i].minVal << "," << testShaderResults[i].maxVal << "}"
                                   << " DirectImage={" << directImageResults[i].minVal << "," << directImageResults[i].maxVal << "}"
                                   << std::endl;
                    }
                }
            }
            if (mismatches == 0) {
                std::cout << "Success: Test shader output matches direct image readback (" << testShaderResults.size() << " elements verified)." << std::endl;
                success = true;
            } else {
                std::cerr << "Error: Found " << mismatches << " mismatches between test shader output and direct image readback." << std::endl;
            }
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during testMinMaxReadback: " << e.what() << std::endl;
        success = false;
    } catch (...) {
        std::cerr << "An unexpected error occurred during testMinMaxReadback." << std::endl;
        success = false;
    }

    // --- Cleanup Test Resources ---
    if (device) {
        if (testPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, testPipeline, nullptr);
        if (testPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, testPipelineLayout, nullptr);
        if (testSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, testSetLayout, nullptr);
        if (testCS.module != VK_NULL_HANDLE) vkDestroyShaderModule(device, testCS.module, nullptr);
        // destroyBuffer(testOutputBuffer, device); // Destroy the buffer created in this function
    }
     std::cout << "--- Finished MinMax Image Read Test ---" << std::endl;
    return success;
}

int filterUnoccupiedBlocks(char **argv, const char *path)
{
    VK_CHECK(volkInitialize());
    std::string spath = argv[0];
    std::string::size_type pos = spath.find_last_of("/\\");
    if (pos == std::string::npos)
        spath = "";
    else
        spath = spath.substr(0, pos + 1);
    spath += path;

    Volume volume = loadVolume(spath.c_str());

    VkInstance instance = createInstance();
    assert(instance);

    volkLoadInstanceOnly(instance);

    VkDebugReportCallbackEXT debugCallback = registerDebugCallback(instance);

    VkPhysicalDevice physicalDevices[16];
    uint32_t physicalDeviceCount = std::size(physicalDevices);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                        physicalDevices));

    VkPhysicalDevice physicalDevice =
        pickPhysicalDevice(physicalDevices, physicalDeviceCount);
    assert(physicalDevice);

    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0,
                                                  &extensionCount, 0));

    std::vector<VkExtensionProperties> extensions(extensionCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &extensionCount, extensions.data()));

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    assert(props.limits.timestampComputeAndGraphics);

    uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevice);
    assert(familyIndex != VK_QUEUE_FAMILY_IGNORED);

    VkDevice device = createDevice(instance, physicalDevice, familyIndex, false);
    assert(device);

    volkLoadDevice(device);

    vkCmdBeginRendering = vkCmdBeginRenderingKHR;
    vkCmdEndRendering = vkCmdEndRenderingKHR;
    vkCmdPipelineBarrier2 = vkCmdPipelineBarrier2KHR;

    VkQueue queue = nullptr;
    vkGetDeviceQueue(device, familyIndex, 0, &queue);
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    VkCommandPool commandPool = createCommandPool(device, familyIndex);
    // --- End Vulkan Initialization ---

    /**
     * The Min-Max range pass
     */
    // --- Load Shader & Create Pipeline ---

    VkPipelineCache pipelineCache = nullptr;


    // Push constants
    PushConstants pushConstants = {};
    pushConstants.volumeDim    = glm::uvec4(volume.volume_dims, 0); // Pad W with 0
    pushConstants.blockDim     = glm::uvec4(8, 8, 8, 0);      // Pad W with 0
    pushConstants.blockGridDim = glm::uvec4(glm::uvec3((pushConstants.volumeDim + pushConstants.blockDim - 1u) / pushConstants.blockDim), 0);       // Pad W with 0
    pushConstants.isovalue     = 60;

    // --- Prepare Buffers and Images ---
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    Shader minMaxCS{};
    assert(loadShader(minMaxCS, device, argv[0], "spirv/computeMinMax.comp.spv"));

    auto [pipelineLayout, setLayout] = createComputeMinMaxPipelineLayout(device);

    VkPipeline computeMinMaxPipeline = createComputePipeline(device, pipelineCache, minMaxCS, pipelineLayout);
    // --- End Pipeline Creation ---

    // Push constants
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);

    // Staging buffer for input volume upload
    VkDeviceSize volumeBufferSize = volume.volume_data.size();
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, device, memoryProperties, volumeBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stagingBuffer.data, volume.volume_data.data(), volume.volume_data.size());

    // Input volume image
    VkExtent3D volumeExtent = {volume.volume_dims.x, volume.volume_dims.y, volume.volume_dims.z};
    Image volImage = {}; // Assuming Image struct { VkImage image; VkDeviceMemory memory; VkImageView imageView; }
    createImage(volImage, device, memoryProperties, VK_IMAGE_TYPE_3D,
                volumeExtent.width, volumeExtent.height, volumeExtent.depth, 1,
                VK_FORMAT_R8_UINT, // Assuming uint8 input volume data
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT); // Usage for upload and shader read
    uploadVolumeImage(commandBuffer, volImage.image, stagingBuffer, volumeExtent);
    // Input volume image is now in VK_IMAGE_LAYOUT_GENERAL

    // --- Create output image for MinMax results ---
    VkExtent3D gridExtent = {pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z};
    VkDeviceSize minMaxTotalBytes = static_cast<size_t>(gridExtent.width) * gridExtent.height * gridExtent.depth * sizeof(MinMaxResult); // Size based on uint32_t struct
    Image minMaxImage = {}; // Assuming Image struct
    createImage(minMaxImage, device, memoryProperties, VK_IMAGE_TYPE_3D,
                gridExtent.width, gridExtent.height, gridExtent.depth, 1,
                VK_FORMAT_R32G32_UINT, // Format to store uvec2 (min, max as uint32_t)
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT); // Usage for shader write & readback copy

    // Transition MinMax image layout for shader write
    transitionImage(commandBuffer, minMaxImage.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    );

    // --- Dispatch Compute Shader ---
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeMinMaxPipeline);

    // Push descriptors (using updated function)
    pushMinMaxDescriptorSets(commandBuffer, pipelineLayout,
                       volImage.imageView,    // Input volume image view
                       minMaxImage.imageView); // Output min/max image view

    // Dispatch
    vkCmdDispatch(commandBuffer, pushConstants.blockGridDim.x, pushConstants.blockGridDim.y, pushConstants.blockGridDim.z);
    // --- End Dispatch ---

    // Barrier between min-max and occupied block filtering
    VkImageMemoryBarrier2 imageBarrier = {};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imageBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT; // Or VK_ACCESS_2_SHADER_WRITE_BIT
    imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT; // Or VK_ACCESS_2_SHADER_READ_BIT
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL; // No layout change needed
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // Ignore if using same queue
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // Ignore if using same queue
    imageBarrier.image = minMaxImage.image; // Use the correct VkImage handle
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    // End and submit commands
    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
    VK_CHECK(vkDeviceWaitIdle(device)); // Wait for compute to finish
    vkResetCommandPool(device, commandPool, 0);

    //
    // // Test Barrier
    // VkImageMemoryBarrier2 imageTestBarrier = {};
    // imageTestBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    // imageTestBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    // imageTestBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT; // Write completed
    // imageTestBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    // imageTestBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;  // Read upcoming
    // imageTestBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    // imageTestBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;              // No layout change needed
    // imageTestBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // imageTestBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // imageTestBarrier.image = minMaxImage.image; // The image being accessed by both shaders
    // imageTestBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // imageTestBarrier.subresourceRange.levelCount = 1;
    // imageTestBarrier.subresourceRange.layerCount = 1;
    //
    // VkDependencyInfo dependencyTestInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    // dependencyTestInfo.imageMemoryBarrierCount = 1;
    // dependencyTestInfo.pImageMemoryBarriers = &imageTestBarrier;
    // vkCmdPipelineBarrier2(commandBuffer, &dependencyTestInfo);
    //
    // endSingleTimeCommands(device, commandPool, queue, commandBuffer);
    // VK_CHECK(vkDeviceWaitIdle(device)); // Wait for compute to finish
    // vkResetCommandPool(device, commandPool, 0);
    // // End test barrier
    //
    // testMinMaxReadback(
    //     argv,
    //     device,
    //     queue,
    //     commandPool,
    //     memoryProperties,
    //     minMaxImage,
    //     pushConstants
    // );
    Buffer compactedBlockIdBuffer = {};
    Buffer activeBlockCountBuffer = {};
    int blocks = runOccupiedBlockFiltering(
        argv,
        device,
        queue,
        commandPool,
        memoryProperties,
        minMaxImage,
        pushConstants,
        compactedBlockIdBuffer,
        activeBlockCountBuffer
    );
    // --- Readback Results ---
    // Use the new function to read back from the image
    std::vector<MinMaxResult> results = mapMinMaxImage(
        device, memoryProperties, commandPool, queue,
        minMaxImage, gridExtent, minMaxTotalBytes
    );
    // --- End Readback ---


    // --- Cleanup (Simplified - ensure all created resources are destroyed) ---
    VK_CHECK(vkResetCommandPool(device, commandPool, 0)); // Reset pool before destroying
    // vkDestroyCommandPool(device, commandPool, nullptr); // Destroy pool at end

    // vkDestroyPipeline(device, computeMinMaxPipeline, nullptr);
    // vkDestroyShaderModule(device, minMaxCS.module, nullptr);
    // vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    // vkDestroyDescriptorSetLayout(device, setLayout, nullptr);

    destroyImage(minMaxImage, device); // Destroy the new image
    destroyImage(volImage, device);
    destroyBuffer(stagingBuffer, device);
    // destroyBuffer(minMaxBuffer, device); // No longer needed

    // ... rest of cleanup ...
    // vkDestroyDevice(device, nullptr);
    // vkDestroyInstance(instance, nullptr);
    // volkFinalize();
    // --- End Cleanup ---

    return blocks;
}

