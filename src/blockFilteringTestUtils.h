#pragma once

// NEW Function: Reads back the MinMax data from a VkImage via a staging buffer
inline std::vector<MinMaxResult> mapMinMaxImage(VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
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
inline std::vector<MinMaxResult> mapTestResultBuffer(VkDevice device, VkPhysicalDeviceMemoryProperties memoryProperties,
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

    destroyBuffer(readbackBuffer, device);
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
 * @param minMaxImage The VkImage containing min/max data (output from first
 * pass).
 * @param pushData Push constants containing grid dimensions.
 * @return true if the test shader's output matches a direct readback of the
 * image, false otherwise.
 */
inline bool testMinMaxReadback(
    char** argv,
    VulkanContext &context,
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
        createBuffer(testOutputBuffer, context.getDevice(), context.getMemoryProperties(), outputBufferSize,
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
        VK_CHECK(vkCreateDescriptorSetLayout(context.getDevice(), &layoutInfo, nullptr, &testSetLayout));

        VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &testSetLayout;
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(context.getDevice(), &pipelineLayoutCreateInfo, nullptr, &testPipelineLayout));

        // --- Load Test Shader & Create Test Pipeline ---
        // Ensure TestMinMaxRead.comp is compiled to SPIR-V
        assert(loadShader(testCS, context.getDevice(), argv[0], "spirv/testReadingMinMax.comp.spv"));
        testPipeline = createComputePipeline(context.getDevice(), nullptr, testCS, testPipelineLayout);
        assert(testPipeline != VK_NULL_HANDLE);
        std::cout << "Test pipeline created." << std::endl;

        // --- Dispatch Test Shader ---
        VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, testPipeline);

        // Push Descriptors
        VkDescriptorImageInfo minMaxImageInfo = {VK_NULL_HANDLE, minMaxImage.imageView, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo outputBufferInfo = {testOutputBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2] = {};
        writes[0] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            VK_NULL_HANDLE,
            0, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            &minMaxImageInfo,
            nullptr,
            nullptr
        };
        writes[1] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            VK_NULL_HANDLE,
            1, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &outputBufferInfo,
            nullptr
        };
        vkCmdPushDescriptorSetKHR(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            testPipelineLayout,
            0,
            2,
            writes
        );

        // Push Constants (only blockGridDim needed by test shader)
        vkCmdPushConstants(
            cmd,
            testPipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(PushConstants),
            &pushData
        );

        // // Test Barrier
        VkImageMemoryBarrier2 imageTestBarrier = {};
        imageTestBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        imageTestBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        imageTestBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT; // Write completed
        imageTestBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        imageTestBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;  // Read upcoming
        imageTestBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageTestBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;              // No layout change needed
        imageTestBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageTestBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageTestBarrier.image = minMaxImage.image; // The image being accessed by both shaders
        imageTestBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageTestBarrier.subresourceRange.levelCount = 1;
        imageTestBarrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependencyTestInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyTestInfo.imageMemoryBarrierCount = 1;
        dependencyTestInfo.pImageMemoryBarriers = &imageTestBarrier;
        vkCmdPipelineBarrier2(cmd, &dependencyTestInfo);

        // Calculate dispatch size
        uint32_t localSizeX = 128; // Must match test shader's local_size_x
        uint32_t groupCountX = (totalBlocks + localSizeX - 1) / localSizeX;
        vkCmdDispatch(cmd, groupCountX, 1, 1);

        // Barrier: Ensure test shader writes to testOutputBuffer are done before readback
        VkBufferMemoryBarrier2 minMaxComputeTestReadBarrier = bufferBarrier(
            testOutputBuffer.buffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            0, VK_WHOLE_SIZE
        );

        pipelineBarrier(
            cmd,
            {},
            1,
            &minMaxComputeTestReadBarrier,
            0,
            {}
        );
        endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
        VK_CHECK(vkDeviceWaitIdle(context.getDevice())); // Wait for test shader to finish
        std::cout << "Test shader dispatched and finished." << std::endl;
        vkResetCommandPool(context.getDevice(), context.getCommandPool(), 0);
        // 2. Read back results directly from the minMaxImage (using existing function)
        std::vector<MinMaxResult> directImageResults = mapMinMaxImage(
            context.getDevice(), context.getMemoryProperties(), context.getCommandPool(), context.getQueue(),
            minMaxImage, gridExtent, outputBufferSize // outputBufferSize is same as minMaxTotalBytes
        );

        // --- Verification ---
        // 1. Read back results from the test shader's output buffer
        std::vector<MinMaxResult> testShaderResults = mapTestResultBuffer(
            context.getDevice(), context.getMemoryProperties(), context.getCommandPool(), context.getQueue(),
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
        } else {
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
    if (context.getDevice()) {
        if (testPipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.getDevice(), testPipeline, nullptr);
        if (testPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.getDevice(), testPipelineLayout, nullptr);
        if (testSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.getDevice(), testSetLayout, nullptr);
        if (testCS.module != VK_NULL_HANDLE) vkDestroyShaderModule(context.getDevice(), testCS.module, nullptr);
        destroyBuffer(testOutputBuffer, context.getDevice()); // Destroy the buffer created in this function
    }
     std::cout << "--- Finished MinMax Image Read Test ---" << std::endl;
    return success;
}