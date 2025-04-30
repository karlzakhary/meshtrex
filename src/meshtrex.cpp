#include <fstream>    // For file operations
#include <iostream>

#ifndef __APPLE__
#include <cstdint>
#endif

#include "common.h"
#include "filteringManager.h"
#include "extractionManager.h"
#include "blockFilteringTestUtils.h"
#include "extractionTestUtils.h"

#include "vulkan_context.h"

int main(int argc, char** argv) {
    try {
        std::string volumePath = getFullPath(ROOT_BUILD_PATH, "/raw_volumes/bonsai_256x256x256_uint8.raw");
        float isovalue = 60;
        bool requestMeshShading = false;
#ifndef __APPLE__
        requestMeshShading = true;
#endif
        VulkanContext context(requestMeshShading);
        std::cout << "Vulkan context initialized for filtering." << std::endl;

        Volume volume = loadVolume(volumePath.c_str());
        std::cout << "Volume " << volumePath.c_str() << " is loaded.";
        PushConstants pushConstants = {};
        // ... (Setup pushConstants as before) ...
        pushConstants.volumeDim = glm::uvec4(volume.volume_dims, 1);
        pushConstants.blockDim = glm::uvec4(8, 8, 8, 1);
        pushConstants.blockGridDim = glm::uvec4(
            (volume.volume_dims.x + pushConstants.blockDim.x - 1) / pushConstants.blockDim.x,
            (volume.volume_dims.y + pushConstants.blockDim.y - 1) / pushConstants.blockDim.y,
            (volume.volume_dims.z + pushConstants.blockDim.z - 1) / pushConstants.blockDim.z,
            1);
        pushConstants.isovalue = isovalue;

        std::cout << "Loaded volume dims: ("
                  << pushConstants.volumeDim.x << "x" << pushConstants.volumeDim.y << "x" << pushConstants.volumeDim.z << ")" << std::endl;
        std::cout << "Block grid: " << pushConstants.blockGridDim.x << "x" << pushConstants.blockGridDim.y << "x" << pushConstants.blockGridDim.z << std::endl;

        FilteringOutput filteringResult = filterActiveBlocks(context, volume, pushConstants);

        std::cout << "Filtering complete. Received handles." << std::endl;
        std::cout << "Active blocks: " << filteringResult.activeBlockCount << std::endl;
        ExtractionOutput extractionResultGPU = extractMeshletDescriptors(context, filteringResult, pushConstants);
        CPUExtractionOutput extractionResultCPU = extractMeshletsCPU(context, volume, filteringResult, isovalue);

        // Perform the comparison
        bool match = compareExtractionOutputs(context, extractionResultGPU, extractionResultCPU);

        if (match) {
            std::cout << "Verification Passed!" << std::endl;
        } else {
            std::cout << "Verification Failed!" << std::endl;
        }
        // --- Now use the results in the next stage ---
        // Example: Setting up descriptors for a task/mesh shader
        // VkDescriptorBufferInfo activeBlockCountInfo = { filteringResult.activeBlockCountBuffer.buffer, 0, VK_WHOLE_SIZE };
        // VkDescriptorBufferInfo compactedBlockIdInfo = { filteringResult.compactedBlockIdBuffer.buffer, 0, VK_WHOLE_SIZE };
        // VkDescriptorImageInfo volumeTextureInfo = { VK_NULL_HANDLE, filteringResult.volumeImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; // Assuming layout is transitioned

        // ... bind these descriptors ...
        // ... record task/mesh shader dispatch ...

        // --- EVENTUALLY, when these resources are no longer needed ---
        // (e.g., end of frame, shutdown)
        // The caller MUST clean up the resources contained in filteringResult.
        // It needs access to the VkDevice (and VmaAllocator if used).
        // Get the device handle (e.g., from a global context or stored separately)
        // VkDevice device = get_my_vulkan_device();

        std::cout << "Cleaning up filtering resources..." << std::endl;
        filteringResult.cleanup(context.getDevice());
        // Assuming destroyImage/destroyBuffer take VkDevice:
        // destroyImage(filteringResult.volumeImage, device);
        // destroyImage(filteringResult.minMaxImage, device); // Destroy if created/returned
        // destroyBuffer(filteringResult.compactedBlockIdBuffer, device);
        // destroyBuffer(filteringResult.activeBlockCountBuffer, device);
        // Or, if FilteringOutput has a cleanup method:
        // filteringResult.cleanup(device);


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}