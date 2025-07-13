#include <fstream>    // For file operations
#include <iostream>

#ifndef __APPLE__
#include <cstdint>
#endif

#include "common.h"
#include "minMaxManager.h"
#include "filteringManager.h"
#include "extractionManager.h"
#include "blockFilteringTestUtils.h"
#include "extractionTestUtils.h"
#include <dlfcn.h>
#include "renderdoc_app.h"

#include "vulkan_context.h"

RENDERDOC_API_1_1_2 *rdoc_api = NULL;

std::vector<uint8_t> generateSphereVolume(int width, int height, int depth) {
    std::vector<uint8_t> data(width * height * depth);
    
    float radius = width * 0.4f;  // 40% of volume size
    float centerX = width / 2.0f;
    float centerY = height / 2.0f;
    float centerZ = depth / 2.0f;
    
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float dx = x - centerX;
                float dy = y - centerY;
                float dz = z - centerZ;
                float distance = sqrt(dx*dx + dy*dy + dz*dz) - radius;
                
                // Map distance to 0-255 range
                // 0 = far inside, 128 = at surface, 255 = far outside
                // Add small offset to avoid exact values
                float normalized = (distance / radius) * 127.0f + 128.0f + 0.001f;
                normalized = std::fmax(0.0f, std::fmin(255.0f, normalized));
                
                int index = z * width * height + y * width + x;
                data[index] = static_cast<uint8_t>(normalized);
            }
        }
    }
    
    return data;
}

// Use with isovalue = 128

int main(int argc, char** argv) {
    try {
        std::string volumePath = getFullPath(ROOT_BUILD_PATH, "/raw_volumes/bonsai_256x256x256_uint8.raw");
        float isovalue = 80;
        bool requestMeshShading = false;
#ifndef __APPLE__
        requestMeshShading = true;
#endif
        // For Linux, use dlopen() and dlsym()
        void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (mod) {
            pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
            int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
            assert(ret == 1);
        }
        VulkanContext context(requestMeshShading);
        std::cout << "Vulkan context initialized for filtering." << std::endl;

        Volume volume = loadVolume(volumePath.c_str());
        // Volume volume {glm::vec3(64,64,64), "uint_8", generateSphereVolume(64,64,64)};
        std::cout << "Volume " << volumePath.c_str() << " is loaded.";
        std::cout << "--- Debug: Printing C++ volume data values > 41 for cell (0,0,0) of block (0,0,0) ---" << std::endl;
        // for (int i = 0; i < 1000 ; i++) {
        //     std::cout << "CPU Vol: " << static_cast<unsigned int>(volume.volume_data[i]) << "\n" <<std::endl;
        // }
        std::cout << "--- End Debug: C++ specific volume data print ---" << std::endl;
        // --- End of debug print section --
        PushConstants pushConstants = {};
        pushConstants.volumeDim = glm::uvec4(volume.volume_dims, 1);
        pushConstants.blockDim = glm::uvec4(4, 4, 4, 1);
        pushConstants.blockGridDim = glm::uvec4(
            (volume.volume_dims.x + pushConstants.blockDim.x - 1) / pushConstants.blockDim.x,
            (volume.volume_dims.y + pushConstants.blockDim.y - 1) / pushConstants.blockDim.y,
            (volume.volume_dims.z + pushConstants.blockDim.z - 1) / pushConstants.blockDim.z,
            1);
        pushConstants.isovalue = isovalue;

        std::cout << "Loaded volume dims: ("
                  << pushConstants.volumeDim.x << "x" << pushConstants.volumeDim.y << "x" << pushConstants.volumeDim.z << ")" << std::endl;
        std::cout << "Block grid: " << pushConstants.blockGridDim.x << "x" << pushConstants.blockGridDim.y << "x" << pushConstants.blockGridDim.z << std::endl;

        MinMaxOutput minMaxOutput = computeMinMaxMip(context, volume, pushConstants);
        FilteringOutput filteringResult = filterActiveBlocks(context, minMaxOutput, pushConstants);

        std::cout << "Filtering complete. Received handles." << std::endl;
        std::cout << "Active blocks: (count remains on GPU for GPU-driven pipeline)" << std::endl;
        try {
            ExtractionOutput extractionResultGPU = extractMeshletDescriptors(context, minMaxOutput, filteringResult, pushConstants);
            writeGPUExtractionToOBJ(context, extractionResultGPU, "/home/ge26mot/Projects/meshtrex/build/aikalam.obj");
        } catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }


        // CPUExtractionOutput extractionResultCPU = extractMeshletsCPU(context, volume, filteringResult, isovalue);
        try {
            // CPUExtractionOutput extractionResultCPU = extractMeshletsCPU(context, volume, filteringResult, isovalue);
        } catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }

        // Perform the comparison
        // bool match = compareExtractionOutputs(context, extractionResultGPU, {});

        // if (match) {
        //     std::cout << "Verification Passed!" << std::endl;
        // } else {
        //     std::cout << "Verification Failed!" << std::endl;
        // }
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
        minMaxOutput.cleanup(context.getDevice());
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