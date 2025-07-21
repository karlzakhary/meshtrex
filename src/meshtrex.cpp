#include <fstream>
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
#include "renderingManager.h"
#include "profilingManager.h"

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

int main(int argc, char** argv) {
    try {
        std::string volumePath = getFullPath(ROOT_BUILD_PATH, "/raw_volumes/marmoset_neurons_1024x1024x314_uint8.raw");
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

        Volume volume = loadVolume(volumePath.c_str());
        // Use with isovalue = 128
        // Volume volume {glm::vec3(64,64,64), "uint_8", generateSphereVolume(64,64,64)};
        std::cout << "Volume " << volumePath.c_str() << " is loaded.";
        
        PushConstants pushConstants = {};
        pushConstants.volumeDim = glm::uvec4(volume.volume_dims, 1);
        pushConstants.blockDim = glm::uvec4(3, 3, 3, 1);
        pushConstants.blockGridDim = glm::uvec4(
            (volume.volume_dims.x + pushConstants.blockDim.x - 1) / pushConstants.blockDim.x,
            (volume.volume_dims.y + pushConstants.blockDim.y - 1) / pushConstants.blockDim.y,
            (volume.volume_dims.z + pushConstants.blockDim.z - 1) / pushConstants.blockDim.z,
            1);
        pushConstants.isovalue = isovalue;

        std::cout << "Loaded volume dims: ("
                  << pushConstants.volumeDim.x << "x" << pushConstants.volumeDim.y << "x" << pushConstants.volumeDim.z << ")" << std::endl;
        std::cout << "Block grid: " << pushConstants.blockGridDim.x << "x" << pushConstants.blockGridDim.y << "x" << pushConstants.blockGridDim.z << std::endl;

        // Add profiling option
        bool enableProfiling = true; // You can make this a command line argument
        
        MinMaxOutput minMaxOutput;
        FilteringOutput filteringResult;
        
        if (enableProfiling) {
            std::cout << "\n--- Running with Performance Profiling ---" << std::endl;
            
            try {
                ProfilingManager profiler(context.getDevice(), context.getPhysicalDevice());
                
                // Create command buffer for profiled GPU execution
                VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
                profiler.beginFrame(cmd);
                
                // Run min-max generation with GPU profiling
                minMaxOutput = computeMinMaxMip(context, volume, pushConstants, cmd, &profiler.gpu());
                
                // Run filtering with GPU profiling  
                filteringResult = filterActiveBlocks(context, minMaxOutput, pushConstants, cmd, &profiler.gpu());
                
                // Submit the command buffer and wait for completion
                endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
                
                // Read back the active block count from GPU now that command buffer is submitted
                readActiveBlockCount(context, filteringResult);
                
                // Clean up temporary resources from min-max and filtering
                minMaxOutput.tempResources.cleanup();
                filteringResult.tempResources.cleanup();
                
                // Create new command buffer for extraction
                cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
                
                // Run extraction with GPU profiling
                ExtractionOutput extractionResultGPU = extractMeshletDescriptors(context, minMaxOutput, filteringResult, pushConstants, cmd, &profiler.gpu());
                
                // Submit the extraction command buffer
                endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
                
                // Clean up extraction temporary resources
                extractionResultGPU.tempResources.cleanup();
                minMaxOutput.cleanup(context.getDevice());
                filteringResult.cleanup(context.getDevice());
                                
                profiler.setExtractionStats(
                    0, // Active block count stays on GPU
                    extractionResultGPU.vertexCount,
                    extractionResultGPU.indexCount / 3,
                    extractionResultGPU.meshletCount
                );
                
                // writeGPUExtractionToOBJ(context, extractionResultGPU, "/home/ge26mot/Projects/meshtrex/build/aikalam.obj");
                
                if (extractionResultGPU.meshletCount > 0) {
                    std::cout << "\n--- Starting Renderer ---" << std::endl;
                    RenderingManager renderingManager(context);
                    renderingManager.render(extractionResultGPU);
                } else {
                    std::cout << "\nSkipping rendering as no meshlets were generated." << std::endl;
                }
                
                profiler.endFrame();
                profiler.printSummary();
                profiler.exportCSV("meshtrex_profile.csv");
            } catch (const std::exception& e) {
                std::cerr << "Profiling error: " << e.what() << std::endl;
                // Fall back to non-profiled execution
                if (minMaxOutput.minMaxImage.image == VK_NULL_HANDLE) {
                    minMaxOutput = computeMinMaxMip(context, volume, pushConstants);
                }
                if (filteringResult.activeBlockCount == 0) {
                    filteringResult = filterActiveBlocks(context, minMaxOutput, pushConstants);
                }
            }
            
        } else {
            // Original code without profiling
            minMaxOutput = computeMinMaxMip(context, volume, pushConstants);
            filteringResult = filterActiveBlocks(context, minMaxOutput, pushConstants);
            
            std::cout << "Filtering complete. Active block count remains on GPU." << std::endl;
            
            try {
                ExtractionOutput extractionResultGPU = extractMeshletDescriptors(context, minMaxOutput, filteringResult, pushConstants);
                writeGPUExtractionToOBJ(context, extractionResultGPU, "/home/ge26mot/Projects/meshtrex/build/aikalam.obj");
                if (extractionResultGPU.meshletCount > 0) {
                    std::cout << "\n--- Starting Renderer ---" << std::endl;
                    RenderingManager renderingManager(context);
                    renderingManager.render(extractionResultGPU);
                } else {
                    std::cout << "\nSkipping rendering as no meshlets were generated." << std::endl;
                }
            } catch (std::exception& e) {
                std::cout << e.what() << std::endl;
            }
        }
        
        // Note: filteringResult and minMaxOutput cleanup happens automatically
        // when they go out of scope at the end of main()
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}