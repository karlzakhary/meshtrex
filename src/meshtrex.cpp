#include <fstream>    // For file operations
#include <iostream>
#include <chrono>

#ifndef __APPLE__
#include <cstdint>
#endif

#include "common.h"
#include "minMaxManager.h"
#include "filteringManager.h"
#include "extractionManager.h"
#include "renderingFramework.h"
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

int main(int argc, char** argv) {
    try {
        std::string volumePath = getFullPath(ROOT_BUILD_PATH, "/raw_volumes/bonsai_256x256x256_uint8.raw");
        float isovalue = 80;
        bool requestMeshShading = false;
#ifndef __APPLE__
        requestMeshShading = true;
#endif
        
        // Check for command line arguments
        if (argc > 1) {
            volumePath = argv[1];
        }
        if (argc > 2) {
            isovalue = std::stof(argv[2]);
        }
        
        // RenderDoc integration
        void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (mod) {
            pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
            int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
            assert(ret == 1);
        }
        
        // Initialize Vulkan context
        VulkanContext context(requestMeshShading);
        std::cout << "Vulkan context initialized." << std::endl;

        // Load volume data
        Volume volume = loadVolume(volumePath.c_str());
        // Alternative: Generate procedural sphere
        // Volume volume {glm::vec3(64,64,64), "uint_8", generateSphereVolume(64,64,64)};
        
        std::cout << "Volume loaded: " << volumePath << std::endl;
        std::cout << "Volume dimensions: " << volume.volume_dims.x << "x" 
                  << volume.volume_dims.y << "x" << volume.volume_dims.z << std::endl;
        std::cout << "Isovalue: " << isovalue << std::endl;
        
        // Setup push constants
        PushConstants pushConstants = {};
        pushConstants.volumeDim = glm::uvec4(volume.volume_dims, 1);
        pushConstants.blockDim = glm::uvec4(4, 4, 4, 1);
        pushConstants.blockGridDim = glm::uvec4(
            (volume.volume_dims.x + pushConstants.blockDim.x - 1) / pushConstants.blockDim.x,
            (volume.volume_dims.y + pushConstants.blockDim.y - 1) / pushConstants.blockDim.y,
            (volume.volume_dims.z + pushConstants.blockDim.z - 1) / pushConstants.blockDim.z,
            1);
        pushConstants.isovalue = isovalue;

        std::cout << "Block grid: " << pushConstants.blockGridDim.x << "x" 
                  << pushConstants.blockGridDim.y << "x" << pushConstants.blockGridDim.z << std::endl;

        // --- Marching Cubes Pipeline ---
        std::cout << "\n--- Starting Marching Cubes Pipeline ---" << std::endl;
        
        // 1. Min-Max Octree Construction
        auto startTime = std::chrono::high_resolution_clock::now();
        MinMaxOutput minMaxOutput = computeMinMaxMip(context, volume, pushConstants);
        auto endTime = std::chrono::high_resolution_clock::now();
        std::cout << "Min-Max octree computed in " 
                  << std::chrono::duration<float, std::milli>(endTime - startTime).count() 
                  << " ms" << std::endl;
        
        // 2. Active Block Filtering
        startTime = std::chrono::high_resolution_clock::now();
        FilteringOutput filteringResult = filterActiveBlocks(context, minMaxOutput, pushConstants);
        endTime = std::chrono::high_resolution_clock::now();
        std::cout << "Active blocks filtered in " 
                  << std::chrono::duration<float, std::milli>(endTime - startTime).count() 
                  << " ms" << std::endl;
        
        // 3. Mesh Extraction
        startTime = std::chrono::high_resolution_clock::now();
        ExtractionOutput extractionResult = extractMeshletDescriptors(context, minMaxOutput, filteringResult, pushConstants);
        endTime = std::chrono::high_resolution_clock::now();
        std::cout << "Mesh extracted in " 
                  << std::chrono::duration<float, std::milli>(endTime - startTime).count() 
                  << " ms" << std::endl;
        
        // Optional: Write to OBJ file for debugging
        // writeGPUExtractionToOBJ(context, extractionResult, "/tmp/meshtrex_output.obj");
        
        // --- Create Rendering Framework ---
        std::cout << "\n--- Starting Rendering ---" << std::endl;
        RenderingFramework renderer(context, 1280, 720);
        
        // Main rendering loop
        double lastTime = glfwGetTime();
        int frameCount = 0;
        
        while (!renderer.shouldClose()) {
            // Calculate FPS
            double currentTime = glfwGetTime();
            frameCount++;
            if (currentTime - lastTime >= 1.0) {
                std::cout << "FPS: " << frameCount << std::endl;
                frameCount = 0;
                lastTime = currentTime;
            }
            
            // Handle input
            glfwPollEvents();
            renderer.processInput();
            
            // Render frame
            renderer.renderFrame(extractionResult);
        }
        
        // Wait for GPU to finish before cleanup
        renderer.waitIdle();
        
        // Cleanup resources
        std::cout << "\nCleaning up resources..." << std::endl;
        filteringResult.cleanup(context.getDevice());
        minMaxOutput.cleanup(context.getDevice());
        // extractionResult cleanup is handled by its destructor

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "MeshTrex completed successfully!" << std::endl;
    return 0;
}