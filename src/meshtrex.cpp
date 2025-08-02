#include <fstream>
#include <iostream>
#include <cstring>

#ifndef __APPLE__
#include <cstdint>
#endif

#include "common.h"
#include "minMaxManager.h"
#include "filteringManager.h"
#include "extractionManager.h"
#include "transientExtractionManager.h"
#include "blockFilteringTestUtils.h"
#include "extractionTestUtils.h"
#include <dlfcn.h>
#include "renderdoc_app.h"

#include "vulkan_context.h"
#include "renderingManager.h"
#include "profilingManager.h"
#include "densityUtils.h"
#include "common.h"
#include "swapchain.h"
#include "image.h"
#include "buffer.h"
#include "vulkan_utils.h"
#include "resources.h"
#include <GLFW/glfw3.h>

RENDERDOC_API_1_1_2 *rdoc_api = NULL;

void renderTransientExtraction(
    VulkanContext& context,
    MinMaxOutput& minMaxOutput,
    FilteringOutput& filteringResult,
    PushConstants& pushConstants,
    bool pmb
) {
    VkDevice device = context.getDevice();
    
    // Create swapchain and window
    GLFWwindow* window = nullptr;
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(1280, 720, "MeshTrex Transient Renderer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create window");
    }
    
    // Create surface
    VkSurfaceKHR surface = createSurface(context.getInstance(), window);
    
    // Create swapchain
    Swapchain swapchain{};
    VkFormat swapchainFormat = getSwapchainFormat(context.getPhysicalDevice(), surface);
    createSwapchain(swapchain, context.getPhysicalDevice(), device, surface, context.getGraphicsQueueFamilyIndex(), window, swapchainFormat, VK_NULL_HANDLE);
    
    // Create depth buffer
    Image depthImage{};
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    createImage(depthImage, device, context.getMemoryProperties(), VK_IMAGE_TYPE_2D, swapchain.width, swapchain.height, 1, 1, depthFormat, 
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    
    // Create image views
    std::vector<VkImageView> swapchainImageViews(swapchain.imageCount);
    for (uint32_t i = 0; i < swapchain.imageCount; i++) {
        swapchainImageViews[i] = createImageView(device, swapchain.images[i], swapchainFormat, VK_IMAGE_TYPE_2D, 0, 1);
    }
    
    // Camera state
    glm::vec3 cameraPos = glm::vec3(100.f, 100.f, 100.f);
    glm::vec3 cameraTarget = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    double lastMouseX = 0, lastMouseY = 0;
    float lastFrameTime = 0.0f;
    
    // Create synchronization objects
    VkSemaphore acquireSemaphore = createSemaphore(device);
    VkSemaphore releaseSemaphore = createSemaphore(device);
    VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence frameFence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frameFence));
    
    // Allocate command buffer
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = context.getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));
    
    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        glfwPollEvents();
        
        // Simple camera controls
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            glm::vec3 forward = glm::normalize(cameraTarget - cameraPos);
            cameraPos += forward * deltaTime * 50.0f;
            cameraTarget += forward * deltaTime * 50.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            glm::vec3 forward = glm::normalize(cameraTarget - cameraPos);
            cameraPos -= forward * deltaTime * 50.0f;
            cameraTarget -= forward * deltaTime * 50.0f;
        }
        
        // Update camera based on mouse
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            float dx = (float)(mouseX - lastMouseX);
            float dy = (float)(mouseY - lastMouseY);
            
            glm::mat4 rotationY = glm::rotate(glm::mat4(1.0f), -dx * 0.005f, glm::vec3(0,1,0));
            cameraPos = glm::vec3(rotationY * glm::vec4(cameraPos - cameraTarget, 1.0f)) + cameraTarget;
            
            glm::vec3 right = glm::normalize(glm::cross(cameraTarget - cameraPos, glm::vec3(0,1,0)));
            glm::mat4 rotationX = glm::rotate(glm::mat4(1.0f), -dy * 0.005f, right);
            cameraPos = glm::vec3(rotationX * glm::vec4(cameraPos - cameraTarget, 1.0f)) + cameraTarget;
        }
        lastMouseX = mouseX;
        lastMouseY = mouseY;
        
        VK_CHECK(vkWaitForFences(device, 1, &frameFence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &frameFence));
        
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            vkDeviceWaitIdle(device);
            // Recreate swapchain
            for (auto& view : swapchainImageViews) {
                vkDestroyImageView(device, view, nullptr);
            }
            swapchainImageViews.clear();
            destroyImage(depthImage, device);
            destroySwapchain(device, swapchain);
            
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            createSwapchain(swapchain, context.getPhysicalDevice(), device, surface, context.getGraphicsQueueFamilyIndex(), window, swapchainFormat, swapchain.swapchain);
            createImage(depthImage, device, context.getMemoryProperties(), VK_IMAGE_TYPE_2D, swapchain.width, swapchain.height, 1, 1, depthFormat,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            
            swapchainImageViews.resize(swapchain.imageCount);
            for (uint32_t i = 0; i < swapchain.imageCount; i++) {
                swapchainImageViews[i] = createImageView(device, swapchain.images[i], swapchainFormat, VK_IMAGE_TYPE_2D, 0, 1);
            }
            continue;
        }
        assert(result == VK_SUCCESS);
        
        // Build view-projection matrix and frustum planes
        float fov = glm::radians(60.0f);
        float aspect = (float)swapchain.width / (float)swapchain.height;
        float nearPlane = 0.1f;
        float farPlane = 1000.0f;
        
        glm::mat4 proj = glm::mat4(0.0f);
        float tanHalfFov = tan(fov / 2.0f);
        proj[0][0] = 1.0f / (aspect * tanHalfFov);
        proj[1][1] = 1.0f / tanHalfFov;
        proj[2][2] = nearPlane / (farPlane - nearPlane);
        proj[2][3] = -1.0f;
        proj[3][2] = (farPlane * nearPlane) / (farPlane - nearPlane);
        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0, 1, 0));
        glm::mat4 viewProj = proj * view;
        
        // Extract frustum planes
        TransientExtractionPushConstants renderConstants;
        renderConstants.viewProj = viewProj;
        extractFrustumPlanes(viewProj, renderConstants.frustumPlanes);
        
        // Record command buffer
        VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        
        // Prepare all barriers for transient extraction BEFORE beginning rendering
        std::vector<VkBufferMemoryBarrier2> bufferBarriers;
        std::vector<VkImageMemoryBarrier2> imageBarriers;
        
        // Transition images for rendering
        VkImageMemoryBarrier2 colorBarrier = imageBarrier(swapchain.images[imageIndex], 
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        
        VkImageMemoryBarrier2 depthBarrier = imageBarrier(depthImage.image,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        
        imageBarriers.push_back(colorBarrier);
        imageBarriers.push_back(depthBarrier);
        
        // Add barriers for transient extraction inputs
        const VkPipelineStageFlags2 EXTRACTION_SHADER_STAGES = 
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        
        // Ensure filtering outputs are readable
        bufferBarriers.push_back(bufferBarrier(
            filteringResult.activeBlockCountBuffer.buffer,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            0, VK_WHOLE_SIZE
        ));
        
        bufferBarriers.push_back(bufferBarrier(
            filteringResult.compactedBlockIdBuffer.buffer,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            0, VK_WHOLE_SIZE
        ));

        
        // Volume image is already in GENERAL layout from min-max pass, no barrier needed

        // Min-max image barrier
        // imageBarriers.push_back(imageBarrier(
        //     minMaxOutput.minMaxImage.image,
        //     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        //     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        //     VK_IMAGE_LAYOUT_GENERAL,
        //     VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
        //     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        //     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        //     VK_IMAGE_ASPECT_COLOR_BIT
        // ));
        
        // Execute all barriers before beginning rendering
        pipelineBarrier(commandBuffer, 0, bufferBarriers.size(), bufferBarriers.data(), 
                       imageBarriers.size(), imageBarriers.data());
        
        // Begin rendering
        VkClearColorValue clearColor = {0.1f, 0.2f, 0.3f, 1.0f};
        VkClearDepthStencilValue clearDepth = {0.0f, 0};
        
        VkRenderingAttachmentInfo colorAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView = swapchainImageViews[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = clearColor;
        
        VkRenderingAttachmentInfo depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView = depthImage.imageView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = clearDepth;
        
        VkRenderingInfo renderingInfo = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderingInfo.renderArea = {{0, 0}, {swapchain.width, swapchain.height}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;
        
        vkCmdBeginRendering(commandBuffer, &renderingInfo);
        
        VkViewport viewport = {0.0f, (float)swapchain.height, (float)swapchain.width, -(float)swapchain.height, 0.0f, 1.0f};
        VkRect2D scissor = {{0, 0}, {swapchain.width, swapchain.height}};
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        
        // Call transient extraction and render (no barriers needed inside)
        extractAndRenderTransient(context, minMaxOutput, filteringResult, pushConstants, renderConstants, commandBuffer, swapchainFormat, depthFormat, nullptr, pmb);
        
        vkCmdEndRendering(commandBuffer);
        
        // Transition for present
        VkImageMemoryBarrier2 presentBarrier = imageBarrier(swapchain.images[imageIndex],
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &presentBarrier);
        
        VK_CHECK(vkEndCommandBuffer(commandBuffer));
        
        // Submit
        VkPipelineStageFlags submitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &acquireSemaphore;
        submitInfo.pWaitDstStageMask = &submitStageMask;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &releaseSemaphore;
        
        VK_CHECK(vkQueueSubmit(context.getQueue(), 1, &submitInfo, frameFence));
        
        // Present
        VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &releaseSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        
        result = vkQueuePresentKHR(context.getQueue(), &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            // Handle next iteration
        } else {
            assert(result == VK_SUCCESS);
        }
    }
    
    vkDeviceWaitIdle(device);
    
    // Cleanup
    vkFreeCommandBuffers(device, context.getCommandPool(), 1, &commandBuffer);
    vkDestroyFence(device, frameFence, nullptr);
    vkDestroySemaphore(device, releaseSemaphore, nullptr);
    vkDestroySemaphore(device, acquireSemaphore, nullptr);
    
    for (auto& view : swapchainImageViews) {
        vkDestroyImageView(device, view, nullptr);
    }
    destroyImage(depthImage, device);
    destroySwapchain(device, swapchain);
    vkDestroySurfaceKHR(context.getInstance(), surface, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
    
    // Cleanup transient extraction static resources
    cleanupTransientExtractionResources(device);
}

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
        bool pmb = false; // Use regular marching cubes for transient rendering
        // Parse command line arguments
        bool useDensityDispatch = false;
        bool useTransientExtraction = true;
        std::string volumePath = getFullPath(ROOT_BUILD_PATH, "/raw_volumes/aneurism_256x256x256_uint8.raw");
        float isovalue = 80;
        bool requestMeshShading = false;
#ifndef __APPLE__
        requestMeshShading = true;
#endif
        
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--density-dispatch") == 0) {
                useDensityDispatch = false;
                std::cout << "Density-based dispatch enabled" << std::endl;
            } else if (strcmp(argv[i], "--transient") == 0) {
                useTransientExtraction = true;
                std::cout << "Transient extraction enabled (on-the-fly rendering)" << std::endl;
            } else if (strcmp(argv[i], "--volume") == 0 && i + 1 < argc) {
                volumePath = argv[++i];
            } else if (strcmp(argv[i], "--isovalue") == 0 && i + 1 < argc) {
                isovalue = std::stof(argv[++i]);
            }
        }
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
        pushConstants.blockDim = glm::uvec4(1, 1, 1, 1);
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
                
                ExtractionOutput extractionResultGPU;
                
                if (!useTransientExtraction) {
                    // Run extraction with GPU profiling only for persistent extraction
                    if (useDensityDispatch) {
                        // Read volume data for density analysis
                        std::vector<uint8_t> volumeData = DensityUtils::readVolumeData(volumePath);
                        extractionResultGPU = extractMeshletDescriptorsWithDensity(
                            context, minMaxOutput, filteringResult, pushConstants, 
                            volume, true, cmd, &profiler.gpu(), pmb);
                    } else {
                        extractionResultGPU = extractMeshletDescriptors(
                            context, minMaxOutput, filteringResult, pushConstants, 
                            cmd, &profiler.gpu(), pmb);
                    }
                    
                    // Submit the extraction command buffer
                    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
                }
                
                // Debug validation for density-based extraction
                if (!useTransientExtraction && useDensityDispatch) {
                    uint32_t actualVertices = readCounterFromBuffer(context, extractionResultGPU.vertexCountBuffer);
                    uint32_t actualIndices = readCounterFromBuffer(context, extractionResultGPU.indexCountBuffer);
                    uint32_t actualMeshlets = readCounterFromBuffer(context, extractionResultGPU.meshletDescriptorCountBuffer);
                    
                    std::cout << "Density-based extraction results: " << actualVertices << " vertices, " 
                              << actualIndices << " indices (" << actualIndices/3 << " triangles), "
                              << actualMeshlets << " meshlets" << std::endl;
                    
                    // Warning if output seems too low
                    if (actualIndices < 100 && filteringResult.activeBlockCount > 10) {
                        std::cout << "WARNING: Suspiciously low triangle count (" << actualIndices/3 
                                  << " triangles) for " << filteringResult.activeBlockCount 
                                  << " active blocks!" << std::endl;
                    }
                }
                
                // Clean up extraction temporary resources
                if (!useTransientExtraction) {
                    extractionResultGPU.tempResources.cleanup();
                }
                
                if (useTransientExtraction) {
                    // For transient extraction, keep min-max and filtering results for rendering
                    // They will be cleaned up after rendering
                } else {
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                }
                                
                if (!useTransientExtraction) {
                    profiler.setExtractionStats(
                        0, // Active block count stays on GPU
                        extractionResultGPU.vertexCount,
                        extractionResultGPU.indexCount / 3,
                        extractionResultGPU.meshletCount
                    );
                }
                profiler.endFrame();
                profiler.printSummary();
                profiler.exportCSV("meshtrex_profile.csv");
                                
                // writeGPUExtractionToOBJ(context, extractionResultGPU, "/home/ge26mot/Projects/meshtrex/build/aikalam.obj");

                
                if (useTransientExtraction) {
                    // Transient extraction - render on-the-fly without storing geometry
                    std::cout << "\n--- Starting Transient Renderer ---" << std::endl;
                    renderTransientExtraction(context, minMaxOutput, filteringResult, pushConstants, pmb);
                    
                    // Cleanup min-max and filtering results after transient rendering
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                } else if (extractionResultGPU.meshletCount > 0) {
                    std::cout << "\n--- Starting Persistent Renderer ---" << std::endl;
                    RenderingManager renderingManager(context);
                    renderingManager.render(extractionResultGPU);
                } else {
                    std::cout << "\nSkipping rendering as no meshlets were generated." << std::endl;
                }
                
                
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
                ExtractionOutput extractionResultGPU;
                
                if (!useTransientExtraction) {
                    if (useDensityDispatch) {
                        std::vector<uint8_t> volumeData = DensityUtils::readVolumeData(volumePath);
                        extractionResultGPU = extractMeshletDescriptorsWithDensity(
                            context, minMaxOutput, filteringResult, pushConstants, 
                            volume, useDensityDispatch, nullptr, nullptr, pmb);
                    } else {
                        extractionResultGPU = extractMeshletDescriptors(
                            context, minMaxOutput, filteringResult, pushConstants, nullptr, nullptr, pmb);
                    }
                }
                
                // Debug validation for density-based extraction
                if (!useTransientExtraction && useDensityDispatch) {
                    uint32_t actualVertices = readCounterFromBuffer(context, extractionResultGPU.vertexCountBuffer);
                    uint32_t actualIndices = readCounterFromBuffer(context, extractionResultGPU.indexCountBuffer);
                    uint32_t actualMeshlets = readCounterFromBuffer(context, extractionResultGPU.meshletDescriptorCountBuffer);
                    
                    std::cout << "Density-based extraction results: " << actualVertices << " vertices, " 
                              << actualIndices << " indices (" << actualIndices/3 << " triangles), "
                              << actualMeshlets << " meshlets" << std::endl;
                    
                    // Warning if output seems too low
                    if (actualIndices < 100 && filteringResult.activeBlockCount > 10) {
                        std::cout << "WARNING: Suspiciously low triangle count (" << actualIndices/3 
                                  << " triangles) for " << filteringResult.activeBlockCount 
                                  << " active blocks!" << std::endl;
                    }
                }
                
                if (!useTransientExtraction) {
                    writeGPUExtractionToOBJ(context, extractionResultGPU, "/home/ge26mot/Projects/meshtrex/build/aikalam.obj");
                }
                
                if (useTransientExtraction) {
                    // Transient extraction - render on-the-fly without storing geometry
                    std::cout << "\n--- Starting Transient Renderer ---" << std::endl;
                    renderTransientExtraction(context, minMaxOutput, filteringResult, pushConstants, pmb);
                    
                    // Cleanup min-max and filtering results after transient rendering
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                } else if (extractionResultGPU.meshletCount > 0) {
                    std::cout << "\n--- Starting Persistent Renderer ---" << std::endl;
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