#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#ifndef __APPLE__
#include <cstdint>
#endif

#include "common.h"
#include "minMaxManager.h"
#include "filteringManager.h"
#include "extractionManager.h"
#include "transientExtractionManager.h"
#include "extractionTestUtils.h"
#include <dlfcn.h>
#include "renderdoc_app.h"
#include "rasterOcclusionPass.h"
#include "computeOcclusionPass.h"
#include "transientExtractionPass.h"
#include "generateMultiSphereVolume.h"

#include "vulkan_context.h"
#include "renderingManager.h"
#include "profilingManager.h"
#include "densityUtils.h"
#include "common.h"
#include "config.h"
#include "swapchain.h"
#include "image.h"
#include "buffer.h"
#include "vulkan_utils.h"
#include "resources.h"
#include <GLFW/glfw3.h>

RENDERDOC_API_1_1_2 *rdoc_api = NULL;

void renderTemporalCoherence(
    VulkanContext& context,
    MinMaxOutput& minMaxOutput,
    PushConstants& pushConstants,
    bool pmb,
    bool disableCoherenceOptimization = false
) {
    VkDevice device = context.getDevice();
    
    // Initialize all resources to null for proper cleanup
    GLFWwindow* window = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    Swapchain swapchain{};
    Image depthImage{};
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkSemaphore> acquireSemaphores;
    std::vector<VkSemaphore> releaseSemaphores;
    std::vector<VkFence> frameFences;
    std::vector<VkCommandBuffer> commandBuffers;
    RasterOcclusionPass::Output occlusionOutput;
    
    try {
        // Create swapchain and window
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(1280, 720, "MeshTrex Temporal Coherence Renderer", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("Failed to create window");
        }
        
        // Create surface
        surface = createSurface(context.getInstance(), window);
    
        // Create swapchain
        VkFormat swapchainFormat = getSwapchainFormat(context.getPhysicalDevice(), surface);
        createSwapchain(swapchain, context.getPhysicalDevice(), device, surface, context.getGraphicsQueueFamilyIndex(), window, swapchainFormat, VK_NULL_HANDLE);
        
        // Create depth buffer (with sampled bit for Hi-Z generation)
        VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
        createImage(depthImage, device, context.getMemoryProperties(), VK_IMAGE_TYPE_2D, swapchain.width, swapchain.height, 1, 1, depthFormat, 
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        
        // Create image views
        swapchainImageViews.resize(swapchain.imageCount);
        for (uint32_t i = 0; i < swapchain.imageCount; i++) {
            swapchainImageViews[i] = createImageView(device, swapchain.images[i], swapchainFormat, VK_IMAGE_TYPE_2D, 0, 1);
        }
    
    // Camera state - will be properly initialized based on actual volume dimensions
    // For now, assume typical volume centered around 128,128,128 (common for 256^3 volumes)
    glm::vec3 volumeCenter = glm::vec3(128.0f, 128.0f, 128.0f);
    float cameraDistance = 400.0f;  // Distance from center
    float cameraYaw = -45.0f * 3.14159f / 180.0f;    // Horizontal angle in radians
    float cameraPitch = 30.0f * 3.14159f / 180.0f;   // Vertical angle in radians (looking down from above)
    
    // Calculate initial camera position from spherical coordinates
    glm::vec3 cameraPos;
    cameraPos.x = volumeCenter.x + cameraDistance * cos(cameraPitch) * cos(cameraYaw);
    cameraPos.y = volumeCenter.y + cameraDistance * sin(cameraPitch);
    cameraPos.z = volumeCenter.z + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
    
    glm::vec3 cameraTarget = volumeCenter;
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    double lastMouseX = 0, lastMouseY = 0;
    float lastFrameTime = 0.0f;
    
    // Create synchronization objects
    // We need more acquire semaphores than swapchain images to avoid reuse conflicts
    acquireSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    // Release semaphores are per swapchain image
    releaseSemaphores.resize(swapchain.imageCount);
    
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        acquireSemaphores[i] = createSemaphore(device);
    }
    for (uint32_t i = 0; i < swapchain.imageCount; i++) {
        releaseSemaphores[i] = createSemaphore(device);
    }
    
    // Fences are per swapchain image
    frameFences.resize(swapchain.imageCount);
    for (uint32_t i = 0; i < swapchain.imageCount; i++) {
        VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frameFences[i]));
    }
    
    uint32_t currentFrame = 0;  // Track which set of sync objects to use
    
    // Allocate command buffers - one per frame to avoid conflicts
    commandBuffers.resize(swapchain.imageCount);
    VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = context.getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = swapchain.imageCount;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));
    
    // Track per-frame temporary resources
    struct FrameResources {
        bool hasResources = false;
        uint32_t frameNumber = 0;
        uint32_t createdAtFrame = 0;  // Track when resources were created
        // Store the actual temp resources to destroy later
        RasterOcclusionPass::Output::TempResources occlusionTempResources;
        RasterOcclusionPass::IndirectTempResources indirectTempResources;
        TransientExtractionPass::TempResources transientTempResources;
    };
    std::vector<FrameResources> frameResources(swapchain.imageCount);
    
    RasterOcclusionPass rasterOcclusionPass(context);
    ComputeOcclusionPass computeOcclusionPass(context);
    TransientExtractionPass transientPass(context, swapchainFormat);
    occlusionOutput.isFirstFrame = true;
    
    // Main render loop
    bool enableDebugColors = false;
    bool disableTemporalCoherence = disableCoherenceOptimization;  // Flag to force occlusion updates every frame
    bool freezePVS = false;  // Flag to freeze PVS updates for debugging
    bool useComputeOcclusion = true;  // Toggle between raster and compute occlusion
    int framesProcessed = 0;  // Track frames processed for first-frame handling
    int occlusionUpdateCount = 0;  // Track occlusion updates for statistics
    int totalFrames = 0;  // Total frames rendered
    bool occlusionUpdated = true;
    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        glfwPollEvents();
        
        // Handle input
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        
        // Camera controls - improved for better exploration
        const float zoomSpeed = 300.0f * deltaTime;  // Increased from 100
        const float panSpeed = 150.0f * deltaTime;   // Increased from 50
        const float mouseSensitivity = 1.0f;         // Increased from 0.5
        
        // W/S: Zoom in/out (move camera closer/farther from target)
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraDistance = glm::max(10.0f, cameraDistance - zoomSpeed);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraDistance = glm::min(1000.0f, cameraDistance + zoomSpeed);
        }
        
        // A/D: Pan camera target left/right
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            glm::vec3 forward = glm::normalize(cameraPos - cameraTarget);
            glm::vec3 right = glm::normalize(glm::cross(cameraUp, forward));
            cameraTarget -= right * panSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            glm::vec3 forward = glm::normalize(cameraPos - cameraTarget);
            glm::vec3 right = glm::normalize(glm::cross(cameraUp, forward));
            cameraTarget += right * panSpeed;
        }
        
        // Q/E: Pan camera target up/down
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            cameraTarget.y -= panSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            cameraTarget.y += panSpeed;
        }
        
        // Update camera position based on spherical coordinates
        cameraPos.x = cameraTarget.x + cameraDistance * cos(cameraPitch) * cos(cameraYaw);
        cameraPos.y = cameraTarget.y + cameraDistance * sin(cameraPitch);
        cameraPos.z = cameraTarget.z + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
        
        // Toggle debug colors with 'C' key
        static bool cKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !cKeyPressed) {
            enableDebugColors = !enableDebugColors;
            std::cout << "Debug colors " << (enableDebugColors ? "enabled" : "disabled") << std::endl;
            cKeyPressed = true;
        } else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE) {
            cKeyPressed = false;
        }
        
        // Toggle temporal coherence with 'T' key
        static bool tKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && !tKeyPressed) {
            disableTemporalCoherence = !disableTemporalCoherence;
            std::cout << "\n=== Temporal coherence " << (disableTemporalCoherence ? "DISABLED" : "ENABLED") << " ==="<< std::endl;
            if (disableTemporalCoherence) {
                std::cout << "Mode: Update occlusion EVERY frame (baseline)" << std::endl;
            } else {
                std::cout << "Mode: Smart updates only when camera moves" << std::endl;
                std::cout << "Current stats: " << occlusionUpdateCount << " updates in " << totalFrames 
                         << " frames (" << std::fixed << std::setprecision(1) 
                         << (100.0f * float(occlusionUpdateCount) / float(std::max(1, totalFrames))) 
                         << "% update rate)" << std::endl;
            }
            // Reset counters when switching modes
            occlusionUpdateCount = 0;
            totalFrames = 0;
            tKeyPressed = true;
        } else if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE) {
            tKeyPressed = false;
        }
        
        // Freeze/unfreeze PVS with 'F' key
        static bool fKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS && !fKeyPressed) {
            freezePVS = !freezePVS;
            std::cout << "\n=== PVS " << (freezePVS ? "FROZEN" : "UNFROZEN") << " ===" << std::endl;
            if (freezePVS) {
                std::cout << "PVS updates disabled - continuously rendering same PVS" << std::endl;
                std::cout << "Current PVS blocks: " << occlusionOutput.pvsPreviousCount << std::endl;
            } else {
                std::cout << "PVS updates resumed" << std::endl;
            }
            fKeyPressed = true;
        } else if (glfwGetKey(window, GLFW_KEY_F) == GLFW_RELEASE) {
            fKeyPressed = false;
        }
        
        // Toggle occlusion method with 'O' key
        static bool oKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS && !oKeyPressed) {
            useComputeOcclusion = !useComputeOcclusion;
            std::cout << "\n=== Occlusion method: " << (useComputeOcclusion ? "COMPUTE (Hi-Z)" : "RASTER") << " ===" << std::endl;
            if (useComputeOcclusion) {
                std::cout << "Using compute-based Hi-Z occlusion culling" << std::endl;
            } else {
                std::cout << "Using raster-based proxy quad occlusion culling" << std::endl;
            }
            // Reset occlusion state when switching methods
            occlusionOutput.isFirstFrame = true;
            framesProcessed = 0;
            oKeyPressed = true;
        } else if (glfwGetKey(window, GLFW_KEY_O) == GLFW_RELEASE) {
            oKeyPressed = false;
        }
        
        // Reset temporal state with 'R' key
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            occlusionOutput.isFirstFrame = true;
            framesProcessed = 0;  // Reset frame counter for first-frame handling
            std::cout << "Temporal state reset" << std::endl;
        }
        
        // Mouse controls for camera rotation
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            float deltaX = (float)(mouseX - lastMouseX) * mouseSensitivity;
            float deltaY = (float)(mouseY - lastMouseY) * mouseSensitivity;
            
            // Update yaw and pitch based on mouse movement (increased sensitivity)
            cameraYaw -= deltaX * 0.02f;   // Doubled from 0.01
            cameraPitch -= deltaY * 0.02f; // Doubled from 0.01
            
            // Clamp pitch to avoid flipping
            cameraPitch = glm::clamp(cameraPitch, -89.0f * 3.14159f / 180.0f, 89.0f * 3.14159f / 180.0f);
            
            // Update camera position
            cameraPos.x = cameraTarget.x + cameraDistance * cos(cameraPitch) * cos(cameraYaw);
            cameraPos.y = cameraTarget.y + cameraDistance * sin(cameraPitch);
            cameraPos.z = cameraTarget.z + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
        }
        
        // Mouse scroll for zoom
        static double lastScrollY = 0.0;
        double scrollX, scrollY;
        scrollY = 0; // This would need to be handled via callback
        if (scrollY != lastScrollY) {
            cameraDistance = glm::clamp(cameraDistance - (float)(scrollY - lastScrollY) * 20.0f, 10.0f, 1000.0f);
            lastScrollY = scrollY;
            
            // Update camera position with new distance
            cameraPos.x = cameraTarget.x + cameraDistance * cos(cameraPitch) * cos(cameraYaw);
            cameraPos.y = cameraTarget.y + cameraDistance * sin(cameraPitch);
            cameraPos.z = cameraTarget.z + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
        }
        
        lastMouseX = mouseX;
        lastMouseY = mouseY;
        
        // Compute matrices
        glm::mat4 viewMatrix = glm::lookAt(cameraPos, cameraTarget, cameraUp);
        // Standard projection first
        glm::mat4 projMatrix = glm::perspective(glm::radians(45.0f), (float)swapchain.width / (float)swapchain.height, 0.1f, 1000.0f);
        projMatrix[1][1] *= -1; // Flip Y for Vulkan
        
        // Convert to reversed-Z by modifying the projection matrix
        // Reversed-Z: z' = -z_near / (z_far - z)
        projMatrix[2][2] = 0.0f;  // Was: -(far+near)/(far-near) for standard
        projMatrix[2][3] = -1.0f;
        projMatrix[3][2] = 0.1f;  // near plane value
        
        glm::mat4 viewProjMatrix = projMatrix * viewMatrix;
        
        // DEBUG: Print frame info
        static int frameCount = 0;
        frameCount++;
        
        // Acquire swapchain image
        uint32_t imageIndex;
        // Use a simple round-robin for acquire semaphores since we just need one that's free
        VK_CHECK(vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, acquireSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex));
        
        // Wait for this image's fence (from its previous use)
        VK_CHECK(vkWaitForFences(device, 1, &frameFences[imageIndex], VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &frameFences[imageIndex]));
        
        // Read back and destroy resources associated with this image from its previous use
        // Only destroy if resources are old enough (at least MAX_FRAMES_IN_FLIGHT frames old)
        if (frameResources[imageIndex].hasResources) {
            uint32_t framesSinceCreation = frameCount - frameResources[imageIndex].createdAtFrame;
            
            // Only destroy if old enough (at least 3 frames old to ensure GPU is done)
            if (framesSinceCreation >= 3) {
                // Only read back if we have a readback buffer
                if (frameResources[imageIndex].occlusionTempResources.readbackBuffer.buffer) {
                    // Temporarily restore the readback buffer to occlusionOutput for reading
                    occlusionOutput.tempResources.readbackBuffer = frameResources[imageIndex].occlusionTempResources.readbackBuffer;
                    occlusionOutput.readbackPVSCounts(device);
                    occlusionOutput.tempResources.readbackBuffer = {};
                }
                
                // Now destroy the saved temp resources
                frameResources[imageIndex].occlusionTempResources.destroy(device);
                frameResources[imageIndex].indirectTempResources.destroy(device);
                frameResources[imageIndex].transientTempResources.destroy(device);
                frameResources[imageIndex].hasResources = false;
            }
        }
        
        // Use the command buffer associated with this swapchain image
        VkCommandBuffer commandBuffer = commandBuffers[imageIndex];
        
        // Reset descriptor pool for this frame
        // Begin frame for the active occlusion pass
        if (useComputeOcclusion) {
            computeOcclusionPass.beginFrame();
        } else {
            rasterOcclusionPass.beginFrame();
        }
        
        // Begin command buffer
        VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        
        // Transition swapchain image to color attachment
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
        
        VkImageMemoryBarrier2 barriers[] = {colorBarrier, depthBarrier};
        pipelineBarrier(commandBuffer, 0, 0, nullptr, 2, barriers);
        
        // Clear color buffer always, but only clear depth on first frame
        VkClearColorValue clearColor = {0.1f, 0.2f, 0.3f, 1.0f};
        VkClearDepthStencilValue clearDepth = {0.0f, 0};  // Reversed-Z: clear to far plane
        
        VkRenderingAttachmentInfo colorAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView = swapchainImageViews[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = clearColor;
        
        VkRenderingAttachmentInfo depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView = depthImage.imageView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        // Depth clear logic depends on occlusion method:
        // - Raster: Always clear (hardware depth test needs clean slate)
        // - Compute: Preserve previous frame's depth for Hi-Z (except first frame)
        bool shouldClearDepth = false;
        if (useComputeOcclusion) {
            // Compute path: only clear on very first frame, preserve depth for Hi-Z generation
            shouldClearDepth = occlusionOutput.isFirstFrame;
        } else {
            // Raster path: always clear depth for proper hardware occlusion testing
            shouldClearDepth = true;
        }
        depthAttachment.loadOp = shouldClearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = clearDepth;
        
        VkRenderingInfo renderingInfo = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderingInfo.renderArea = {{0, 0}, {swapchain.width, swapchain.height}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;
        
        vkCmdBeginRendering(commandBuffer, &renderingInfo);
        vkCmdEndRendering(commandBuffer);
        
        // Kreskowski's correct rendering order: Pass1 -> Occlusion Culling -> Pass2
        // This ensures occlusion culling tests against Pass1's depth buffer
        
        // Step 1: Render Pass 1 - Previous frame's visible blocks (establishes depth)
        TransientExtractionPass::ShadingParameters shadingParams = TransientExtractionPass::getDefaultShadingParams();
        shadingParams.viewPos = cameraPos;
        shadingParams.enableDebugColors = enableDebugColors ? 1 : 0;
        
        // Step 2: Prepare for occlusion culling (check if update needed, initialize buffers)
        totalFrames++;
        bool occlusionShouldUpdate = false;
        
        // Skip occlusion updates if PVS is frozen
        if (freezePVS) {
            occlusionUpdated = false;
        } else {
            // Ensure output buffers are initialized (needed for both methods)
            if (occlusionOutput.pvsCurrentBuffer.buffer == VK_NULL_HANDLE) {
                uint32_t blocksX = (pushConstants.volumeDim.x + pushConstants.blockDim.x - 1) / pushConstants.blockDim.x;
                uint32_t blocksY = (pushConstants.volumeDim.y + pushConstants.blockDim.y - 1) / pushConstants.blockDim.y;
                uint32_t blocksZ = (pushConstants.volumeDim.z + pushConstants.blockDim.z - 1) / pushConstants.blockDim.z;
                uint32_t totalBlocks = blocksX * blocksY * blocksZ;
                
                if (useComputeOcclusion) {
                    computeOcclusionPass.initializeOutput(occlusionOutput, totalBlocks);
                } else {
                    rasterOcclusionPass.initializeOutput(occlusionOutput, totalBlocks);
                }
            }
            
            // Choose occlusion method based on toggle
            if (useComputeOcclusion) {
                // For compute occlusion, only prepare buffers at this stage
                // IMPORTANT: Always force update on first frame to populate initial PVS
                bool forceUpdate = (disableTemporalCoherence && !freezePVS) || occlusionOutput.isFirstFrame;
               
                occlusionShouldUpdate = computeOcclusionPass.performComputeOcclusionCulling(
                    commandBuffer,
                    occlusionOutput,
                    minMaxOutput,
                    pushConstants,
                    viewProjMatrix,
                    depthImage.image,
                    depthImage.imageView,
                    {swapchain.width, swapchain.height},
                    forceUpdate
                );
            }
            // Note: Raster occlusion will be performed AFTER Pass 1 rendering
        }
        
        // Step 3: Different flow for raster vs compute occlusion
        if (useComputeOcclusion && !freezePVS) {
            // COMPUTE PATH: Generate Hi-Z from previous frame's complete depth, then cull
            if (occlusionShouldUpdate) {
                // Generate Hi-Z pyramid from the previous frame's COMPLETE depth buffer
                // The depth buffer here contains the full rendered scene from the last frame
                computeOcclusionPass.generateHiZPyramid(
                    commandBuffer,
                    depthImage.image,
                    depthImage.imageView,
                    {swapchain.width, swapchain.height}
                );
                
                // Run occlusion culling - test current frame's blocks against previous frame's complete depth
                computeOcclusionPass.runOcclusionCulling(
                    commandBuffer,
                    occlusionOutput,
                    minMaxOutput,
                    pushConstants,
                    viewProjMatrix,
                    {swapchain.width, swapchain.height}
                );
                occlusionUpdated = true;
            } else {
                // Temporal coherence says no update needed - keep using existing PVS
                occlusionUpdated = false;
            }
        }
        
        // Step 4: Render Pass 1 - Previous frame's visible blocks
        // For raster: this establishes depth for occlusion testing
        // For compute: this is just normal rendering after occlusion was already done
        if (occlusionOutput.pvsPreviousCount > 0 || true) {
            transientPass.renderPass1_PreviousVisible(
                commandBuffer,
                occlusionOutput,
                minMaxOutput,
                pushConstants,
                viewProjMatrix,
                cameraPos,
                swapchainImageViews[imageIndex],
                depthImage.imageView,
                {swapchain.width, swapchain.height},
                shadingParams
            );
        }
        
        // Step 5: For raster occlusion, perform occlusion culling AFTER Pass 1
        // Pass 1 has established depth, now test new blocks against it
        if (!useComputeOcclusion && !freezePVS) {
            occlusionUpdated = rasterOcclusionPass.performTemporalOcclusionCulling(
                commandBuffer,
                occlusionOutput,
                minMaxOutput,
                pushConstants,
                viewProjMatrix,
                depthImage.imageView,
                {swapchain.width, swapchain.height},
                disableTemporalCoherence && !freezePVS
            );
        }
        
        if (occlusionUpdated) {
            occlusionUpdateCount++;
            
            // Always show stats, but with different formatting based on mode
            if (disableTemporalCoherence) {
                // When forcing updates every frame, show periodic stats
                if (totalFrames % 60 == 0) {  // Report every second at 60fps
                    printf("[Frame %d] Temporal coherence DISABLED - updating every frame (%d/%d, 0.0%% skip rate)\n", 
                           totalFrames, occlusionUpdateCount, totalFrames);
                }
            } else {
                // Smart mode - show why we updated
                const char* reason = occlusionOutput.pvsChanged ? "PVS changed" : "Camera moved";
                // printf("[Frame %d] Occlusion culling performed (update %d/%d, %.1f%% skip rate) - %s\n", 
                //        totalFrames, occlusionUpdateCount, totalFrames,
                //        100.0f * (1.0f - float(occlusionUpdateCount) / float(totalFrames)),
                //        reason);
                
                // Report PVS stability if it's been stable for a while
                if (occlusionOutput.framesWithStablePVS > 5) {
                    printf("  -> PVS has been stable for %d frames (count=%d)\n", 
                           occlusionOutput.framesWithStablePVS, occlusionOutput.pvsCurrentCount);
                }
            }
        }
        
        // Step 5: Render Pass 2 - Newly visible blocks (PVS difference)
        // Simple condition: render if we have new blocks or during bootstrap
        if (!freezePVS && (occlusionUpdated || occlusionOutput.pvsDifferenceCount > 0)) {
            transientPass.renderPass2_NewlyVisible(
                commandBuffer,
                occlusionOutput,
                minMaxOutput,
                pushConstants,
                viewProjMatrix,
                cameraPos,
                swapchainImageViews[imageIndex],
                depthImage.imageView,
                {swapchain.width, swapchain.height},
                shadingParams
            );
        }
        
        // Keep first frame mode active for all frames in flight during startup
        if (occlusionOutput.isFirstFrame) {
            framesProcessed++;
            // Only clear first frame flag after all frames in flight have been processed
            if (framesProcessed >= swapchain.imageCount) {
                occlusionOutput.isFirstFrame = false;
            }
        }
        
        // Buffer management for both raster and compute occlusion
        if (!freezePVS) {
            // Both paths now use buffer swapping for consistency
            // After an update, swap buffers so current becomes previous for next frame
            if (occlusionUpdated || occlusionOutput.frameIndex < 3) {
                occlusionOutput.swapTemporalBuffers();
            }
        }
        
        // Transition for present
        VkImageMemoryBarrier2 presentBarrier = imageBarrier(swapchain.images[imageIndex],
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &presentBarrier);
        
        VK_CHECK(vkEndCommandBuffer(commandBuffer));
        
        // Submit using the current frame's semaphores and fence
        VkPipelineStageFlags submitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &acquireSemaphores[currentFrame];
        submitInfo.pWaitDstStageMask = &submitStageMask;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &releaseSemaphores[imageIndex];
        VK_CHECK(vkQueueSubmit(context.getQueue(), 1, &submitInfo, frameFences[imageIndex]));
        
        // Present using the release semaphore for this image
        VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &releaseSemaphores[imageIndex];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        VK_CHECK(vkQueuePresentKHR(context.getQueue(), &presentInfo));
        
        frameResources[imageIndex].occlusionTempResources = occlusionOutput.tempResources;
        occlusionOutput.tempResources = {}; // Clear the original so it doesn't get destroyed prematurely
        frameResources[imageIndex].indirectTempResources = rasterOcclusionPass.getIndirectTempResources();
        frameResources[imageIndex].transientTempResources = transientPass.getTempResources();
        frameResources[imageIndex].hasResources = true;
        frameResources[imageIndex].createdAtFrame = frameCount - 1;  // Record when created
        
        // Advance to next frame
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    
        // Wait for device idle
        vkDeviceWaitIdle(device);
        
        // Cleanup
        // Clean up any remaining frame resources
        for (auto& frame : frameResources) {
            if (frame.hasResources) {
                frame.occlusionTempResources.destroy(device);
                frame.indirectTempResources.destroy(device);
                frame.transientTempResources.destroy(device);
            }
        }
        occlusionOutput.destroy(device);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in renderTemporalCoherence: " << e.what() << std::endl;
    }
    
    // Cleanup all resources - safe to call even if not created
    if (!commandBuffers.empty()) {
        vkFreeCommandBuffers(device, context.getCommandPool(), commandBuffers.size(), commandBuffers.data());
    }
    
    for (size_t i = 0; i < frameFences.size(); i++) {
        if (frameFences[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device, frameFences[i], nullptr);
        }
    }
    
    for (size_t i = 0; i < releaseSemaphores.size(); i++) {
        if (releaseSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, releaseSemaphores[i], nullptr);
        }
    }
    
    for (uint32_t i = 0; i < acquireSemaphores.size(); i++) {
        if (acquireSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, acquireSemaphores[i], nullptr);
        }
    }
    
    for (auto imageView : swapchainImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, imageView, nullptr);
        }
    }
    
    if (depthImage.image != VK_NULL_HANDLE) {
        destroyImage(depthImage, device);
    }
    
    if (swapchain.swapchain != VK_NULL_HANDLE) {
        destroySwapchain(device, swapchain);
    }
    
    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(context.getInstance(), surface, nullptr);
    }
    
    if (window) {
        glfwDestroyWindow(window);
    }
    
    glfwTerminate();
}

void renderTransientExtraction(
    VulkanContext& context,
    MinMaxOutput& minMaxOutput,
    FilteringOutput& filteringResult,
    PushConstants& pushConstants,
    bool pmb
) {
    VkDevice device = context.getDevice();
    
    // Initialize all resources to null for proper cleanup
    GLFWwindow* window = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    Swapchain swapchain{};
    Image depthImage{};
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkSemaphore> acquireSemaphores;
    std::vector<VkSemaphore> releaseSemaphores;
    std::vector<VkFence> frameFences;
    std::vector<VkCommandBuffer> commandBuffers;
    
    try {
        // Create swapchain and window
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(1280, 720, "MeshTrex Transient Renderer", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("Failed to create window");
        }
        
        // Create surface
        surface = createSurface(context.getInstance(), window);
        
        // Create swapchain
        VkFormat swapchainFormat = getSwapchainFormat(context.getPhysicalDevice(), surface);
        createSwapchain(swapchain, context.getPhysicalDevice(), device, surface, context.getGraphicsQueueFamilyIndex(), window, swapchainFormat, VK_NULL_HANDLE);
        
        // Create depth buffer (with sampled bit for Hi-Z generation)
        VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
        createImage(depthImage, device, context.getMemoryProperties(), VK_IMAGE_TYPE_2D, swapchain.width, swapchain.height, 1, 1, depthFormat, 
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        
        // Create image views
        swapchainImageViews.resize(swapchain.imageCount);
        for (uint32_t i = 0; i < swapchain.imageCount; i++) {
            swapchainImageViews[i] = createImageView(device, swapchain.images[i], swapchainFormat, VK_IMAGE_TYPE_2D, 0, 1);
        }
    
        // Camera state - properly initialized for better viewing
        glm::vec3 volumeCenter = glm::vec3(128.0f, 128.0f, 128.0f);  // Typical center for 256^3 volumes
        float cameraDistance = 400.0f;
        float cameraYaw = -45.0f * 3.14159f / 180.0f;
        float cameraPitch = 30.0f * 3.14159f / 180.0f;
        
        glm::vec3 cameraPos;
        cameraPos.x = volumeCenter.x + cameraDistance * cos(cameraPitch) * cos(cameraYaw);
        cameraPos.y = volumeCenter.y + cameraDistance * sin(cameraPitch);
        cameraPos.z = volumeCenter.z + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
        
        glm::vec3 cameraTarget = volumeCenter;
        glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
        double lastMouseX = 0, lastMouseY = 0;
        float lastFrameTime = 0.0f;
        
        // Create synchronization objects
        // We need more acquire semaphores than swapchain images to avoid reuse conflicts
        acquireSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        // Release semaphores are per swapchain image
        releaseSemaphores.resize(swapchain.imageCount);
        
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            acquireSemaphores[i] = createSemaphore(device);
        }
        for (uint32_t i = 0; i < swapchain.imageCount; i++) {
            releaseSemaphores[i] = createSemaphore(device);
        }
        
        // Fences are per swapchain image
        frameFences.resize(swapchain.imageCount);
        for (uint32_t i = 0; i < swapchain.imageCount; i++) {
            VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frameFences[i]));
        }
        
        uint32_t currentFrame = 0;  // Track which set of sync objects to use
        
        // Allocate command buffers - one per frame to avoid conflicts
        commandBuffers.resize(swapchain.imageCount);
        VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.commandPool = context.getCommandPool();
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = swapchain.imageCount;
        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));
        
        // Main render loop
        while (!glfwWindowShouldClose(window)) {
            float currentTime = (float)glfwGetTime();
            float deltaTime = currentTime - lastFrameTime;
            lastFrameTime = currentTime;
            glfwPollEvents();
            
            // Camera controls - improved for better exploration
            const float zoomSpeed = 300.0f * deltaTime;  // Increased from 100
            const float panSpeed = 150.0f * deltaTime;   // Increased from 50
            
            // W/S: Zoom in/out
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                cameraDistance = glm::max(10.0f, cameraDistance - zoomSpeed);
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                cameraDistance = glm::min(1000.0f, cameraDistance + zoomSpeed);
            }
            
            // A/D: Pan camera target left/right
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                glm::vec3 forward = glm::normalize(cameraPos - cameraTarget);
                glm::vec3 right = glm::normalize(glm::cross(cameraUp, forward));
                cameraTarget -= right * panSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                glm::vec3 forward = glm::normalize(cameraPos - cameraTarget);
                glm::vec3 right = glm::normalize(glm::cross(cameraUp, forward));
                cameraTarget += right * panSpeed;
            }
            
            // Q/E: Pan camera target up/down  
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                cameraTarget.y -= panSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
                cameraTarget.y += panSpeed;
            }
            
            // Update camera position based on spherical coordinates
            cameraPos.x = cameraTarget.x + cameraDistance * cos(cameraPitch) * cos(cameraYaw);
            cameraPos.y = cameraTarget.y + cameraDistance * sin(cameraPitch);
            cameraPos.z = cameraTarget.z + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
            
            // Mouse controls for rotation
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                float deltaX = (float)(mouseX - lastMouseX) * 1.0f;  // Doubled from 0.5
                float deltaY = (float)(mouseY - lastMouseY) * 1.0f;  // Doubled from 0.5
                
                // Update yaw and pitch (increased sensitivity)
                cameraYaw -= deltaX * 0.02f;   // Doubled from 0.01
                cameraPitch -= deltaY * 0.02f; // Doubled from 0.01
                cameraPitch = glm::clamp(cameraPitch, -89.0f * 3.14159f / 180.0f, 89.0f * 3.14159f / 180.0f);
                
                // Update camera position
                cameraPos.x = cameraTarget.x + cameraDistance * cos(cameraPitch) * cos(cameraYaw);
                cameraPos.y = cameraTarget.y + cameraDistance * sin(cameraPitch);
                cameraPos.z = cameraTarget.z + cameraDistance * cos(cameraPitch) * sin(cameraYaw);
            }
            lastMouseX = mouseX;
            lastMouseY = mouseY;
            
            VK_CHECK(vkWaitForFences(device, 1, &frameFences[currentFrame], VK_TRUE, UINT64_MAX));
            VK_CHECK(vkResetFences(device, 1, &frameFences[currentFrame]));
            
            uint32_t imageIndex;
            VkResult result = vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, acquireSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
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
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
                
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
            
            // Reversed-Z projection matrix (infinite far plane version for better precision)
            glm::mat4 proj = glm::mat4(0.0f);
            float tanHalfFov = tan(fov / 2.0f);
            proj[0][0] = 1.0f / (aspect * tanHalfFov);
            proj[1][1] = 1.0f / tanHalfFov;
            proj[2][2] = 0.0f;  // Reversed-Z
            proj[2][3] = -1.0f;
            proj[3][2] = nearPlane;  // Reversed-Z
            glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0, 1, 0));
            glm::mat4 viewProj = proj * view;
            
            // Extract frustum planes
            TransientExtractionPushConstants renderConstants;
            renderConstants.viewProj = viewProj;
            extractFrustumPlanes(viewProj, renderConstants.frustumPlanes);
            
            // Use the current frame's command buffer
            VkCommandBuffer commandBuffer = commandBuffers[currentFrame];
            
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
            submitInfo.pWaitSemaphores = &acquireSemaphores[currentFrame];
            submitInfo.pWaitDstStageMask = &submitStageMask;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &releaseSemaphores[currentFrame];
            
            VK_CHECK(vkQueueSubmit(context.getQueue(), 1, &submitInfo, frameFences[currentFrame]));
            
            // Clean up temporary resources after submission
            // Note: Temp resources from extractAndRenderTransient are cleaned up internally
            
            // Present using the current frame's release semaphore
            VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &releaseSemaphores[currentFrame];
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapchain.swapchain;
            presentInfo.pImageIndices = &imageIndex;
            
            result = vkQueuePresentKHR(context.getQueue(), &presentInfo);
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                // Handle next iteration
            } else {
                assert(result == VK_SUCCESS);
            }
            
            // Advance to next frame
            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        vkDeviceWaitIdle(device);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in renderTransientExtraction: " << e.what() << std::endl;
    }
    
    // Cleanup - safe to call even if resources weren't created
    if (!commandBuffers.empty()) {
        vkFreeCommandBuffers(device, context.getCommandPool(), commandBuffers.size(), commandBuffers.data());
    }
    
    // Destroy all synchronization objects
    for (size_t i = 0; i < frameFences.size(); i++) {
        if (frameFences[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device, frameFences[i], nullptr);
        }
    }
    
    for (size_t i = 0; i < releaseSemaphores.size(); i++) {
        if (releaseSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, releaseSemaphores[i], nullptr);
        }
    }
    
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (i < acquireSemaphores.size() && acquireSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, acquireSemaphores[i], nullptr);
        }
    }
    
    for (auto& view : swapchainImageViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
        }
    }
    
    if (depthImage.image != VK_NULL_HANDLE) {
        destroyImage(depthImage, device);
    }
    
    if (swapchain.swapchain != VK_NULL_HANDLE) {
        destroySwapchain(device, swapchain);
    }
    
    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(context.getInstance(), surface, nullptr);
    }
    
    if (window) {
        glfwDestroyWindow(window);
    }
    
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
        bool disableCoherenceOptimization = false;
        bool useDensityDispatch = false;
        bool useTransientExtraction = true;
        bool useTemporalCoherence = true;
        std::string volumePath = getFullPath(ROOT_BUILD_PATH, "/raw_volumes/bonsai_256x256x256_uint8.raw");
        float isovalue = 80;
        bool requestMeshShading = false;
        
        // Synthetic data options
        bool useSyntheticData = true;
        std::string syntheticType = "stress";  // "random", "layered", or "stress"
        int numSpheres = 1000;
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
            } else if (strcmp(argv[i], "--temporal") == 0) {
                useTemporalCoherence = true;
                useTransientExtraction = true; // Temporal coherence requires transient extraction
                std::cout << "Temporal coherence rendering enabled (Kreskowski's two-pass approach)" << std::endl;
            } else if (strcmp(argv[i], "--volume") == 0 && i + 1 < argc) {
                volumePath = argv[++i];
            } else if (strcmp(argv[i], "--isovalue") == 0 && i + 1 < argc) {
                isovalue = std::stof(argv[++i]);
            } else if (strcmp(argv[i], "--synthetic") == 0) {
                useSyntheticData = true;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    syntheticType = argv[++i];  // "random", "layered", or "stress"
                }
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    numSpheres = std::stoi(argv[++i]);
                }
                std::cout << "Using synthetic " << syntheticType << " sphere data with " 
                         << numSpheres << " spheres" << std::endl;
            } else if (strcmp(argv[i], "--help") == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                         << "Options:\n"
                         << "  --density-dispatch    Enable density-based dispatch\n"
                         << "  --transient          Enable transient extraction (on-the-fly rendering)\n"
                         << "  --temporal           Enable temporal coherence rendering (two-pass approach)\n"
                         << "  --no-coherence-opt   Disable temporal coherence optimization (force occlusion test every frame)\n"
                         << "  --volume <path>      Path to volume file\n"
                         << "  --isovalue <value>   Isovalue for surface extraction\n"
                         << "  --synthetic [type] [count]  Use synthetic sphere data\n"
                         << "                       type: random, layered, or stress (default: random)\n"
                         << "                       count: number of spheres (default: 1000)\n"
                         << "  --help               Show this help message\n"
                         << "\nRuntime controls:\n"
                         << "  T                    Toggle temporal coherence optimization on/off\n"
                         << "  O                    Toggle occlusion method (Raster vs Compute Hi-Z)\n"
                         << "  F                    Freeze/unfreeze PVS updates (debug visualization)\n"
                         << "  R                    Reset temporal state\n"
                         << "  C                    Toggle debug colors\n"
                         << "  W/S                  Zoom in/out\n"
                         << "  A/D                  Pan left/right\n"
                         << "  Q/E                  Pan up/down\n"
                         << "  Mouse drag           Rotate camera\n";
                return 0;
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

        Volume volume;
        
        if (useSyntheticData) {
            // Generate synthetic sphere data for testing occlusion
            int volumeSize = 256;  // Use 256^3 for good testing
            
            if (syntheticType == "layered") {
                std::cout << "Generating layered sphere volume (" << volumeSize << "^3)..." << std::endl;
                volume = {
                    glm::vec3(volumeSize, volumeSize, volumeSize),
                    "uint_8",
                    MultiSphereVolumeGenerator::generateLayeredSphereVolume(
                        volumeSize, volumeSize, volumeSize,
                        numSpheres / 10,  // spheres per layer
                        10  // number of layers
                    )
                };
            } else if (syntheticType == "stress") {
                std::cout << "Generating occlusion stress test volume (" << volumeSize << "^3)..." << std::endl;
                volume = {
                    glm::vec3(volumeSize, volumeSize, volumeSize),
                    "uint_8",
                    MultiSphereVolumeGenerator::generateOcclusionStressTest(
                        volumeSize, volumeSize, volumeSize
                    )
                };
            } else {  // "random" or default
                std::cout << "Generating random sphere volume (" << volumeSize << "^3) with " 
                         << numSpheres << " spheres..." << std::endl;
                volume = {
                    glm::vec3(volumeSize, volumeSize, volumeSize),
                    "uint_8",
                    MultiSphereVolumeGenerator::generateMultiSphereVolume(
                        volumeSize, volumeSize, volumeSize,
                        numSpheres,
                        8.0f,   // minRadius - increased from 2.0f
                        20.0f,  // maxRadius - increased from 8.0f
                        true,   // randomPlacement
                        true    // allowOverlap
                    )
                };
            }
            std::cout << "Synthetic volume generated successfully.";
        } else {
            volume = loadVolume(volumePath.c_str());
            std::cout << "Volume " << volumePath.c_str() << " is loaded.";
        }
        
        PushConstants pushConstants = {};
        pushConstants.volumeDim = glm::uvec4(volume.volume_dims, 1);
        pushConstants.blockDim = glm::uvec4(8, 8, 8, 1);
        pushConstants.blockGridDim = glm::uvec4(
            (volume.volume_dims.x + pushConstants.blockDim.x - 1) / pushConstants.blockDim.x,
            (volume.volume_dims.y + pushConstants.blockDim.y - 1) / pushConstants.blockDim.y,
            (volume.volume_dims.z + pushConstants.blockDim.z - 1) / pushConstants.blockDim.z,
            1);
        pushConstants.isovalue = isovalue / 255.0f;  // Convert to normalized float

        std::cout << "Loaded volume dims: ("
                  << pushConstants.volumeDim.x << "x" << pushConstants.volumeDim.y << "x" << pushConstants.volumeDim.z << ")" << std::endl;
        std::cout << "Block grid: " << pushConstants.blockGridDim.x << "x" << pushConstants.blockGridDim.y << "x" << pushConstants.blockGridDim.z << std::endl;
        std::cout << "Isovalue: " << isovalue << " -> normalized: " << pushConstants.isovalue << " (for sphere surface)" << std::endl;

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
                
                // Don't cleanup here - resources are still needed for rendering
                // Cleanup will happen after rendering completes
                                
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
                    if (useTemporalCoherence) {
                        // Temporal coherence rendering with two-pass approach
                        std::cout << "\n--- Starting Temporal Coherence Renderer ---" << std::endl;
                        renderTemporalCoherence(context, minMaxOutput, pushConstants, pmb, disableCoherenceOptimization);
                    } else {
                        // Standard transient extraction - render on-the-fly without storing geometry
                        std::cout << "\n--- Starting Transient Renderer ---" << std::endl;
                        renderTransientExtraction(context, minMaxOutput, filteringResult, pushConstants, pmb);
                    }
                    
                    // Cleanup min-max and filtering results after transient rendering
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                } else if (extractionResultGPU.meshletCount > 0) {
                    std::cout << "\n--- Starting Persistent Renderer ---" << std::endl;
                    RenderingManager renderingManager(context);
                    renderingManager.render(extractionResultGPU);
                    
                    // Cleanup min-max and filtering results after persistent rendering
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                } else {
                    std::cout << "\nSkipping rendering as no meshlets were generated." << std::endl;
                    
                    // Still need to cleanup even if no rendering happened
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
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
                    if (useTemporalCoherence) {
                        // Temporal coherence rendering with two-pass approach
                        std::cout << "\n--- Starting Temporal Coherence Renderer ---" << std::endl;
                        renderTemporalCoherence(context, minMaxOutput, pushConstants, pmb, disableCoherenceOptimization);
                    } else {
                        // Standard transient extraction - render on-the-fly without storing geometry
                        std::cout << "\n--- Starting Transient Renderer ---" << std::endl;
                        renderTransientExtraction(context, minMaxOutput, filteringResult, pushConstants, pmb);
                    }
                    
                    // Cleanup min-max and filtering results after transient rendering
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                } else if (extractionResultGPU.meshletCount > 0) {
                    std::cout << "\n--- Starting Persistent Renderer ---" << std::endl;
                    RenderingManager renderingManager(context);
                    renderingManager.render(extractionResultGPU);
                    
                    // Cleanup min-max and filtering results after persistent rendering
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                } else {
                    std::cout << "\nSkipping rendering as no meshlets were generated." << std::endl;
                    
                    // Still need to cleanup even if no rendering happened
                    minMaxOutput.cleanup(context.getDevice());
                    filteringResult.cleanup(context.getDevice());
                }
            } catch (std::exception& e) {
                std::cout << e.what() << std::endl;
                // Cleanup resources in case of exception
                minMaxOutput.cleanup(context.getDevice());
                filteringResult.cleanup(context.getDevice());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}