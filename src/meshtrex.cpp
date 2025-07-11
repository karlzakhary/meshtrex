#include <fstream>
#include <iostream>
#include <regex>
#include <thread>
#include <set>
#include <chrono>

#ifndef __APPLE__
#include <cstdint>
#endif

#include "common.h"
#include "minMaxManager.h"
#include "filteringManager.h"
#include "extractionManager.h"
#include "streamingShaderInterface.h"
#include "blockFilteringTestUtils.h"
#include "extractionTestUtils.h"
#include "testMinMax.h"
#include "image.h"
#include "buffer.h"
#include <dlfcn.h>
#include "renderdoc_app.h"
#include "vulkan_context.h"
#include "streamingSystem.h"
#include "persistentGeometryExtraction.h"
#include "deviceGeneratedCommands.h"
#include "autonomousDGCSystem.h"

RENDERDOC_API_1_1_2 *rdoc_api = NULL;

// Helper function to read back image data from GPU
void readImageData(VulkanContext& context, VkImage image, VkFormat format,
    uint32_t width, uint32_t height, uint32_t depth,
    void* data, size_t dataSize)
{
    Buffer stagingBuffer = {};
    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(),
        dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());

    // Transition image to transfer source
    VkImageMemoryBarrier2 barrier = imageBarrier(
        image,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1
    );
    pipelineBarrier(cmd, 0, 0, nullptr, 1, &barrier);

    // Copy image to buffer
    VkExtent3D extent = { width, height, depth };
    copy3DImageTo1DBuffer(stagingBuffer, cmd, image, extent);

    // Transition back to shader read-only
    barrier = imageBarrier(
        image,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1
    );
    pipelineBarrier(cmd, 0, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);

    memcpy(data, stagingBuffer.data, dataSize);
    destroyBuffer(stagingBuffer, context.getDevice());
}

std::vector<uint8_t> generateSphereVolume(int width, int height, int depth) {
    std::vector<uint8_t> data(width * height * depth);
    float radius = width * 0.4f;
    float centerX = width / 2.0f;
    float centerY = height / 2.0f;
    float centerZ = depth / 2.0f;

    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float dx = x - centerX;
                float dy = y - centerY;
                float dz = z - centerZ;
                float distance = sqrt(dx * dx + dy * dy + dz * dz) - radius;
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
        void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (!mod) {
            mod = dlopen("librenderdoc.so.1", RTLD_NOW | RTLD_NOLOAD);
        }
        if (!mod) {
            mod = dlopen("/usr/lib/librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        }

        if (mod) {
            pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
            if (RENDERDOC_GetAPI) {
                int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void**)&rdoc_api);
                if (ret == 1) {
                    std::cout << "RenderDoc API loaded successfully" << std::endl;
                }
                else {
                    std::cout << "Failed to get RenderDoc API" << std::endl;
                }
            }
        }
        else {
            std::cout << "RenderDoc library not found: " << dlerror() << std::endl;
        }

        VulkanContext context(requestMeshShading);
        Volume volume = loadVolume(volumePath.c_str());
        std::cout << "Volume " << volumePath.c_str() << " is loaded." << std::endl;
        std::cout << "--- Debug: Printing C++ volume data values > 41 for cell (0,0,0) of block (0,0,0) ---" << std::endl;
        std::cout << "--- End Debug: C++ specific volume data print ---" << std::endl;

        std::cout << "\n=== CPU Min-Max Analysis ===" << std::endl;
        uint32_t nonZeroCount = 0;
        uint32_t valuesAboveIso = 0;
        for (size_t i = 0; i < volume.volume_data.size(); i++) {
            if (volume.volume_data[i] > 0) nonZeroCount++;
            if (volume.volume_data[i] > isovalue) valuesAboveIso++;
        }
        std::cout << "Volume stats: " << nonZeroCount << " non-zero voxels, "
            << valuesAboveIso << " above isovalue " << isovalue << std::endl;

        testStreamingMinMaxForPage(
            volume.volume_data,
            volume.volume_dims.x,
            volume.volume_dims.y,
            volume.volume_dims.z,
            4, 0, 0, 0, 64, 32, 32, isovalue
        );
        testStreamingMinMaxForPage(
            volume.volume_data,
            volume.volume_dims.x,
            volume.volume_dims.y,
            volume.volume_dims.z,
            4, 1, 1, 1, 64, 32, 32, isovalue
        );

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
        std::cout << "Using isovalue: " << isovalue << std::endl;

        try {
            std::regex dimRegex(R"((\d+)x(\d+)x(\d+))");
            std::smatch match;
            uint32_t volumeSizeX = 256, volumeSizeY = 256, volumeSizeZ = 256;
            if (std::regex_search(volumePath, match, dimRegex)) {
                volumeSizeX = std::stoi(match[1]);
                volumeSizeY = std::stoi(match[2]);
                volumeSizeZ = std::stoi(match[3]);
            }

            StreamingParams streamParams;
            streamParams.pageSizeX = 64;
            streamParams.pageSizeY = 32;
            streamParams.pageSizeZ = 32;
            streamParams.atlasSizeX = 1024;
            streamParams.atlasSizeY = 1024;
            streamParams.atlasSizeZ = 1024;
            streamParams.maxResidentPages = 4096;

            std::cout << "Creating autonomous streaming components..." << std::endl;
            VolumeStreamer volumeStreamer(context, streamParams);
            PersistentGeometryBuffers persistentBuffers(context);
            DeviceGeneratedCommands dgcCommands(context);

            volumeStreamer.loadVolume(volumePath);
            persistentBuffers.initialize();
            dgcCommands.initialize();

            AutonomousDGCConfig dgcConfig = {};
            dgcConfig.volumeDimX = volumeSizeX;
            dgcConfig.volumeDimY = volumeSizeY;
            dgcConfig.volumeDimZ = volumeSizeZ;
            dgcConfig.pageSizeX = streamParams.pageSizeX;
            dgcConfig.pageSizeY = streamParams.pageSizeY;
            dgcConfig.pageSizeZ = streamParams.pageSizeZ;
            dgcConfig.maxResidentPages = streamParams.maxResidentPages;
            dgcConfig.targetMemoryUsage = 80;
            dgcConfig.maxCommandsPerFrame = 256;
            dgcConfig.viewDistanceThreshold = 500.0f;
            dgcConfig.evictionDistanceScale = 2.0f;

            uint32_t pagesX = (volumeSizeX + streamParams.pageSizeX - 1) / streamParams.pageSizeX;
            uint32_t pagesY = (volumeSizeY + streamParams.pageSizeY - 1) / streamParams.pageSizeY;
            uint32_t pagesZ = (volumeSizeZ + streamParams.pageSizeZ - 1) / streamParams.pageSizeZ;
            uint32_t totalPages = pagesX * pagesY * pagesZ;
            dgcConfig.prioritizeVisible = (totalPages > dgcConfig.maxResidentPages);

            AutonomousDGCSystem autonomousSystem(context, &dgcCommands);
            autonomousSystem.initialize(dgcConfig);
            autonomousSystem.setExtractionParameter(isovalue);

            static uint32_t pagesWithMinMaxMismatches = 0;
            static uint32_t totalBlocksChecked = 0;
            static uint32_t totalBlockMismatches = 0;

            std::cout << "System initialized successfully" << std::endl;
            std::cout << "  Volume: " << volumeSizeX << "x" << volumeSizeY << "x" << volumeSizeZ << std::endl;
            std::cout << "  Total pages: " << totalPages << std::endl;
            std::cout << "  Max resident: " << dgcConfig.maxResidentPages << std::endl;
            std::cout << "  Volume " << (totalPages <= dgcConfig.maxResidentPages ? "FITS" : "DOES NOT FIT") << " in memory" << std::endl;

            glm::mat4 view = glm::lookAt(glm::vec3(300.0f, 300.0f, 300.0f), glm::vec3(128.0f, 128.0f, 128.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 1000.0f);
            glm::vec3 cameraPos(300.0f, 300.0f, 300.0f);
            autonomousSystem.updateViewParameters(view, proj, cameraPos);

            std::cout << "\nStarting autonomous streaming execution..." << std::endl;
            std::cout << "Creating streaming pipeline infrastructure..." << std::endl;

            StreamingDGCManager streamingDGCManager(context);
            streamingDGCManager.initialize();
            std::set<PageCoord> extractedPages;
            std::map<PageCoord, uint32_t> pageActiveBlocks;
            uint32_t frameIndex = 0;

            while (!autonomousSystem.isExtractionComplete()) {
                std::cout << "\n--- Frame " << frameIndex << " ---" << std::endl;
                uint32_t residentPageCount = 0;
                uint32_t shownCount = 0;
                for (uint32_t z = 0; z < pagesZ; z++) {
                    for (uint32_t y = 0; y < pagesY; y++) {
                        for (uint32_t x = 0; x < pagesX; x++) {
                            if (volumeStreamer.isPageResident({ x, y, z })) {
                                if (shownCount < 10) {
                                    shownCount++;
                                }
                                residentPageCount++;
                            }
                        }
                    }
                }
                std::cout << "Total resident pages: " << residentPageCount << std::endl;

                persistentBuffers.beginFrame(frameIndex);
                volumeStreamer.updateStreaming(frameIndex);
                volumeStreamer.processPendingBindings();
                volumeStreamer.processPendingCopies();

                VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
                autonomousSystem.executeAutonomousFrame(cmd, frameIndex);

                uint32_t pagesProcessedThisFrame = 0;
                std::cout << "Volume size: " << volumeSizeX << "x" << volumeSizeY << "x" << volumeSizeZ << std::endl;
                std::cout << "Total pages: " << pagesX << "x" << pagesY << "x" << pagesZ << " = " << (pagesX * pagesY * pagesZ) << std::endl;

                static bool initialRequestDone = false;
                if (!initialRequestDone) {
                    std::cout << "Initial request for all pages: " << pagesX << "x" << pagesY << "x" << pagesZ
                        << " = " << (pagesX * pagesY * pagesZ) << " total pages" << std::endl;
                    for (uint32_t z = 0; z < pagesZ; z++) {
                        for (uint32_t y = 0; y < pagesY; y++) {
                            for (uint32_t x = 0; x < pagesX; x++) {
                                PageCoord pageCoord = { x, y, z, 0 };
                                volumeStreamer.requestPage(pageCoord);
                            }
                        }
                    }
                    initialRequestDone = true;
                }

                if (frameIndex == 0) {
                    std::cout << "Waiting for initial page loading..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                }
                else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                volumeStreamer.processPendingBindings();
                volumeStreamer.processPendingCopies();
                volumeStreamer.updateStreaming(frameIndex);
                VK_CHECK(vkQueueWaitIdle(context.getQueue()));

                std::vector<PageCoord> pagesToProcess;
                for (uint32_t z = 0; z < pagesZ; z++) {
                    for (uint32_t y = 0; y < pagesY; y++) {
                        for (uint32_t x = 0; x < pagesX; x++) {
                            PageCoord pageCoord = { x, y, z, 0 };
                            if (volumeStreamer.isPageResident(pageCoord)) {
                                // Process page immediately without checking neighbors
                                // This will cause cross-page sampling issues
                                pagesToProcess.push_back(pageCoord);
                            }
                        }
                    }
                }

                uint32_t unextractedCount = 0;
                for (const auto& page : pagesToProcess) {
                    if (extractedPages.find(page) == extractedPages.end()) {
                        unextractedCount++;
                    }
                }

                // No longer tracking pages waiting for neighbors since we process immediately

                std::cout << "Found " << pagesToProcess.size() << " pages ready, "
                    << unextractedCount << " not yet extracted" << std::endl;
                if (unextractedCount == 0 && extractedPages.size() < pagesX* pagesY* pagesZ) {
                    std::cout << "No pages to process this frame, waiting for more pages to load..." << std::endl;
                    std::cout << "Extracted: " << extractedPages.size() << "/" << (pagesX * pagesY * pagesZ)
                        << " pages" << std::endl;
                    frameIndex++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                uint32_t totalActiveBlocks = 0;
                uint32_t pagesProcessed = 0;
                for (const auto& pageCoord : pagesToProcess) {
                    if (extractedPages.find(pageCoord) != extractedPages.end()) {
                        continue;
                    }

                    VkDescriptorSet streamingDescriptorSet = volumeStreamer.getStreamingDescriptorSet();
                    PushConstants pagePushConstants = pushConstants;
                    pagePushConstants.volumeDim = glm::uvec4(volumeSizeX, volumeSizeY, volumeSizeZ, 1);
                    pagePushConstants.blockDim = glm::uvec4(4, 4, 4, 1);
                    pagePushConstants.blockGridDim = glm::uvec4(
                        streamParams.pageSizeX / 4,
                        streamParams.pageSizeY / 4,
                        streamParams.pageSizeZ / 4,
                        1);

                    std::cout << "  Computing min-max for page with grid " << pagePushConstants.blockGridDim.x
                        << "x" << pagePushConstants.blockGridDim.y << "x" << pagePushConstants.blockGridDim.z << std::endl;
                    
                    MinMaxOutput minMaxOutput = computeStreamingMinMaxMip(
                        context,
                        volumeStreamer.getVolumeAtlasView(),
                        volumeStreamer.getVolumeSampler(),
                        volumeStreamer.getPageTableBuffer(),
                        pageCoord,
                        pagePushConstants
                    );
                    VK_CHECK(vkQueueWaitIdle(context.getQueue()));

                    {
                        std::cout << "\n  Comparing Min-Max for page (" << pageCoord.x << "," << pageCoord.y << "," << pageCoord.z << ")" << std::endl;
                        uint32_t blocksPerPageX = streamParams.pageSizeX / 4;
                        uint32_t blocksPerPageY = streamParams.pageSizeY / 4;
                        uint32_t blocksPerPageZ = streamParams.pageSizeZ / 4;
                        size_t minMaxDataSize = blocksPerPageX * blocksPerPageY * blocksPerPageZ * sizeof(uint32_t) * 2;
                        std::vector<uint32_t> gpuMinMaxData(blocksPerPageX* blocksPerPageY* blocksPerPageZ * 2);
                        readImageData(context, minMaxOutput.minMaxImage.image, VK_FORMAT_R32G32_UINT,
                            blocksPerPageX, blocksPerPageY, blocksPerPageZ,
                            gpuMinMaxData.data(), minMaxDataSize);

                        uint32_t pageStartX = pageCoord.x * streamParams.pageSizeX;
                        uint32_t pageStartY = pageCoord.y * streamParams.pageSizeY;
                        uint32_t pageStartZ = pageCoord.z * streamParams.pageSizeZ;
                        std::cout << "    Checking all blocks in this page..." << std::endl;
                        uint32_t pageBlockMismatches = 0;
                        uint32_t activeBlocksGPU = 0;
                        uint32_t activeBlocksCPU = 0;

                        for (uint32_t bz = 0; bz < blocksPerPageZ; bz++) {
                            for (uint32_t by = 0; by < blocksPerPageY; by++) {
                                for (uint32_t bx = 0; bx < blocksPerPageX; bx++) {
                                    uint32_t blockIdx = bz * blocksPerPageX * blocksPerPageY + by * blocksPerPageX + bx;
                                    uint32_t gpuMin = gpuMinMaxData[blockIdx * 2];
                                    uint32_t gpuMax = gpuMinMaxData[blockIdx * 2 + 1];
                                    uint32_t blockWorldX = pageStartX + bx * 4;
                                    uint32_t blockWorldY = pageStartY + by * 4;
                                    uint32_t blockWorldZ = pageStartZ + bz * 4;
                                    uint32_t cpuMin = 255;
                                    uint32_t cpuMax = 0;
                                    for (uint32_t dz = 0; dz <= 4; dz++) {
                                        for (uint32_t dy = 0; dy <= 4; dy++) {
                                            for (uint32_t dx = 0; dx <= 4; dx++) {
                                                uint32_t worldX = blockWorldX + dx;
                                                uint32_t worldY = blockWorldY + dy;
                                                uint32_t worldZ = blockWorldZ + dz;
                                                if (worldX >= volumeSizeX || worldY >= volumeSizeY || worldZ >= volumeSizeZ) {
                                                    continue;
                                                }
                                                uint32_t idx = worldZ * volumeSizeY * volumeSizeX + worldY * volumeSizeX + worldX;
                                                uint32_t value = volume.volume_data[idx];
                                                cpuMin = std::min(cpuMin, value);
                                                cpuMax = std::max(cpuMax, value);
                                            }
                                        }
                                    }
                                    if (gpuMin != cpuMin || gpuMax != cpuMax) {
                                        pageBlockMismatches++;
                                        if (pageBlockMismatches <= 5) {
                                            std::cout << "      Block (" << bx << "," << by << "," << bz << "): "
                                                << "GPU[" << gpuMin << "," << gpuMax << "] vs CPU[" << cpuMin << "," << cpuMax << "]" << std::endl;
                                        }
                                    }
                                    if (gpuMin <= isovalue && gpuMax >= isovalue) activeBlocksGPU++;
                                    if (cpuMin <= isovalue && cpuMax >= isovalue) activeBlocksCPU++;
                                    totalBlocksChecked++;
                                }
                            }
                        }
                        totalBlockMismatches += pageBlockMismatches;
                        if (pageBlockMismatches > 0) {
                            pagesWithMinMaxMismatches++;
                        }
                        std::cout << "    Page summary: " << pageBlockMismatches << " mismatches out of "
                            << (blocksPerPageX * blocksPerPageY * blocksPerPageZ) << " blocks" << std::endl;
                        std::cout << "    Active blocks - GPU: " << activeBlocksGPU << ", CPU: " << activeBlocksCPU << std::endl;
                    }

                    FilteringOutput filteringOutput = filterStreamingActiveBlocks(
                        context,
                        minMaxOutput,
                        volumeStreamer.getPageTableBuffer(),
                        pageCoord,
                        pagePushConstants
                    );

                    static std::set<uint32_t> seenCounts;
                    if (seenCounts.find(filteringOutput.activeBlockCount) == seenCounts.end() || filteringOutput.activeBlockCount > 100) {
                        std::cout << "  Page (" << pageCoord.x << "," << pageCoord.y << "," << pageCoord.z
                            << ") filtering result: " << filteringOutput.activeBlockCount << " active blocks"
                            << " (max possible: " << (pagePushConstants.blockGridDim.x* pagePushConstants.blockGridDim.y* pagePushConstants.blockGridDim.z) << ")" << std::endl;
                        seenCounts.insert(filteringOutput.activeBlockCount);
                    }
                    totalActiveBlocks += filteringOutput.activeBlockCount;

                    if (filteringOutput.activeBlockCount > 0) {
                        static uint32_t pagesWithActiveBlocks = 0;
                        pagesWithActiveBlocks++;
                        if (pagesWithActiveBlocks <= 10 || filteringOutput.activeBlockCount > 50) {
                            std::cout << "  Page (" << pageCoord.x << "," << pageCoord.y << "," << pageCoord.z
                                << ") has " << filteringOutput.activeBlockCount << " active blocks" << std::endl;
                        }
                    }

                    pageActiveBlocks[pageCoord] = filteringOutput.activeBlockCount;
                    if (filteringOutput.activeBlockCount > 0) {
                        StreamingExtractionConstants extractConstants;
                        extractConstants.pageCoord = glm::uvec3(pageCoord.x, pageCoord.y, pageCoord.z);
                        extractConstants.mipLevel = 0;
                        extractConstants.isoValue = isovalue;
                        extractConstants.blockSize = 4;
                        extractConstants.pageSizeX = streamParams.pageSizeX;
                        extractConstants.pageSizeY = streamParams.pageSizeY;
                        extractConstants.pageSizeZ = streamParams.pageSizeZ;

                        PersistentExtractionResult extractResult =
                            persistentBuffers.extractPageToGlobalBuffers(
                                pageCoord,
                                extractConstants,
                                filteringOutput,
                                volumeStreamer.getVolumeAtlasView(),
                                volumeStreamer.getPageTableBuffer(),
                                volumeSizeX,
                                volumeSizeY,
                                volumeSizeZ
                            );
                        if (extractResult.success) {
                            pagesProcessedThisFrame++;
                            pagesProcessed++;
                            std::cout << "  Page (" << pageCoord.x << "," << pageCoord.y << ","
                                << pageCoord.z << ") extracted: "
                                << extractResult.verticesGenerated << " vertices, "
                                << extractResult.indicesGenerated << " indices, "
                                << extractResult.meshletsGenerated << " meshlets" << std::endl;
                            extractedPages.insert(pageCoord);
                        }
                    }
                    else {
                        std::cout << "  Page (" << pageCoord.x << "," << pageCoord.y << ","
                            << pageCoord.z << ") has no active blocks, skipping extraction" << std::endl;
                        extractedPages.insert(pageCoord);
                    }
                    filteringOutput.cleanup(context.getDevice());
                    minMaxOutput.cleanup(context.getDevice());
                }

                std::cout << "\nSummary: Processed " << pagesProcessed << " pages this frame with "
                    << totalActiveBlocks << " total active blocks" << std::endl;
                std::cout << "Total extracted pages: " << extractedPages.size() << " / " << (pagesX * pagesY * pagesZ) << std::endl;
                static uint32_t cumulativeActiveBlocks = 0;
                cumulativeActiveBlocks += totalActiveBlocks;
                std::cout << "Cumulative active blocks across all frames: " << cumulativeActiveBlocks << std::endl;

                if (extractedPages.size() == pagesX * pagesY * pagesZ || frameIndex % 10 == 0) {
                    std::cout << "\nExtraction coverage map (X=extracted, .=not extracted):" << std::endl;
                    for (uint32_t z = 0; z < pagesZ && z < 8; z++) {
                        std::cout << "Z=" << z << ": ";
                        for (uint32_t y = 0; y < pagesY && y < 8; y++) {
                            for (uint32_t x = 0; x < pagesX && x < 8; x++) {
                                PageCoord pc = { x, y, z, 0 };
                                std::cout << (extractedPages.find(pc) != extractedPages.end() ? "X" : ".");
                            }
                            std::cout << " ";
                        }
                        std::cout << std::endl;
                    }
                }

                if (extractedPages.size() >= pagesX * pagesY * pagesZ) {
                    std::cout << "\nAll pages extracted! Extraction complete." << std::endl;
                    uint32_t pagesWithGeometry = 0;
                    uint32_t totalBlocksWithGeometry = 0;
                    uint32_t totalActiveBlocksGPU = 0;
                    for (const auto& [page, blocks] : pageActiveBlocks) {
                        totalActiveBlocksGPU += blocks;
                        if (blocks > 0) {
                            pagesWithGeometry++;
                            totalBlocksWithGeometry += blocks;
                        }
                    }
                    std::cout << "Pages with geometry: " << pagesWithGeometry << " / " << (pagesX * pagesY * pagesZ) << std::endl;
                    std::cout << "Total active blocks (GPU): " << totalActiveBlocksGPU << std::endl;
                    std::cout << "\n=== Min-Max Comparison Summary ===" << std::endl;
                    std::cout << "Total blocks checked: " << totalBlocksChecked << std::endl;
                    std::cout << "Total block mismatches: " << totalBlockMismatches << std::endl;
                    std::cout << "Pages with mismatches: " << pagesWithMinMaxMismatches << std::endl;
                    std::cout << "\n=== CPU vs GPU Active Block Comparison ===" << std::endl;
                    uint32_t totalActiveBlocksCPU = 0;
                    uint32_t blocksX = (volumeSizeX + 3) / 4;
                    uint32_t blocksY = (volumeSizeY + 3) / 4;
                    uint32_t blocksZ = (volumeSizeZ + 3) / 4;
                    for (uint32_t bz = 0; bz < blocksZ; bz++) {
                        for (uint32_t by = 0; by < blocksY; by++) {
                            for (uint32_t bx = 0; bx < blocksX; bx++) {
                                uint32_t blockWorldX = bx * 4;
                                uint32_t blockWorldY = by * 4;
                                uint32_t blockWorldZ = bz * 4;
                                uint32_t minVal = 255;
                                uint32_t maxVal = 0;
                                for (uint32_t dz = 0; dz <= 4; dz++) {
                                    for (uint32_t dy = 0; dy <= 4; dy++) {
                                        for (uint32_t dx = 0; dx <= 4; dx++) {
                                            uint32_t worldX = blockWorldX + dx;
                                            uint32_t worldY = blockWorldY + dy;
                                            uint32_t worldZ = blockWorldZ + dz;
                                            if (worldX >= volumeSizeX || worldY >= volumeSizeY || worldZ >= volumeSizeZ) {
                                                continue;
                                            }
                                            uint32_t idx = worldZ * volumeSizeY * volumeSizeX + worldY * volumeSizeX + worldX;
                                            uint32_t value = volume.volume_data[idx];
                                            minVal = std::min(minVal, value);
                                            maxVal = std::max(maxVal, value);
                                        }
                                    }
                                }
                                if (minVal <= isovalue && maxVal >= isovalue) {
                                    totalActiveBlocksCPU++;
                                }
                            }
                        }
                    }
                    std::cout << "Total active blocks (CPU): " << totalActiveBlocksCPU << std::endl;
                    std::cout << "Total blocks in volume: " << (blocksX* blocksY* blocksZ) << std::endl;
                    std::cout << "Difference (GPU - CPU): " << (int32_t)totalActiveBlocksGPU - (int32_t)totalActiveBlocksCPU << std::endl;
                    if (totalActiveBlocksGPU == totalActiveBlocksCPU) {
                        std::cout << "SUCCESS: GPU and CPU results match!" << std::endl;
                    }
                    else {
                        std::cout << "WARNING: GPU and CPU results differ!" << std::endl;
                    }
                    break;
                }

                if (frameIndex > 50 && extractedPages.size() < pagesX* pagesY* pagesZ) {
                    std::cout << "\nDebugging stuck extraction - checking unextracted pages:" << std::endl;
                    uint32_t missingCount = 0;
                    for (uint32_t z = 0; z < pagesZ && missingCount < 10; z++) {
                        for (uint32_t y = 0; y < pagesY && missingCount < 10; y++) {
                            for (uint32_t x = 0; x < pagesX && missingCount < 10; x++) {
                                PageCoord pc = { x, y, z, 0 };
                                if (extractedPages.find(pc) == extractedPages.end()) {
                                    bool isResident = volumeStreamer.isPageResident(pc);
                                    std::cout << "  Page (" << x << "," << y << "," << z << ") - resident: "
                                        << (isResident ? "YES" : "NO") << std::endl;
                                    missingCount++;
                                }
                            }
                        }
                    }
                    if (missingCount == 0) {
                        std::cout << "ERROR: No missing pages found but count doesn't match!" << std::endl;
                    }
                    break;
                }

                endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);

                if (pagesProcessedThisFrame > 0) {
                }

                persistentBuffers.endFrame();
                float progress = autonomousSystem.getExtractionProgress();
                uint32_t residentPages = autonomousSystem.getResidentPageCount();
                std::cout << "Progress: " << (int)(progress * 100) << "%"
                    << " | Resident pages: " << residentPages
                    << " | Meshlets: " << persistentBuffers.getCurrentMeshletCount() << std::endl;
                frameIndex++;
                if (frameIndex > 100) {
                    std::cout << "\nReached frame limit, stopping..." << std::endl;
                    break;
                }
            }

            std::cout << "\n=== Final Statistics ===" << std::endl;
            std::cout << "Total meshlets extracted: " << persistentBuffers.getCurrentMeshletCount() << std::endl;
            std::cout << "Total vertices: " << persistentBuffers.getCurrentVertexCount() << std::endl;
            std::cout << "Total indices: " << persistentBuffers.getCurrentIndexCount() << std::endl;
            std::cout << "Extraction complete: " << (autonomousSystem.isExtractionComplete() ? "Yes" : "No") << std::endl;

            if (persistentBuffers.getCurrentVertexCount() > 0) {
                std::cout << "\nWriting extracted geometry to OBJ file..." << std::endl;
                std::string objPath = std::string(ROOT_BUILD_PATH) + "/autonomous_extraction.obj";
                persistentBuffers.exportToOBJ(objPath);
                ExtractionOutput extractionOutput;
                extractionOutput.vertexBuffer = persistentBuffers.getGlobalVertexBuffer();
                extractionOutput.indexBuffer = persistentBuffers.getGlobalIndexBuffer();
                extractionOutput.meshletDescriptorBuffer = persistentBuffers.getGlobalMeshletBuffer();
                extractionOutput.meshletCount = persistentBuffers.getCurrentMeshletCount();
                extractionOutput.indexCount = persistentBuffers.getCurrentIndexCount();
                extractionOutput.vertexCount = persistentBuffers.getCurrentVertexCount();
                extractionOutput.device = context.getDevice();
                try {
                    writeGPUExtractionToOBJ(context, extractionOutput, "/home/ge26mot/Projects/meshtrex/build/autonomous_extraction.obj");
                    std::cout << "Successfully wrote geometry to autonomous_extraction.obj" << std::endl;
                }
                catch (const std::exception& e) {
                    std::cout << "Error writing OBJ file: " << e.what() << std::endl;
                }
            }
            else {
                std::cout << "\nNo geometry extracted yet - autonomous system may need more frames" << std::endl;
            }

            std::cout << "\n=== Testing Isovalue Change ===" << std::endl;
            float newIsovalue = 100.0f;
            std::cout << "Changing isovalue from " << isovalue << " to " << newIsovalue << std::endl;
            autonomousSystem.setExtractionParameter(newIsovalue);

            for (uint32_t frameIndex = 10; frameIndex < 15; ++frameIndex) {
                std::cout << "\n--- Frame " << frameIndex << " (new isovalue) ---" << std::endl;
                VkCommandBuffer cmd2 = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
                autonomousSystem.executeAutonomousFrame(cmd2, frameIndex);
                endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd2);
                if (autonomousSystem.isExtractionComplete()) {
                    std::cout << "Re-extraction complete!" << std::endl;
                    break;
                }
                else {
                    float progress = autonomousSystem.getExtractionProgress();
                    std::cout << "Re-extraction progress: " << (int)(progress * 100) << "%" << std::endl;
                }
                vkDeviceWaitIdle(context.getDevice());
            }
            std::cout << "\n=== Autonomous Streaming System Test Complete ===" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error in autonomous streaming system: " << e.what() << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
