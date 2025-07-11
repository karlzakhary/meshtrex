#include "common.h"
#include "streamingSystem.h"
#include "streamingShaderInterface.h"
#include "buffer.h"
#include "image.h"
#include "vulkan_utils.h"
#include "resources.h"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <iostream>

void createBuffer(VulkanContext& context, size_t size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    Buffer& buffer)
{
    ::createBuffer(buffer, context.getDevice(), context.getMemoryProperties(),
        size, usage, properties);
}

void updateBuffer(VulkanContext& context, const Buffer& buffer,
    const void* data, size_t size)
{
    if (buffer.data) {
        memcpy(buffer.data, data, size);
    }
    else {
        Buffer stagingBuffer;
        createBuffer(context, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer);
        memcpy(stagingBuffer.data, data, size);
        VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
        VkBufferCopy copyRegion = { 0, 0, size };
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, buffer.buffer, 1, &copyRegion);
        endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
        destroyBuffer(stagingBuffer, context.getDevice());
    }
}

void cleanupBuffer(VulkanContext& context, Buffer& buffer) {
    destroyBuffer(buffer, context.getDevice());
    buffer = {};
}

void createImage(VulkanContext& context, uint32_t width, uint32_t height, uint32_t depth,
    VkFormat format, VkImageUsageFlags usage, Image& image)
{
    ::createImage(image, context.getDevice(), context.getMemoryProperties(),
        VK_IMAGE_TYPE_3D, width, height, depth, 1, format, usage);
}

void cleanupImage(VulkanContext& context, Image& image) {
    destroyImage(image, context.getDevice());
    image = {};
}

void uploadImageData(VulkanContext& context, const Image& image, const void* data,
    uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
    uint32_t width, uint32_t height, uint32_t depth)
{
    size_t dataSize = width * height * depth;
    Buffer stagingBuffer;
    createBuffer(context, dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer);
    memcpy(stagingBuffer.data, data, dataSize);

    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    transitionImage(cmd, image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { (int32_t)offsetX, (int32_t)offsetY, (int32_t)offsetZ };
    region.imageExtent = { width, height, depth };
    vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    transitionImage(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);
    destroyBuffer(stagingBuffer, context.getDevice());
}

VkCommandBuffer beginSingleTimeCommands(VulkanContext& context) {
    return ::beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
}

void endSingleTimeCommands(VulkanContext& context, VkCommandBuffer commandBuffer) {
    ::endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), commandBuffer);
}

uint32_t getPageIndex(const PageCoord& coord, uint32_t volumeWidth, uint32_t volumeHeight,
    uint32_t volumeDepth, uint32_t pageSizeX, uint32_t pageSizeY, uint32_t pageSizeZ)
{
    uint32_t pagesX = (volumeWidth + pageSizeX - 1) / pageSizeX;
    uint32_t pagesY = (volumeHeight + pageSizeY - 1) / pageSizeY;
    return coord.z * pagesX * pagesY + coord.y * pagesX + coord.x;
}

VolumeStreamer::VolumeStreamer(VulkanContext& context, const StreamingParams& params)
    : context(context), params(params), sparseVolumeAtlas(VK_NULL_HANDLE),
    sparseVolumeAtlasView(VK_NULL_HANDLE), pageMemoryBlocks(nullptr), pageMemoryBlockCount(0),
    streamingCommandPool(VK_NULL_HANDLE), hasOwnQueue(false)
{
    VkCommandPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = context.getGraphicsQueueFamilyIndex();
    VK_CHECK(vkCreateCommandPool(context.getDevice(), &poolCreateInfo, nullptr, &streamingCommandPool));

    streamingQueue = context.getQueue();
    hasOwnQueue = false;
    createSparseVolumeAtlas();

    uint32_t granularityX = sparseMemoryReqs.formatProperties.imageGranularity.width;
    uint32_t granularityY = sparseMemoryReqs.formatProperties.imageGranularity.height;
    uint32_t granularityZ = sparseMemoryReqs.formatProperties.imageGranularity.depth;
    uint32_t blocksPerDimX = params.atlasSizeX / granularityX;
    uint32_t blocksPerDimY = params.atlasSizeY / granularityY;
    uint32_t blocksPerDimZ = params.atlasSizeZ / granularityZ;
    uint32_t actualAtlasCapacity = blocksPerDimX * blocksPerDimY * blocksPerDimZ;
    uint32_t safeBlocksX = (params.atlasSizeX - params.pageSizeX) / granularityX + 1;
    uint32_t safeBlocksY = (params.atlasSizeY - params.pageSizeY) / granularityY + 1;
    uint32_t safeBlocksZ = (params.atlasSizeZ - params.pageSizeZ) / granularityZ + 1;
    uint32_t safeAtlasCapacity = safeBlocksX * safeBlocksY * safeBlocksZ;
    uint32_t maxPages = std::min(params.maxResidentPages, std::min(actualAtlasCapacity, safeAtlasCapacity));

    std::cout << "Sparse atlas configuration:" << std::endl;
    std::cout << "  Atlas size: " << params.atlasSizeX << "x" << params.atlasSizeY << "x" << params.atlasSizeZ << std::endl;
    std::cout << "  Granularity: " << granularityX << "x" << granularityY << "x" << granularityZ << std::endl;
    std::cout << "  Blocks per dimension: " << blocksPerDimX << "x" << blocksPerDimY << "x" << blocksPerDimZ << std::endl;
    std::cout << "  Actual atlas capacity: " << actualAtlasCapacity << " pages" << std::endl;
    std::cout << "  Max resident pages: " << maxPages << std::endl;

    createStreamingDescriptors();
    uint32_t pageTableSize = params.maxResidentPages * sizeof(PageTableEntry);
    createBuffer(context, pageTableSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, pageTableBuffer);
    createBuffer(context, sizeof(IndirectDispatchCommand) * 1024,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indirectDispatchBuffer);

    for (uint32_t i = 0; i < maxPages; i++) {
        freeAtlasPages.push_back(i);
    }
    streamingThread = std::thread(&VolumeStreamer::streamingThreadFunc, this);
}

VolumeStreamer::~VolumeStreamer() {
    shouldStop = true;
    streamingCV.notify_all();
    if (streamingThread.joinable()) {
        streamingThread.join();
    }
    if (streamingCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context.getDevice(), streamingCommandPool, nullptr);
    }
    if (volumeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context.getDevice(), volumeSampler, nullptr);
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.getDevice(), descriptorPool, nullptr);
    }
    if (streamingDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.getDevice(), streamingDescriptorSetLayout, nullptr);
    }
    if (sparseVolumeAtlasView != VK_NULL_HANDLE) {
        vkDestroyImageView(context.getDevice(), sparseVolumeAtlasView, nullptr);
    }
    if (sparseVolumeAtlas != VK_NULL_HANDLE) {
        vkDestroyImage(context.getDevice(), sparseVolumeAtlas, nullptr);
    }
    if (pageMemoryBlocks) {
        for (uint32_t i = 0; i < pageMemoryBlockCount; i++) {
            if (pageMemoryBlocks[i] != VK_NULL_HANDLE) {
                vkFreeMemory(context.getDevice(), pageMemoryBlocks[i], nullptr);
            }
        }
        delete[] pageMemoryBlocks;
    }
    cleanupBuffer(context, pageTableBuffer);
    cleanupBuffer(context, indirectDispatchBuffer);
}

void VolumeStreamer::loadVolume(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open volume file: " + filename);
    }

    std::string basename = filename.substr(filename.find_last_of("/\\") + 1);
    basename = basename.substr(0, basename.find_last_of('.'));
    size_t firstUnderscore = basename.find('_');
    size_t lastUnderscore = basename.find_last_of('_');

    if (firstUnderscore != std::string::npos && lastUnderscore != std::string::npos) {
        std::string dimensions = basename.substr(firstUnderscore + 1, lastUnderscore - firstUnderscore - 1);
        size_t firstX = dimensions.find('x');
        size_t lastX = dimensions.find_last_of('x');
        if (firstX != std::string::npos && lastX != std::string::npos) {
            volumeWidth = std::stoul(dimensions.substr(0, firstX));
            volumeHeight = std::stoul(dimensions.substr(firstX + 1, lastX - firstX - 1));
            volumeDepth = std::stoul(dimensions.substr(lastX + 1));
        }
    }

    size_t volumeSize = volumeWidth * volumeHeight * volumeDepth;
    volumeData.resize(volumeSize);
    file.read(reinterpret_cast<char*>(volumeData.data()), volumeSize);
    file.close();

    std::cout << "Loaded volume: " << volumeWidth << "x" << volumeHeight << "x" << volumeDepth << std::endl;
    uint32_t nonZeroCount = 0;
    uint8_t minValue = 255, maxValue = 0;
    uint32_t histogram[256] = { 0 };
    for (uint32_t i = 0; i < volumeSize; i++) {
        if (volumeData[i] > 0) nonZeroCount++;
        minValue = std::min(minValue, volumeData[i]);
        maxValue = std::max(maxValue, volumeData[i]);
        histogram[volumeData[i]]++;
    }
    std::cout << "Volume data stats: non-zero values: " << nonZeroCount << "/" << volumeSize
        << ", min: " << (int)minValue << ", max: " << (int)maxValue << std::endl;
    std::cout << "Value distribution (showing values with >1% of voxels):" << std::endl;
    uint32_t threshold = volumeSize / 100;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > threshold) {
            std::cout << "  Value " << i << ": " << histogram[i] << " voxels ("
                << (histogram[i] * 100.0 / volumeSize) << "%)" << std::endl;
        }
    }
}

void VolumeStreamer::requestPage(const PageCoord& coord) {
    std::lock_guard<std::mutex> lock(streamingMutex);
    auto it = pageMap.find(coord);
    if (it != pageMap.end() && it->second.isResident) {
        return;
    }
    if (requestedPages.find(coord) == requestedPages.end()) {
        requestedPages.insert(coord);
        loadQueue.push(coord);
        streamingCV.notify_one();
    }
}

void VolumeStreamer::updateStreaming(uint32_t frameIndex) {
    std::lock_guard<std::mutex> lock(streamingMutex);
    std::vector<PageCoord> pagesToEvict;
    for (auto& [coord, entry] : pageMap) {
        if (entry.isResident && entry.atlasPageIndex != UINT32_MAX &&
            (frameIndex - entry.lastUsedFrame) > 60) {
            pagesToEvict.push_back(coord);
        }
    }
    for (const auto& coord : pagesToEvict) {
        evictPage(coord);
    }
    updatePageTable();
}

bool VolumeStreamer::isPageResident(const PageCoord& coord) const {
    std::lock_guard<std::mutex> lock(streamingMutex);
    auto it = pageMap.find(coord);
    if (it != pageMap.end() && it->second.isResident) {
        const_cast<PageEntry&>(it->second).lastUsedFrame = 0;
        return true;
    }
    return false;
}

void VolumeStreamer::processPendingBindings() {
    std::queue<PendingBinding> bindingsToProcess;
    {
        std::lock_guard<std::mutex> lock(pendingBindingsMutex);
        std::swap(bindingsToProcess, pendingBindings);
    }
    while (!bindingsToProcess.empty()) {
        PendingBinding& pending = bindingsToProcess.front();
        if (!pending.binds.empty() || !pending.unbinds.empty()) {
            VkSparseImageMemoryBindInfo imageBindInfo = {};
            imageBindInfo.image = sparseVolumeAtlas;
            imageBindInfo.bindCount = static_cast<uint32_t>(pending.binds.size() + pending.unbinds.size());
            std::vector<VkSparseImageMemoryBind> allBinds;
            allBinds.insert(allBinds.end(), pending.binds.begin(), pending.binds.end());
            allBinds.insert(allBinds.end(), pending.unbinds.begin(), pending.unbinds.end());
            imageBindInfo.pBinds = allBinds.data();
            VkBindSparseInfo bindInfo = {};
            bindInfo.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
            bindInfo.imageBindCount = 1;
            bindInfo.pImageBinds = &imageBindInfo;
            VkQueue sparseQueue = context.getQueue();
            VK_CHECK(vkQueueBindSparse(sparseQueue, 1, &bindInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(sparseQueue));
        }
        bindingsToProcess.pop();
    }
}

void VolumeStreamer::processPendingCopies() {
    std::queue<PendingCopy> copiesToProcess;
    {
        std::lock_guard<std::mutex> lock(pendingCopiesMutex);
        std::swap(copiesToProcess, pendingCopies);
    }
    while (!copiesToProcess.empty()) {
        PendingCopy& pending = copiesToProcess.front();
        VkCommandBuffer copyCmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = { static_cast<int32_t>(pending.atlasX), static_cast<int32_t>(pending.atlasY), static_cast<int32_t>(pending.atlasZ) };
        uint32_t copyWidth = std::min(pending.granularityX, params.atlasSizeX - pending.atlasX);
        uint32_t copyHeight = std::min(pending.granularityY, params.atlasSizeY - pending.atlasY);
        uint32_t copyDepth = std::min(pending.granularityZ, params.atlasSizeZ - pending.atlasZ);
        copyRegion.imageExtent = { copyWidth, copyHeight, copyDepth };
        vkCmdCopyBufferToImage(copyCmd, pending.stagingBuffer.buffer, sparseVolumeAtlas, VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
        VkImageMemoryBarrier2 copyBarrier = {};
        copyBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        copyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        copyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        copyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        copyBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        copyBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        copyBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarrier.image = sparseVolumeAtlas;
        copyBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyBarrier.subresourceRange.baseMipLevel = 0;
        copyBarrier.subresourceRange.levelCount = 1;
        copyBarrier.subresourceRange.baseArrayLayer = 0;
        copyBarrier.subresourceRange.layerCount = 1;
        VkDependencyInfo depInfo = {};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &copyBarrier;
        vkCmdPipelineBarrier2(copyCmd, &depInfo);
        endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), copyCmd);
        VK_CHECK(vkQueueWaitIdle(context.getQueue()));
        {
            std::lock_guard<std::mutex> lock(streamingMutex);
            PageEntry entry;
            entry.atlasX = pending.atlasX;
            entry.atlasY = pending.atlasY;
            entry.atlasZ = pending.atlasZ;
            entry.atlasPageIndex = pending.atlasPageIndex;
            entry.lastUsedFrame = 0;
            entry.isResident = true;
            entry.isLoading = false;
            pageMap[pending.pageCoord] = entry;
            requestedPages.erase(pending.pageCoord);
        }
        destroyBuffer(pending.stagingBuffer, context.getDevice());
        copiesToProcess.pop();
    }
}

void VolumeStreamer::streamingThreadFunc() {
    while (!shouldStop) {
        std::unique_lock<std::mutex> lock(streamingMutex);
        streamingCV.wait(lock, [this] { return !loadQueue.empty() || shouldStop; });
        if (shouldStop) break;
        if (!loadQueue.empty()) {
            PageCoord coord = loadQueue.front();
            loadQueue.pop();
            lock.unlock();
            loadPage(coord);
        }
    }
}

void VolumeStreamer::loadPage(const PageCoord& coord) {
    if (freeAtlasPages.empty()) {
        return;
    }
    uint32_t atlasPageIndex = allocateAtlasPage();
    if (atlasPageIndex == UINT32_MAX) {
        return;
    }
    VkDeviceMemory pageMemory = allocatePageMemory();
    if (pageMemory == VK_NULL_HANDLE) {
        freeAtlasPages.push_back(atlasPageIndex);
        return;
    }
    uint32_t granularityX = sparseMemoryReqs.formatProperties.imageGranularity.width;
    uint32_t granularityY = sparseMemoryReqs.formatProperties.imageGranularity.height;
    uint32_t granularityZ = sparseMemoryReqs.formatProperties.imageGranularity.depth;
    std::vector<uint8_t> pageData(granularityX* granularityY* granularityZ, 0);
    if (params.pageSizeX != granularityX || params.pageSizeY != granularityY || params.pageSizeZ != granularityZ) {
        std::cout << "WARNING: Page size (" << params.pageSizeX << "x" << params.pageSizeY << "x" << params.pageSizeZ
            << ") doesn't match granularity (" << granularityX << "x" << granularityY << "x" << granularityZ << ")" << std::endl;
    }
    uint32_t copyWidth = params.pageSizeX;
    uint32_t copyHeight = params.pageSizeY;
    uint32_t copyDepth = params.pageSizeZ;
    uint8_t pageMinValue = 255;
    uint8_t pageMaxValue = 0;
    for (uint32_t z = 0; z < copyDepth; z++) {
        for (uint32_t y = 0; y < copyHeight; y++) {
            for (uint32_t x = 0; x < copyWidth; x++) {
                uint32_t worldX = coord.x * params.pageSizeX + x;
                uint32_t worldY = coord.y * params.pageSizeY + y;
                uint32_t worldZ = coord.z * params.pageSizeZ + z;
                if (worldX < volumeWidth && worldY < volumeHeight && worldZ < volumeDepth) {
                    uint32_t volumeIndex = worldZ * volumeWidth * volumeHeight + worldY * volumeWidth + worldX;
                    uint32_t pageIndex = z * granularityX * granularityY + y * granularityX + x;
                    pageData[pageIndex] = volumeData[volumeIndex];
                    pageMinValue = std::min(pageMinValue, volumeData[volumeIndex]);
                    pageMaxValue = std::max(pageMaxValue, volumeData[volumeIndex]);
                }
            }
        }
    }
    std::cout << "Page (" << coord.x << "," << coord.y << "," << coord.z
        << ") loaded with min=" << (int)pageMinValue << ", max=" << (int)pageMaxValue << std::endl;
    Buffer stagingBuffer;
    createBuffer(stagingBuffer, context.getDevice(), context.getMemoryProperties(),
        pageData.size(),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (stagingBuffer.data) {
        memcpy(stagingBuffer.data, pageData.data(), pageData.size());
    }
    else {
        void* mappedMemory;
        vkMapMemory(context.getDevice(), stagingBuffer.memory, 0, pageData.size(), 0, &mappedMemory);
        memcpy(mappedMemory, pageData.data(), pageData.size());
        vkUnmapMemory(context.getDevice(), stagingBuffer.memory);
    }
    pageMemoryBlocks[atlasPageIndex] = pageMemory;
    bindPageToMemory(coord, atlasPageIndex);
    uint32_t blocksPerDimX = params.atlasSizeX / granularityX;
    uint32_t blocksPerDimY = params.atlasSizeY / granularityY;
    uint32_t blockX = atlasPageIndex % blocksPerDimX;
    uint32_t blockY = (atlasPageIndex / blocksPerDimX) % blocksPerDimY;
    uint32_t blockZ = atlasPageIndex / (blocksPerDimX * blocksPerDimY);
    uint32_t atlasX = blockX * granularityX;
    uint32_t atlasY = blockY * granularityY;
    uint32_t atlasZ = blockZ * granularityZ;
    {
        std::lock_guard<std::mutex> lock(pendingCopiesMutex);
        pendingCopies.push({
            stagingBuffer,
            atlasX, atlasY, atlasZ,
            granularityX, granularityY, granularityZ,
            coord,
            atlasPageIndex
            });
    }
}

void VolumeStreamer::evictPage(const PageCoord& coord) {
    auto it = pageMap.find(coord);
    if (it != pageMap.end() && it->second.atlasPageIndex != UINT32_MAX) {
        uint32_t granularityX = sparseMemoryReqs.formatProperties.imageGranularity.width;
        uint32_t granularityY = sparseMemoryReqs.formatProperties.imageGranularity.height;
        uint32_t granularityZ = sparseMemoryReqs.formatProperties.imageGranularity.depth;
        uint32_t blocksPerDimX = params.atlasSizeX / granularityX;
        uint32_t blocksPerDimY = params.atlasSizeY / granularityY;
        uint32_t blocksPerDimZ = params.atlasSizeZ / granularityZ;
        uint32_t actualCapacity = blocksPerDimX * blocksPerDimY * blocksPerDimZ;
        if (it->second.atlasPageIndex < actualCapacity) {
            freeAtlasPages.push_back(it->second.atlasPageIndex);
        }
        else {
            std::cerr << "WARNING: Attempted to return invalid atlas page index "
                << it->second.atlasPageIndex << " to free list (capacity: "
                << actualCapacity << ")" << std::endl;
        }
        pageMap.erase(it);
    }
}

uint32_t VolumeStreamer::allocateAtlasPage() {
    if (freeAtlasPages.empty()) {
        std::cerr << "ERROR: No free atlas pages available!" << std::endl;
        return UINT32_MAX;
    }
    uint32_t pageIndex = freeAtlasPages.back();
    freeAtlasPages.pop_back();
    uint32_t granularityX = sparseMemoryReqs.formatProperties.imageGranularity.width;
    uint32_t granularityY = sparseMemoryReqs.formatProperties.imageGranularity.height;
    uint32_t granularityZ = sparseMemoryReqs.formatProperties.imageGranularity.depth;
    uint32_t blocksPerDimX = params.atlasSizeX / granularityX;
    uint32_t blocksPerDimY = params.atlasSizeY / granularityY;
    uint32_t blocksPerDimZ = params.atlasSizeZ / granularityZ;
    uint32_t actualCapacity = blocksPerDimX * blocksPerDimY * blocksPerDimZ;
    if (pageIndex >= actualCapacity) {
        std::cerr << "ERROR: Allocated atlas page index " << pageIndex << " exceeds capacity " << actualCapacity << std::endl;
        return UINT32_MAX;
    }
    return pageIndex;
}

void VolumeStreamer::updatePageTable() {
    std::vector<PageTableEntry> pageTableData(params.maxResidentPages);
    for (const auto& [coord, entry] : pageMap) {
        if (entry.isResident) {
            uint32_t pageIndex = getPageIndex(coord, volumeWidth, volumeHeight, volumeDepth,
                params.pageSizeX, params.pageSizeY, params.pageSizeZ);
            if (pageIndex < params.maxResidentPages) {
                pageTableData[pageIndex].atlasX = entry.atlasX;
                pageTableData[pageIndex].atlasY = entry.atlasY;
                pageTableData[pageIndex].atlasZ = entry.atlasZ;
                pageTableData[pageIndex].isResident = 1;
            }
        }
    }
    updateBuffer(context, pageTableBuffer, pageTableData.data(),
        pageTableData.size() * sizeof(PageTableEntry));
    updateStreamingDescriptors();
}

void GPUWorkQueue::initialize(VulkanContext& context, uint32_t maxCommands) {
    this->maxCommands = maxCommands;
    uint32_t commandBufferSize = sizeof(WorkQueueHeader) +
        maxCommands * sizeof(IndirectDispatchCommand);
    createBuffer(context, commandBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, commandBuffer);
    createBuffer(context, sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, counterBuffer);
}

void GPUWorkQueue::reset() {
    uint32_t zero = 0;
}

StreamingManager::StreamingManager(VulkanContext& context) : context(context) {}
StreamingManager::~StreamingManager() = default;

void StreamingManager::initialize(const StreamingParams& params) {
    streamer = std::make_unique<VolumeStreamer>(context, params);
    workQueue.initialize(context, 1024);
}

void StreamingManager::loadVolume(const std::string& filename) {
    streamer->loadVolume(filename);
}

void StreamingManager::beginFrame(uint32_t frameIndex) {
    currentFrame = frameIndex;
    workQueue.reset();
    streamer->updateStreaming(frameIndex);
}

void StreamingManager::endFrame() {}

VkDescriptorSet StreamingManager::getStreamingDescriptors() const {
    return streamer->getStreamingDescriptorSet();
}

void VolumeStreamer::createSparseVolumeAtlas() {
    VkFormatProperties formatProps;
    vkGetPhysicalDeviceFormatProperties(context.getPhysicalDevice(), VK_FORMAT_R8_UINT, &formatProps);
    std::cout << "Format properties for VK_FORMAT_R8_UINT:" << std::endl;
    std::cout << "  Optimal tiling features: " << formatProps.optimalTilingFeatures << std::endl;
    std::cout << "  Supports storage image: " << ((formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0) << std::endl;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.format = VK_FORMAT_R8_UINT;
    imageInfo.extent = { params.atlasSizeX, params.atlasSizeY, params.atlasSizeZ };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(context.getDevice(), &imageInfo, nullptr, &sparseVolumeAtlas));

    uint32_t sparseReqCount = 0;
    vkGetImageSparseMemoryRequirements(context.getDevice(), sparseVolumeAtlas, &sparseReqCount, nullptr);
    std::vector<VkSparseImageMemoryRequirements> sparseReqs(sparseReqCount);
    vkGetImageSparseMemoryRequirements(context.getDevice(), sparseVolumeAtlas, &sparseReqCount, sparseReqs.data());
    if (!sparseReqs.empty()) {
        sparseMemoryReqs = sparseReqs[0];
        std::cout << "Sparse image requirements:" << std::endl;
        std::cout << "  Format properties supported: " << (sparseMemoryReqs.formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) << std::endl;
        std::cout << "  Image granularity: " << sparseMemoryReqs.formatProperties.imageGranularity.width
            << "x" << sparseMemoryReqs.formatProperties.imageGranularity.height
            << "x" << sparseMemoryReqs.formatProperties.imageGranularity.depth << std::endl;
        std::cout << "  Flags: " << sparseMemoryReqs.formatProperties.flags << std::endl;
    }

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = sparseVolumeAtlas;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = VK_FORMAT_R8_UINT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(context.getDevice(), &viewInfo, nullptr, &sparseVolumeAtlasView));

    VkCommandBuffer cmd = beginSingleTimeCommands(context.getDevice(), context.getCommandPool());
    VkImageMemoryBarrier2 barrier2 = {};
    barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier2.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barrier2.srcAccessMask = 0;
    barrier2.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    barrier2.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barrier2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier2.image = sparseVolumeAtlas;
    barrier2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier2.subresourceRange.baseMipLevel = 0;
    barrier2.subresourceRange.levelCount = 1;
    barrier2.subresourceRange.baseArrayLayer = 0;
    barrier2.subresourceRange.layerCount = 1;
    VkDependencyInfo depInfo = {};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &barrier2;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    endSingleTimeCommands(context.getDevice(), context.getCommandPool(), context.getQueue(), cmd);

    pageMemoryBlockCount = params.maxResidentPages;
    pageMemoryBlocks = new VkDeviceMemory[pageMemoryBlockCount];
    for (uint32_t i = 0; i < pageMemoryBlockCount; i++) {
        pageMemoryBlocks[i] = VK_NULL_HANDLE;
    }
}

VkDeviceMemory VolumeStreamer::allocatePageMemory() {
    VkDeviceSize allocationSize = sparseMemoryReqs.formatProperties.imageGranularity.width *
        sparseMemoryReqs.formatProperties.imageGranularity.height *
        sparseMemoryReqs.formatProperties.imageGranularity.depth;
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(context.getDevice(), sparseVolumeAtlas, &memReqs);
    allocationSize = (allocationSize + memReqs.alignment - 1) & ~(memReqs.alignment - 1);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = allocationSize;
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(context.getPhysicalDevice(), &memProps);
    bool found = false;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            allocInfo.memoryTypeIndex = i;
            found = true;
            break;
        }
    }
    if (!found) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if (memReqs.memoryTypeBits & (1 << i)) {
                allocInfo.memoryTypeIndex = i;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        std::cout << "Failed to find suitable memory type for sparse binding" << std::endl;
        return VK_NULL_HANDLE;
    }
    VkDeviceMemory memory;
    if (vkAllocateMemory(context.getDevice(), &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return memory;
}

void VolumeStreamer::bindPageToMemory(const PageCoord& coord, uint32_t atlasPageIndex) {
    VkSparseImageMemoryBind bind = {};
    bind.subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bind.subresource.mipLevel = coord.mipLevel;
    bind.subresource.arrayLayer = 0;
    uint32_t granularityX = sparseMemoryReqs.formatProperties.imageGranularity.width;
    uint32_t granularityY = sparseMemoryReqs.formatProperties.imageGranularity.height;
    uint32_t granularityZ = sparseMemoryReqs.formatProperties.imageGranularity.depth;
    uint32_t blocksPerDimX = params.atlasSizeX / granularityX;
    uint32_t blocksPerDimY = params.atlasSizeY / granularityY;
    uint32_t blocksPerDimZ = params.atlasSizeZ / granularityZ;
    uint32_t blockX = atlasPageIndex % blocksPerDimX;
    uint32_t blockY = (atlasPageIndex / blocksPerDimX) % blocksPerDimY;
    uint32_t blockZ = atlasPageIndex / (blocksPerDimX * blocksPerDimY);
    if (blockZ >= blocksPerDimZ) {
        std::cerr << "ERROR: Atlas page index " << atlasPageIndex << " exceeds atlas capacity!" << std::endl;
        std::cerr << "  Block Z: " << blockZ << " >= " << blocksPerDimZ << std::endl;
        return;
    }
    bind.offset = {
        static_cast<int32_t>(blockX * granularityX),
        static_cast<int32_t>(blockY * granularityY),
        static_cast<int32_t>(blockZ * granularityZ)
    };
    bind.extent = { granularityX, granularityY, granularityZ };
    bind.memory = pageMemoryBlocks[atlasPageIndex];
    bind.memoryOffset = 0;
    bind.flags = 0;
    std::vector<VkSparseImageMemoryBind> binds = { bind };
    submitSparseBinding(binds, {});
}

void VolumeStreamer::unbindPageFromMemory(const PageCoord& coord) {
}

void VolumeStreamer::submitSparseBinding(const std::vector<VkSparseImageMemoryBind>& binds, const std::vector<VkSparseImageMemoryBind>& unbinds) {
    if (binds.empty() && unbinds.empty()) return;
    {
        std::lock_guard<std::mutex> lock(pendingBindingsMutex);
        pendingBindings.push({ binds, unbinds });
    }
}

void VolumeStreamer::createStreamingDescriptors() {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(context.getDevice(), &samplerInfo, nullptr, &volumeSampler));

    std::vector<VkDescriptorSetLayoutBinding> bindings(2);
    bindings[0] = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT
    };
    bindings[1] = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(context.getDevice(), &layoutInfo, nullptr, &streamingDescriptorSetLayout));

    std::vector<VkDescriptorPoolSize> poolSizes(2);
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(context.getDevice(), &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &streamingDescriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(context.getDevice(), &allocInfo, &streamingDescriptorSet));
}

void VolumeStreamer::updateStreamingDescriptors() {
    std::vector<VkWriteDescriptorSet> descriptorWrites(2);
    VkDescriptorBufferInfo pageTableBufferInfo = {};
    pageTableBufferInfo.buffer = pageTableBuffer.buffer;
    pageTableBufferInfo.offset = 0;
    pageTableBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0] = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = streamingDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &pageTableBufferInfo;

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView = sparseVolumeAtlasView;
    imageInfo.sampler = VK_NULL_HANDLE;
    descriptorWrites[1] = {};
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = streamingDescriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(context.getDevice(), static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(), 0, nullptr);
}
