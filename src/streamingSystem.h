#pragma once

#include "vulkan_context.h"
#include "vulkan_utils.h"
#include "buffer.h"
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

struct PageCoord {
    uint32_t x, y, z;
    uint32_t mipLevel;
    
    bool operator==(const PageCoord& other) const {
        return x == other.x && y == other.y && z == other.z && mipLevel == other.mipLevel;
    }
    
    bool operator<(const PageCoord& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        if (z != other.z) return z < other.z;
        return mipLevel < other.mipLevel;
    }
};

struct PageCoordHash {
    std::size_t operator()(const PageCoord& coord) const {
        return std::hash<uint64_t>{}(
            ((uint64_t)coord.x << 48) | 
            ((uint64_t)coord.y << 32) | 
            ((uint64_t)coord.z << 16) | 
            (uint64_t)coord.mipLevel
        );
    }
};

struct PageEntry {
    uint32_t atlasX, atlasY, atlasZ;
    uint32_t atlasPageIndex = UINT32_MAX;
    uint32_t lastUsedFrame;
    bool isResident;
    bool isLoading;
};

struct StreamingParams {
    uint32_t pageSizeX = 64;  // Match sparse granularity
    uint32_t pageSizeY = 32;  // Match sparse granularity
    uint32_t pageSizeZ = 32;  // Match sparse granularity
    uint32_t atlasSizeX = 1024;
    uint32_t atlasSizeY = 1024;
    uint32_t atlasSizeZ = 1024;
    uint32_t maxResidentPages = 16384;
    uint32_t prefetchDistance = 64;
};

class VolumeStreamer {
public:
    VolumeStreamer(VulkanContext& context, const StreamingParams& params);
    ~VolumeStreamer();
    
    void loadVolume(const std::string& filename);
    void requestPage(const PageCoord& coord);
    void updateStreaming(uint32_t frameIndex);
    void processPendingBindings(); // Call from main thread
    void processPendingCopies(); // Call from main thread
    bool isPageResident(const PageCoord& coord) const;
    
    VkDescriptorSet getStreamingDescriptorSet() const { return streamingDescriptorSet; }
    VkImageView getVolumeAtlasView() const { return sparseVolumeAtlasView; }
    VkExtent3D getAtlasGranularity() const {
        if (sparseMemoryReqs.formatProperties.imageGranularity.width > 0) {
            return sparseMemoryReqs.formatProperties.imageGranularity;
        }
        // Return a default if not initialized, though it should be.
        return {64, 32, 32};
    }
    const Buffer& getPageTableBuffer() const { return pageTableBuffer; }
    VkSampler getVolumeSampler() const { return volumeSampler; }
    
private:
    VulkanContext& context;
    StreamingParams params;
    
    VkImage sparseVolumeAtlas;
    VkImageView sparseVolumeAtlasView;
    VkDeviceMemory* pageMemoryBlocks;
    uint32_t pageMemoryBlockCount;
    VkSparseImageMemoryRequirements sparseMemoryReqs;
    
    // Streaming thread resources
    VkCommandPool streamingCommandPool;
    VkQueue streamingQueue;
    bool hasOwnQueue;
    
    Buffer pageTableBuffer;
    Buffer indirectDispatchBuffer;
    
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout streamingDescriptorSetLayout;
    VkDescriptorSet streamingDescriptorSet;
    VkSampler volumeSampler;
    
    std::unordered_map<PageCoord, PageEntry, PageCoordHash> pageMap;
    std::unordered_set<PageCoord, PageCoordHash> requestedPages;
    std::queue<PageCoord> loadQueue;
    std::vector<uint32_t> freeAtlasPages;
    
    std::thread streamingThread;
    mutable std::mutex streamingMutex;
    std::condition_variable streamingCV;
    std::atomic<bool> shouldStop{false};
    
    // Queue synchronization
    std::mutex queueMutex;
    
    // Pending sparse bindings to be processed on main thread
    struct PendingBinding {
        std::vector<VkSparseImageMemoryBind> binds;
        std::vector<VkSparseImageMemoryBind> unbinds;
    };
    std::queue<PendingBinding> pendingBindings;
    std::mutex pendingBindingsMutex;
    
    // Pending copy operations to be processed on main thread
    struct PendingCopy {
        Buffer stagingBuffer;
        uint32_t atlasX, atlasY, atlasZ;
        uint32_t granularityX, granularityY, granularityZ;
        PageCoord pageCoord;
        uint32_t atlasPageIndex;
    };
    std::queue<PendingCopy> pendingCopies;
    std::mutex pendingCopiesMutex;
    
    std::vector<uint8_t> volumeData;
    uint32_t volumeWidth, volumeHeight, volumeDepth;
    
    void streamingThreadFunc();
    void loadPage(const PageCoord& coord);
    void evictPage(const PageCoord& coord);
    uint32_t allocateAtlasPage();
    void updatePageTable();
    
    void createSparseVolumeAtlas();
    void bindPageToMemory(const PageCoord& coord, uint32_t atlasPageIndex);
    void unbindPageFromMemory(const PageCoord& coord);
    VkDeviceMemory allocatePageMemory();
    void submitSparseBinding(const std::vector<VkSparseImageMemoryBind>& binds, const std::vector<VkSparseImageMemoryBind>& unbinds);
    
    void createStreamingDescriptors();
    void updateStreamingDescriptors();
};

struct GPUWorkQueue {
    Buffer commandBuffer;
    Buffer counterBuffer;
    uint32_t maxCommands;
    
    void initialize(VulkanContext& context, uint32_t maxCommands);
    void reset();
    void submit(VkCommandBuffer cmd);
};

class StreamingManager {
public:
    StreamingManager(VulkanContext& context);
    ~StreamingManager();
    
    void initialize(const StreamingParams& params);
    void loadVolume(const std::string& filename);
    void beginFrame(uint32_t frameIndex);
    void endFrame();
    
    VkDescriptorSet getStreamingDescriptors() const;
    
    VolumeStreamer& getStreamer() { return *streamer; }
    GPUWorkQueue& getWorkQueue() { return workQueue; }
    
private:
    VulkanContext& context;
    std::unique_ptr<VolumeStreamer> streamer;
    GPUWorkQueue workQueue;
    uint32_t currentFrame = 0;
};