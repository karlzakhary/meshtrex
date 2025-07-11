#pragma once

#include "common.h"
#include "buffer.h"
#include "vulkan_context.h"
#include "streamingSystem.h"
#include <atomic>
#include <mutex>
#include <unordered_map>

// Global allocation result for a page
struct GlobalAllocationResult {
    uint32_t vertexOffset = UINT32_MAX;
    uint32_t indexOffset = UINT32_MAX;
    uint32_t meshletOffset = UINT32_MAX;
    uint32_t allocatedVertices = 0;
    uint32_t allocatedIndices = 0;
    uint32_t allocatedMeshlets = 0;
    bool success = false;
    bool vertexOverflow = false;
    bool indexOverflow = false;
    bool meshletOverflow = false;
};

// Global counters structure for GPU
struct GlobalAllocationCounters {
    uint32_t globalVertexCount;
    uint32_t globalIndexCount;
    uint32_t globalMeshletCount;
    
    uint32_t maxVertices;
    uint32_t maxIndices;
    uint32_t maxMeshlets;
    
    uint32_t overflowFlags;
    uint32_t frameIndex;
    uint32_t totalPagesProcessed;
    uint32_t padding;
};

// Overflow flags
constexpr uint32_t VERTEX_OVERFLOW_FLAG = 0x1;
constexpr uint32_t INDEX_OVERFLOW_FLAG = 0x2;
constexpr uint32_t MESHLET_OVERFLOW_FLAG = 0x4;

// Page allocation tracking
struct PageAllocation {
    PageCoord pageCoord;
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t meshletOffset;
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t meshletCount;
    uint32_t frameAllocated;
};

class GlobalExtractionBuffers {
public:
    GlobalExtractionBuffers(VulkanContext& context);
    ~GlobalExtractionBuffers();

    // Initialize with buffer sizes
    void initialize(uint32_t maxVertices = 100000000,    // 100M vertices
                   uint32_t maxIndices = 300000000,      // 300M indices  
                   uint32_t maxMeshlets = 1000000);      // 1M meshlets

    // Frame management
    void beginFrame(uint32_t frameIndex);
    void endFrame();

    // Allocation for a specific page
    GlobalAllocationResult allocateForPage(const PageCoord& pageCoord,
                                         uint32_t estimatedVertices,
                                         uint32_t estimatedIndices,
                                         uint32_t estimatedMeshlets);

    // Get global buffers for shader binding
    VkBuffer getGlobalVertexBuffer() const { return globalVertexBuffer.buffer; }
    VkBuffer getGlobalIndexBuffer() const { return globalIndexBuffer.buffer; }
    VkBuffer getGlobalMeshletBuffer() const { return globalMeshletBuffer.buffer; }
    VkBuffer getGlobalCountersBuffer() const { return globalCountersBuffer.buffer; }

    // Check for overflows after GPU execution
    void checkOverflows();
    
    // Get allocation info for a page (for rendering)
    const PageAllocation* getPageAllocation(const PageCoord& pageCoord) const;

    // Statistics
    struct Statistics {
        uint32_t currentVerticesUsed;
        uint32_t currentIndicesUsed;
        uint32_t currentMeshletsUsed;
        uint32_t maxVerticesUsed;
        uint32_t maxIndicesUsed;
        uint32_t maxMeshletsUsed;
        uint32_t totalPagesAllocated;
        uint32_t totalOverflows;
    };
    Statistics getStatistics() const;

private:
    VulkanContext& context_;
    VkDevice device_;

    // Global buffers
    Buffer globalVertexBuffer;
    Buffer globalIndexBuffer;
    Buffer globalMeshletBuffer;
    Buffer globalCountersBuffer;

    // CPU tracking
    std::atomic<uint32_t> cpuVertexOffset{0};
    std::atomic<uint32_t> cpuIndexOffset{0};
    std::atomic<uint32_t> cpuMeshletOffset{0};

    // Frame tracking
    uint32_t currentFrame = 0;
    std::unordered_map<PageCoord, PageAllocation, PageCoordHash> pageAllocations;
    mutable std::mutex allocationMutex;

    // Configuration
    uint32_t maxVertices;
    uint32_t maxIndices;
    uint32_t maxMeshlets;

    // Statistics
    mutable Statistics stats = {};

    void resetCounters();
    void updateCountersBuffer();
    void readbackCounters();
};

// Global buffer overflow detection shader interface
struct GlobalBufferConstants {
    uint32_t globalVertexOffset;
    uint32_t globalIndexOffset;
    uint32_t globalMeshletOffset;
    uint32_t maxVertices;
    uint32_t maxIndices;
    uint32_t maxMeshlets;
    uint32_t pageVertexEstimate;
    uint32_t pageIndexEstimate;
    uint32_t pageMeshletEstimate;
    uint32_t padding[3];
};