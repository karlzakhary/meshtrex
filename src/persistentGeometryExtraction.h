#pragma once

#include "common.h"
#include "buffer.h"
#include "vulkan_context.h"
#include "streamingSystem.h"
#include "streamingShaderInterface.h"
#include "filteringOutput.h"

// Result of extracting geometry to persistent global buffers
struct PersistentExtractionResult {
    // Where the geometry was placed in global buffers
    uint32_t globalVertexOffset;
    uint32_t globalIndexOffset;
    uint32_t globalMeshletOffset;
    
    // How much geometry was generated
    uint32_t verticesGenerated;
    uint32_t indicesGenerated;
    uint32_t meshletsGenerated;
    
    // Success/failure info
    bool success;
    bool vertexBufferFull;
    bool indexBufferFull;
    bool meshletBufferFull;
    
    // Source info
    PageCoord sourcePageCoord;
    float isoValue;
    uint32_t frameExtracted;
};

// Global persistent buffers that accumulate geometry across frames
class PersistentGeometryBuffers {
public:
    PersistentGeometryBuffers(VulkanContext& context);
    ~PersistentGeometryBuffers();

    // Initialize the global buffers
    void initialize(uint32_t maxVertices = 50000000,    // 50M vertices
                   uint32_t maxIndices = 150000000,     // 150M indices  
                   uint32_t maxMeshlets = 500000);      // 500K meshlets

    // Frame management
    void beginFrame(uint32_t frameIndex);
    void endFrame();

    // Core extraction function - extracts PMB geometry to global buffers
    PersistentExtractionResult extractPageToGlobalBuffers(
        const PageCoord& pageCoord,
        const StreamingExtractionConstants& constants,
        const FilteringOutput& filteringOutput,
        VkImageView volumeAtlasView,
        const Buffer& pageTableBuffer,
        uint32_t volumeSizeX = 256,
        uint32_t volumeSizeY = 256,
        uint32_t volumeSizeZ = 256
    );
    
    // Get current allocation state
    uint32_t getCurrentVertexCount() const { return currentVertexOffset; }
    uint32_t getCurrentIndexCount() const { return currentIndexOffset; }
    uint32_t getCurrentMeshletCount() const { return currentMeshletOffset; }
    
    // Manual reset (for isovalue changes, etc.)
    void resetAllocations();
    void reset() { resetAllocations(); }  // Alias for IntegratedAutonomousSystem
    
    // Update counters after extraction
    void updateCounters(uint32_t verticesAdded, uint32_t indicesAdded, uint32_t meshletsAdded);
    
    // Get buffers for descriptor sets
    const Buffer& getGlobalVertexBuffer() const { return globalVertexBuffer; }
    const Buffer& getGlobalIndexBuffer() const { return globalIndexBuffer; }
    const Buffer& getGlobalMeshletBuffer() const { return globalMeshletBuffer; }
    const Buffer& getVertexCounterBuffer() const { return globalCountersBuffer; }
    
    // Export accumulated geometry to OBJ file
    bool exportToOBJ(const std::string& filePath) const;

private:
    VulkanContext& context_;
    VkDevice device_;
    
    // Global persistent buffers
    Buffer globalVertexBuffer;
    Buffer globalIndexBuffer;
    Buffer globalMeshletBuffer;
    Buffer globalCountersBuffer;    // Tracks current allocation offsets
    
    // Separate atomic counter buffers for shader use
    Buffer vertexCounterBuffer;   // Single uint for atomic operations
    Buffer indexCounterBuffer;    // Single uint for atomic operations
    Buffer meshletCounterBuffer;  // Single uint for atomic operations
    
    // Buffer configuration
    uint32_t maxVertices;
    uint32_t maxIndices;
    uint32_t maxMeshlets;
    
    // Current allocation state (CPU side)
    uint32_t currentVertexOffset;
    uint32_t currentIndexOffset;
    uint32_t currentMeshletOffset;
    uint32_t currentFrame;
    
    // Track last atomic counter values for per-page counting
    uint32_t lastVertexCounter = 0;
    uint32_t lastIndexCounter = 0;
    uint32_t lastMeshletCounter = 0;
    
    // Internal methods
    bool allocateSpace(uint32_t verticesNeeded, uint32_t indicesNeeded, uint32_t meshletsNeeded,
                      uint32_t& vertexOffset, uint32_t& indexOffset, uint32_t& meshletOffset);
    
    void updateGlobalCounters();
    void executePMBExtractionToGlobalBuffers(const PageCoord& pageCoord,
                                            const StreamingExtractionConstants& constants,
                                            const FilteringOutput& filteringOutput,
                                            VkImageView volumeAtlasView,
                                            const Buffer& pageTableBuffer,
                                            uint32_t globalVertexOffset,
                                            uint32_t globalIndexOffset,
                                            uint32_t globalMeshletOffset,
                                            uint32_t volumeSizeX,
                                            uint32_t volumeSizeY,
                                            uint32_t volumeSizeZ);
    
    Buffer createMCTriTableBuffer();
    Buffer createMCEdgeTableBuffer();
    
    // Forward declaration
    struct ActualGeometryCounts {
        uint32_t verticesGenerated;
        uint32_t indicesGenerated;
        uint32_t meshletsGenerated;
    };
    
    ActualGeometryCounts readbackActualGeometryCounts(const FilteringOutput& filteringOutput);
};