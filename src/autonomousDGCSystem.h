#pragma once

#include "common.h"
#include "vulkan_context.h"
#include "buffer.h"
#include "deviceGeneratedCommands.h"
#include "persistentGeometryExtraction.h"
#include <glm/glm.hpp>

// Autonomous Device Generated Commands System
// Handles volume extraction based on control variables:
// PRIMARY: Surface extraction parameter changes (isovalue)
// SECONDARY: View parameters (only for large volumes)

enum ExtractionCommandType {
    EXTRACT_CMD_FULL = 0,           // Extract entire page
    EXTRACT_CMD_CONDITIONAL = 1,    // Extract if meets criteria (e.g., in frustum)
    EXTRACT_CMD_SKIP = 2,           // Skip extraction
    EXTRACT_CMD_EVICT = 3           // Evict page from memory
};

// Surface extraction state
struct ExtractionState {
    float currentParameter;         // Current extraction parameter (e.g., isovalue)
    float previousParameter;
    uint32_t framesSinceChange;
    uint32_t parameterChanged;
    uint32_t totalPagesToExtract;
    uint32_t pagesExtractedSoFar;
    uint32_t extractionComplete;
    uint32_t padding;
};

// Memory management state
struct MemoryState {
    uint32_t totalPages;
    uint32_t maxResidentPages;
    uint32_t currentResidentPages;
    uint32_t entireVolumeFits;      // 1 if entire volume fits in memory
    uint32_t memoryPressure;        // 0-100 scale
    uint32_t visiblePageCount;
    uint32_t padding[2];
};

// GPU-generated command for autonomous execution
struct AutonomousCommand {
    uint32_t commandType;           // ExtractionCommandType
    uint32_t pageCoordX;
    uint32_t pageCoordY;
    uint32_t pageCoordZ;
    uint32_t mipLevel;
    float priority;                 // Execution priority
    uint32_t requiresExtraction;    // 1 if geometry is invalid
    uint32_t padding;
};

// System configuration
struct AutonomousDGCConfig {
    // Volume parameters
    uint32_t volumeDimX;
    uint32_t volumeDimY;
    uint32_t volumeDimZ;
    uint32_t pageSizeX;
    uint32_t pageSizeY;
    uint32_t pageSizeZ;
    
    // Memory limits
    uint32_t maxResidentPages;
    uint32_t targetMemoryUsage;     // Target percentage
    
    // Execution parameters
    uint32_t maxCommandsPerFrame;
    uint32_t prioritizeVisible;     // Prioritize visible regions when memory limited
    
    // View parameters
    float viewDistanceThreshold;
    float evictionDistanceScale;
};

class AutonomousDGCSystem {
public:
    AutonomousDGCSystem(VulkanContext& context, DeviceGeneratedCommands* dgcSystem);
    ~AutonomousDGCSystem();

    void initialize(const AutonomousDGCConfig& config);
    
    // Primary control - extraction parameter (e.g., isovalue)
    void setExtractionParameter(float parameter);
    
    // Secondary control - view parameters (only matters for large volumes)
    void updateViewParameters(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& position);
    
    // Execute autonomous frame
    void executeAutonomousFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    
    // Query system state
    bool isExtractionComplete() const;
    float getExtractionProgress() const;
    bool entireVolumeFitsInMemory() const;
    uint32_t getResidentPageCount() const;
    
    // Update extraction progress (temporary until GPU-driven)
    void updateExtractionProgress(uint32_t pagesExtracted);

private:
    VulkanContext& context_;
    VkDevice device_;
    DeviceGeneratedCommands* dgcSystem_;
    
    // Configuration
    AutonomousDGCConfig config_;
    
    // GPU buffers for autonomous operation
    Buffer extractionStateBuffer_;      // Current extraction state
    Buffer memoryStateBuffer_;          // Memory usage and pressure
    Buffer pageValidityBuffer_;         // Per-page validity flags
    Buffer commandQueueBuffer_;         // GPU-generated commands
    Buffer pageResidencyBuffer_;        // Page residency status
    Buffer viewParametersBuffer_;       // View matrices and frustum
    Buffer pagePriorityBuffer_;         // Per-page execution priority
    
    // Autonomous processing pipelines
    VkPipeline stateAnalysisPipeline_;     // Analyze what needs extraction
    VkPipeline memoryAnalysisPipeline_;    // Determine memory pressure
    VkPipeline prioritizationPipeline_;    // Calculate page priorities
    VkPipeline commandGenerationPipeline_; // Generate execution commands
    
    // Descriptor sets and layouts
    VkDescriptorSetLayout descriptorLayout_;
    VkDescriptorSet descriptorSet_;
    VkPipelineLayout pipelineLayout_;
    VkDescriptorPool descriptorPool_;
    
    // CPU-side state
    float currentParameter_;
    bool volumeFitsInMemory_;
    uint32_t totalPageCount_;
    
    // Internal methods
    void createPipelines();
    void createDescriptorSets();
    void updateExtractionState(float parameter);
    void analyzeVolumeMemoryFit();
    void recordAutonomousCommands(VkCommandBuffer cmd, uint32_t frameIndex);
};

// GPU Autonomous Workflow:
//
// 1. Parameter change detection:
//    - If extraction parameter changed → mark all pages invalid
//    - Set high priority for all pages
//
// 2. Memory-based decision:
//    - If volume fits → generate EXTRACT_FULL for all invalid pages
//    - If volume too large → use view parameters to prioritize
//
// 3. Command generation:
//    - Generate extraction commands up to maxCommandsPerFrame
//    - Update page validity after successful extraction
//    - Continue until all pages are valid