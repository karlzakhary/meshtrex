#pragma once

#include "gpuProfiler.h"
#include "vulkan_context.h"
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

class OcclusionDebugger {
public:
    // Debug metrics collected from GPU
    struct OcclusionMetrics {
        // Block counts at each stage
        uint32_t totalBlocks = 0;
        uint32_t minMaxPassed = 0;
        uint32_t frustumPassed = 0;
        uint32_t hiZTested = 0;
        uint32_t hiZPassed = 0;
        uint32_t finalVisible = 0;
        
        // Atomic operation stats
        uint32_t atomicOperations = 0;
        uint32_t subgroupAggregations = 0;
        uint32_t uniqueWordsAccessed = 0;
        
        // Efficiency metrics
        float minMaxCullRate = 0.0f;
        float frustumCullRate = 0.0f;
        float hiZCullRate = 0.0f;
        float overallCullRate = 0.0f;
        float atomicReductionRate = 0.0f;
        
        // Timing from GPUProfiler
        float hiZGenerationMs = 0.0f;
        float occlusionTestMs = 0.0f;
        float pvsBuiltMs = 0.0f;
        
        void calculate() {
            minMaxCullRate = totalBlocks > 0 ? 1.0f - (float)minMaxPassed / totalBlocks : 0.0f;
            frustumCullRate = minMaxPassed > 0 ? 1.0f - (float)frustumPassed / minMaxPassed : 0.0f;
            hiZCullRate = hiZTested > 0 ? 1.0f - (float)hiZPassed / hiZTested : 0.0f;
            overallCullRate = totalBlocks > 0 ? 1.0f - (float)finalVisible / totalBlocks : 0.0f;
            atomicReductionRate = finalVisible > 0 ? 1.0f - (float)atomicOperations / finalVisible : 0.0f;
        }
    };
    
    // Configuration for debug features
    struct DebugConfig {
        bool enabled = false;
        bool collectMetrics = true;
        bool compareRasterCompute = false;
        bool dumpHiZPyramid = false;
        bool logPerBlockDecisions = false;
        bool measureAtomicContention = false;
        uint32_t captureFrame = UINT32_MAX;  // Specific frame to capture
        uint32_t captureInterval = 60;      // Capture every N frames
    };
    
    // Comparison result between raster and compute
    struct ComparisonResult {
        uint32_t totalBlocks = 0;
        uint32_t matchingBlocks = 0;
        uint32_t rasterOnlyBlocks = 0;
        uint32_t computeOnlyBlocks = 0;
        float matchPercentage = 0.0f;
        std::vector<uint32_t> mismatchedBlockIndices;
        
        void calculate() {
            matchPercentage = totalBlocks > 0 ? 
                (100.0f * matchingBlocks) / totalBlocks : 100.0f;
        }
    };
    
    OcclusionDebugger(VkDevice device, VkPhysicalDevice physicalDevice, VulkanContext& context);
    ~OcclusionDebugger();
    
    // Initialize debug resources
    void initialize(uint32_t maxBlocks);
    
    // Debug configuration
    void setConfig(const DebugConfig& config) { config_ = config; }
    const DebugConfig& getConfig() const { return config_; }
    
    // Frame management
    void beginFrame(VkCommandBuffer cmd, uint32_t frameNumber);
    void endFrame(VkCommandBuffer cmd);
    
    // Metrics collection
    void clearMetrics(VkCommandBuffer cmd);
    void collectMetrics();
    const OcclusionMetrics& getMetrics() const { return currentMetrics_; }
    
    // Timestamp markers (delegates to GPUProfiler)
    void markHiZGenStart(VkCommandBuffer cmd);
    void markHiZGenEnd(VkCommandBuffer cmd);
    void markOcclusionTestStart(VkCommandBuffer cmd);
    void markOcclusionTestEnd(VkCommandBuffer cmd);
    void markPVSBuildStart(VkCommandBuffer cmd);
    void markPVSBuildEnd(VkCommandBuffer cmd);
    
    // PVS Comparison
    ComparisonResult comparePVS(
        const std::vector<uint32_t>& rasterPVS,
        const std::vector<uint32_t>& computePVS,
        uint32_t totalBlocks
    );
    
    // Hi-Z pyramid debugging
    void captureHiZLevel(VkCommandBuffer cmd, VkImageView hiZView, uint32_t level);
    void saveHiZPyramidVisualization(const std::string& basePath);
    
    // Data export
    void dumpMetricsToJSON(const std::string& filepath) const;
    void dumpMetricsToCSV(const std::string& filepath, bool append = true) const;
    void logMetricsToConsole() const;
    
    // Get debug buffer for shader binding
    VkBuffer getMetricsBuffer() const { return metricsBuffer_.buffer; }
    VkDescriptorBufferInfo getMetricsBufferInfo() const {
        return {metricsBuffer_.buffer, 0, VK_WHOLE_SIZE};
    }
    
private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        size_t size = 0;
    };
    
    struct FrameData {
        uint32_t frameNumber = 0;
        OcclusionMetrics metrics;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    VkDevice device_;
    VkPhysicalDevice physicalDevice_;
    VulkanContext& context_;
    
    // Debug configuration
    DebugConfig config_;
    
    // GPU profiler for timing
    std::unique_ptr<GPUProfiler> profiler_;
    
    // Metrics collection buffer (mapped from GPU)
    Buffer metricsBuffer_;
    void* metricsBufferMapped_ = nullptr;
    
    // Current frame metrics
    OcclusionMetrics currentMetrics_;
    uint32_t currentFrame_ = 0;
    
    // History for analysis
    std::vector<FrameData> metricsHistory_;
    static constexpr size_t MAX_HISTORY = 1000;
    
    // Hi-Z capture data
    std::vector<std::vector<float>> capturedHiZLevels_;
    
    // Helper functions
    void createMetricsBuffer(size_t size);
    void destroyMetricsBuffer();
    void readbackMetrics();
    
    // Analysis functions
    float calculateTemporalCoherence() const;
    float calculateAverageMetric(float OcclusionMetrics::*member) const;
    void detectAnomalies() const;
};

// Inline convenience functions for shader debugging
inline void debugMarkOcclusionStart(VkCommandBuffer cmd, OcclusionDebugger* debugger) {
    if (debugger && debugger->getConfig().enabled) {
        debugger->markOcclusionTestStart(cmd);
    }
}

inline void debugMarkOcclusionEnd(VkCommandBuffer cmd, OcclusionDebugger* debugger) {
    if (debugger && debugger->getConfig().enabled) {
        debugger->markOcclusionTestEnd(cmd);
    }
}