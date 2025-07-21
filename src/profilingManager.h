#pragma once

#include "gpuProfiler.h"
#include "cpuProfiler.h"
#include <memory>
#include <fstream>
#include <sstream>

class ProfilingManager {
public:
    struct FrameStats {
        double cpuFrameTime = 0.0;
        double gpuFrameTime = 0.0;
        double minMaxLeafTime = 0.0;
        double minMaxOctreeTime = 0.0;
        double filteringTime = 0.0;
        double extractionTime = 0.0;
        double renderingTime = 0.0;
        uint32_t activeBlocks = 0;
        uint32_t vertices = 0;
        uint32_t triangles = 0;
        uint32_t meshlets = 0;
    };

    ProfilingManager(VkDevice device, VkPhysicalDevice physicalDevice)
        : gpuProfiler_(std::make_unique<GPUProfiler>(device, physicalDevice)),
          cpuProfiler_(std::make_unique<CPUProfiler>()) {
        frameStats_.reserve(1000); // Reserve space for 1000 frames
    }

    GPUProfiler& gpu() { return *gpuProfiler_; }
    CPUProfiler& cpu() { return *cpuProfiler_; }

    void beginFrame(VkCommandBuffer cmd) {
        frameStartTime_ = std::chrono::high_resolution_clock::now();
        gpuProfiler_->beginFrame(cmd);
        currentFrame_ = FrameStats{};
    }

    void endFrame() {
        auto frameEndTime = std::chrono::high_resolution_clock::now();
        currentFrame_.cpuFrameTime = std::chrono::duration<double, std::milli>(
            frameEndTime - frameStartTime_
        ).count();

        // Get GPU results
        auto gpuResults = gpuProfiler_->getResults();
        for (const auto& result : gpuResults) {
            currentFrame_.gpuFrameTime += result.timeMs;
            
            if (result.name == "MinMax_Leaf_Pass") {
                currentFrame_.minMaxLeafTime = result.timeMs;
            } else if (result.name == "MinMax_Octree_Reduction") {
                currentFrame_.minMaxOctreeTime = result.timeMs;
            } else if (result.name.find("Filtering") != std::string::npos) {
                currentFrame_.filteringTime = result.timeMs;
            } else if (result.name.find("Extraction") != std::string::npos) {
                currentFrame_.extractionTime = result.timeMs;
            } else if (result.name.find("Rendering") != std::string::npos) {
                currentFrame_.renderingTime = result.timeMs;
            }
        }

        frameStats_.push_back(currentFrame_);
        frameCount_++;
    }

    void setExtractionStats(uint32_t activeBlocks, uint32_t vertices, uint32_t triangles, uint32_t meshlets) {
        currentFrame_.activeBlocks = activeBlocks;
        currentFrame_.vertices = vertices;
        currentFrame_.triangles = triangles;
        currentFrame_.meshlets = meshlets;
    }

    void printSummary() {
        std::cout << "\n=== Performance Summary ===" << std::endl;
        
        // Calculate averages
        if (frameStats_.empty()) return;

        double avgCpuTime = 0.0, avgGpuTime = 0.0;
        double avgMinMaxLeaf = 0.0, avgMinMaxOctree = 0.0, avgFiltering = 0.0, avgExtraction = 0.0, avgRendering = 0.0;
        uint32_t avgActiveBlocks = 0, avgVertices = 0, avgTriangles = 0;

        for (const auto& frame : frameStats_) {
            avgCpuTime += frame.cpuFrameTime;
            avgGpuTime += frame.gpuFrameTime;
            avgMinMaxLeaf += frame.minMaxLeafTime;
            avgMinMaxOctree += frame.minMaxOctreeTime;
            avgFiltering += frame.filteringTime;
            avgExtraction += frame.extractionTime;
            avgRendering += frame.renderingTime;
            avgActiveBlocks += frame.activeBlocks;
            avgVertices += frame.vertices;
            avgTriangles += frame.triangles;
        }

        size_t frameCount = frameStats_.size();
        avgCpuTime /= frameCount;
        avgGpuTime /= frameCount;
        avgMinMaxLeaf /= frameCount;
        avgMinMaxOctree /= frameCount;
        avgFiltering /= frameCount;
        avgExtraction /= frameCount;
        avgRendering /= frameCount;
        avgActiveBlocks /= frameCount;
        avgVertices /= frameCount;
        avgTriangles /= frameCount;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average Frame Times:" << std::endl;
        std::cout << "  CPU Frame Time: " << avgCpuTime << " ms (" 
                  << (1000.0 / avgCpuTime) << " FPS)" << std::endl;
        std::cout << "  GPU Frame Time: " << avgGpuTime << " ms (" 
                  << (1000.0 / avgGpuTime) << " FPS)" << std::endl;
        std::cout << "\nGPU Stage Breakdown:" << std::endl;
        std::cout << "  Min-Max Leaf Pass: " << avgMinMaxLeaf << " ms (" 
                  << (avgMinMaxLeaf / avgGpuTime * 100.0) << "%)" << std::endl;
        std::cout << "  Min-Max Octree Reduction: " << avgMinMaxOctree << " ms (" 
                  << (avgMinMaxOctree / avgGpuTime * 100.0) << "%)" << std::endl;
        std::cout << "  Min-Max Total: " << (avgMinMaxLeaf + avgMinMaxOctree) << " ms (" 
                  << ((avgMinMaxLeaf + avgMinMaxOctree) / avgGpuTime * 100.0) << "%)" << std::endl;
        std::cout << "  Active Filtering: " << avgFiltering << " ms (" 
                  << (avgFiltering / avgGpuTime * 100.0) << "%)" << std::endl;
        std::cout << "  Mesh Extraction: " << avgExtraction << " ms (" 
                  << (avgExtraction / avgGpuTime * 100.0) << "%)" << std::endl;
        std::cout << "  Rendering: " << avgRendering << " ms (" 
                  << (avgRendering / avgGpuTime * 100.0) << "%)" << std::endl;
        std::cout << "\nGeometry Statistics:" << std::endl;
        std::cout << "  Active Blocks: " << avgActiveBlocks << std::endl;
        std::cout << "  Vertices: " << avgVertices << std::endl;
        std::cout << "  Triangles: " << avgTriangles << std::endl;
        
        // Also print CPU profiling results
        cpuProfiler_->printResults();
        
        // Print GPU detailed results
        gpuProfiler_->printResults();
    }

    void exportCSV(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open CSV file: " << filename << std::endl;
            return;
        }

        // Write header
        file << "Frame,CPU_Time_ms,GPU_Time_ms,MinMax_Leaf_ms,MinMax_Octree_ms,Filtering_ms,Extraction_ms,"
             << "Rendering_ms,Active_Blocks,Vertices,Triangles,Meshlets\n";

        // Write data
        for (size_t i = 0; i < frameStats_.size(); ++i) {
            const auto& frame = frameStats_[i];
            file << i << ","
                 << frame.cpuFrameTime << ","
                 << frame.gpuFrameTime << ","
                 << frame.minMaxLeafTime << ","
                 << frame.minMaxOctreeTime << ","
                 << frame.filteringTime << ","
                 << frame.extractionTime << ","
                 << frame.renderingTime << ","
                 << frame.activeBlocks << ","
                 << frame.vertices << ","
                 << frame.triangles << ","
                 << frame.meshlets << "\n";
        }

        file.close();
        std::cout << "Performance data exported to: " << filename << std::endl;
    }

private:
    std::unique_ptr<GPUProfiler> gpuProfiler_;
    std::unique_ptr<CPUProfiler> cpuProfiler_;
    std::vector<FrameStats> frameStats_;
    FrameStats currentFrame_;
    std::chrono::high_resolution_clock::time_point frameStartTime_;
    uint64_t frameCount_ = 0;
};