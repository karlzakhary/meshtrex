#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <iomanip>

class GPUProfiler {
public:
    struct TimestampLabel {
        std::string name;
        uint32_t startIndex;
        uint32_t endIndex;
    };

    struct ProfileResult {
        std::string name;
        double timeMs;
    };

    GPUProfiler(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t timestampCount = 64)
        : device_(device), physicalDevice_(physicalDevice), maxTimestamps_(timestampCount) {
        // Check if timestamps are supported
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
        
        if (properties.limits.timestampComputeAndGraphics == VK_FALSE) {
            throw std::runtime_error("GPU does not support timestamp queries in compute and graphics queues!");
        }
        
        createQueryPool();
        queryTimestampPeriod();
    }

    ~GPUProfiler() {
        if (queryPool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, queryPool_, nullptr);
        }
    }

    void beginFrame(VkCommandBuffer cmd) {
        currentTimestamp_ = 0;
        timestampLabels_.clear();
        vkCmdResetQueryPool(cmd, queryPool_, 0, maxTimestamps_);
    }

    uint32_t writeTimestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits stage, const std::string& label) {
        if (currentTimestamp_ >= maxTimestamps_) {
            std::cerr << "GPU Profiler: Exceeded maximum timestamps!" << std::endl;
            return UINT32_MAX;
        }
        
        uint32_t index = currentTimestamp_++;
        vkCmdWriteTimestamp(cmd, stage, queryPool_, index);
        
        // Store label for later retrieval
        if (pendingLabel_.empty()) {
            pendingLabel_ = label;
            pendingStartIndex_ = index;
        } else {
            // This is an end timestamp
            timestampLabels_.push_back({pendingLabel_, pendingStartIndex_, index});
            pendingLabel_.clear();
        }
        
        return index;
    }

    void beginProfileRegion(VkCommandBuffer cmd, VkPipelineStageFlagBits stage, const std::string& label) {
        writeTimestamp(cmd, stage, label);
    }

    void endProfileRegion(VkCommandBuffer cmd, VkPipelineStageFlagBits stage) {
        writeTimestamp(cmd, stage, "");
    }

    std::vector<ProfileResult> getResults() {
        std::vector<ProfileResult> results;
        
        if (currentTimestamp_ == 0) return results;

        std::vector<uint64_t> timestamps(currentTimestamp_);
        VkResult result = vkGetQueryPoolResults(
            device_, queryPool_, 0, currentTimestamp_,
            timestamps.size() * sizeof(uint64_t), timestamps.data(),
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
        );

        if (result != VK_SUCCESS) {
            std::cerr << "Failed to get query results!" << std::endl;
            return results;
        }

        for (const auto& label : timestampLabels_) {
            if (label.startIndex < timestamps.size() && label.endIndex < timestamps.size()) {
                uint64_t startTime = timestamps[label.startIndex];
                uint64_t endTime = timestamps[label.endIndex];
                double timeMs = (endTime - startTime) * timestampPeriod_ / 1000000.0;
                results.push_back({label.name, timeMs});
            }
        }

        return results;
    }

    void printResults() {
        auto results = getResults();
        
        std::cout << "\n=== GPU Performance Profile ===" << std::endl;
        std::cout << std::setw(40) << std::left << "Stage" 
                  << std::setw(15) << std::right << "Time (ms)" << std::endl;
        std::cout << std::string(55, '-') << std::endl;
        
        double totalTime = 0.0;
        for (const auto& result : results) {
            std::cout << std::setw(40) << std::left << result.name 
                      << std::setw(15) << std::right << std::fixed 
                      << std::setprecision(3) << result.timeMs << std::endl;
            totalTime += result.timeMs;
        }
        
        std::cout << std::string(55, '-') << std::endl;
        std::cout << std::setw(40) << std::left << "Total GPU Time" 
                  << std::setw(15) << std::right << std::fixed 
                  << std::setprecision(3) << totalTime << std::endl;
    }

private:
    void createQueryPool() {
        VkQueryPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        createInfo.queryCount = maxTimestamps_;

        if (vkCreateQueryPool(device_, &createInfo, nullptr, &queryPool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create query pool!");
        }
    }

    void queryTimestampPeriod() {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
        timestampPeriod_ = properties.limits.timestampPeriod;
        
        std::cout << "GPU timestamp period: " << timestampPeriod_ << " ns" << std::endl;
    }

    VkDevice device_;
    VkPhysicalDevice physicalDevice_;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    uint32_t maxTimestamps_;
    uint32_t currentTimestamp_ = 0;
    float timestampPeriod_ = 1.0f;
    
    std::vector<TimestampLabel> timestampLabels_;
    std::string pendingLabel_;
    uint32_t pendingStartIndex_ = 0;
};