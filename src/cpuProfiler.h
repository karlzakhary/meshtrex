#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stack>

class CPUProfiler {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::milli>;

    struct ProfileData {
        double totalTime = 0.0;
        double minTime = std::numeric_limits<double>::max();
        double maxTime = 0.0;
        uint64_t callCount = 0;
    };

    class ScopedTimer {
    public:
        ScopedTimer(CPUProfiler& profiler, const std::string& name)
            : profiler_(profiler), name_(name) {
            profiler_.beginRegion(name_);
        }
        
        ~ScopedTimer() {
            profiler_.endRegion(name_);
        }

    private:
        CPUProfiler& profiler_;
        std::string name_;
    };

    void beginRegion(const std::string& name) {
        activeTimers_.push({name, Clock::now()});
    }

    void endRegion(const std::string& name) {
        if (activeTimers_.empty()) {
            std::cerr << "CPUProfiler: No active timer to end!" << std::endl;
            return;
        }

        auto endTime = Clock::now();
        auto& timer = activeTimers_.top();
        
        if (timer.first != name) {
            std::cerr << "CPUProfiler: Timer mismatch! Expected: " << timer.first 
                      << ", Got: " << name << std::endl;
            return;
        }

        Duration elapsed = endTime - timer.second;
        double elapsedMs = elapsed.count();

        auto& data = profileData_[name];
        data.totalTime += elapsedMs;
        data.minTime = std::min(data.minTime, elapsedMs);
        data.maxTime = std::max(data.maxTime, elapsedMs);
        data.callCount++;

        activeTimers_.pop();
    }

    void reset() {
        profileData_.clear();
        while (!activeTimers_.empty()) {
            activeTimers_.pop();
        }
    }

    void printResults() const {
        std::cout << "\n=== CPU Performance Profile ===" << std::endl;
        std::cout << std::setw(30) << std::left << "Function"
                  << std::setw(12) << std::right << "Total (ms)"
                  << std::setw(12) << "Avg (ms)"
                  << std::setw(12) << "Min (ms)"
                  << std::setw(12) << "Max (ms)"
                  << std::setw(10) << "Calls" << std::endl;
        std::cout << std::string(88, '-') << std::endl;

        std::vector<std::pair<std::string, ProfileData>> sortedData(
            profileData_.begin(), profileData_.end()
        );
        
        // Sort by total time descending
        std::sort(sortedData.begin(), sortedData.end(),
            [](const auto& a, const auto& b) {
                return a.second.totalTime > b.second.totalTime;
            }
        );

        for (const auto& [name, data] : sortedData) {
            double avgTime = data.totalTime / data.callCount;
            
            std::cout << std::setw(30) << std::left << name
                      << std::setw(12) << std::right << std::fixed << std::setprecision(3) 
                      << data.totalTime
                      << std::setw(12) << avgTime
                      << std::setw(12) << data.minTime
                      << std::setw(12) << data.maxTime
                      << std::setw(10) << data.callCount << std::endl;
        }
    }

    const std::unordered_map<std::string, ProfileData>& getData() const {
        return profileData_;
    }

private:
    std::unordered_map<std::string, ProfileData> profileData_;
    std::stack<std::pair<std::string, TimePoint>> activeTimers_;
};

// Convenience macro for scoped profiling
#define CPU_PROFILE(profiler, name) CPUProfiler::ScopedTimer _timer##__LINE__(profiler, name)