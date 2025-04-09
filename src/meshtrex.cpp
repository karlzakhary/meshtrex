#include <vector>
#ifndef __APPLE__
#include <cstdint>
#endif
#include "blockFiltering.h"
#include "testMinMax.h"

int main(int argc, char* argv[]) {
    // Compute Min/Max for all blocks by calling the single function
    std::vector<MinMaxResult> cpuMinMaxResults = computeMinMaxFromFile("../cmake-build-debug/raw_volumes/aneurism_256x256x256_uint8.raw");

    // std::vector<MinMaxResult> gpudata = filterUnoccupiedBlocks(
    //     argv, "raw_volumes/aneurism_256x256x256_uint8.raw");

    // compareMinMaxResults(gpudata, cpuMinMaxResults);
    // 2. Compute Active Block Count on CPU using the min/max results
    uint32_t cpuActiveCount = computeActiveBlockCountCPU(cpuMinMaxResults, 60);

    // 3. Run the GPU Occupied Block Filtering pass
    // This function should return the count read back from the GPU atomic counter
    // uint32_t gpuActiveCount = runOccupiedBlockFiltering(
    //     argv, device, computeQueue, commandPool, memoryProperties,
    //     minMaxImage, pushData, compactedBlockIdBuffer, activeBlockCountBuffer
    // );
    uint32_t gpuActiveCount = filterUnoccupiedBlocks(
        argv, "raw_volumes/aneurism_256x256x256_uint8.raw"); // Placeholder - replace with actual GPU result
    std::cout << "\nGPU: Occupied Block Filtering finished. Active blocks found: " << gpuActiveCount << " (Placeholder)" << std::endl;


    // // 4. Compare the counts
    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "CPU Active Block Count: " << cpuActiveCount << std::endl;
    std::cout << "GPU Active Block Count: " << gpuActiveCount << std::endl;
    if (cpuActiveCount == gpuActiveCount) {
        std::cout << "Success: CPU and GPU active block counts match!" << std::endl;
    } else {
        std::cerr << "Error: CPU and GPU active block counts DO NOT match!" << std::endl;
    }
    return 0;
}