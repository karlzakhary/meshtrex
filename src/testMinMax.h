#pragma once

#include "blockFiltering.h"

/**
 * @brief Reads a raw uint8 volume file and computes the min/max value for each block.
 *
 * @param filePath Path to the raw volume file.
 * @return std::vector<MinMaxResult> Vector of min/max results per block.
 * @throws std::runtime_error if file cannot be opened, read, or has incorrect size.
 *
 * @assumptions
 * - File is raw binary data (no header).
 * - Data type is uint8_t.
 * - Data layout is Z-major (slice by slice, row by row).
 * - Volume dimensions are hardcoded inside the function (INFLEXIBLE!).
 */
const int VOLUME_DIM_X = 256;
const int VOLUME_DIM_Y = 256;
const int VOLUME_DIM_Z = 256;
const int BLOCK_DIM_X = 8;
const int BLOCK_DIM_Y = 8;
const int BLOCK_DIM_Z = 8;
const int GRID_DIM_X = static_cast<int>(std::ceil(static_cast<double>(VOLUME_DIM_X) / BLOCK_DIM_X)); // 4
const int GRID_DIM_Y = static_cast<int>(std::ceil(static_cast<double>(VOLUME_DIM_Y) / BLOCK_DIM_Y)); // 4
const int GRID_DIM_Z = static_cast<int>(std::ceil(static_cast<double>(VOLUME_DIM_Z) / BLOCK_DIM_Z)); // 2
inline std::vector<MinMaxResult> computeMinMaxFromFile(const std::string& filePath)
{
    // --- Configuration (Hardcoded - Needs Verification/Flexibility!) ---

    // --- End Configuration ---

    std::cout << "Processing file: " << filePath << std::endl;
    std::cout << "Assuming Dimensions: " << VOLUME_DIM_X << "x" << VOLUME_DIM_Y
              << "x" << VOLUME_DIM_Z << std::endl;
    std::cout << "Using Block Dimensions: " << BLOCK_DIM_X << "x" << BLOCK_DIM_Y
              << "x" << BLOCK_DIM_Z << std::endl;
    std::cout << "Resulting Grid Dimensions: " << GRID_DIM_X << "x"
              << GRID_DIM_Y << "x" << GRID_DIM_Z << " ("
              << GRID_DIM_X * GRID_DIM_Y * GRID_DIM_Z << " blocks total)"
              << std::endl;
    std::cout << "Assuming Data Type: uint8_t" << std::endl;

    // --- File Reading ---
    std::ifstream inFile(filePath, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Error: Cannot open file: " + filePath);
    }
    std::streamsize fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    size_t expectedSize =
        static_cast<size_t>(VOLUME_DIM_X) * VOLUME_DIM_Y * VOLUME_DIM_Z;
    if (static_cast<size_t>(fileSize) != expectedSize) {
        inFile.close();
        throw std::runtime_error("Error: File size mismatch. Expected " +
                                 std::to_string(expectedSize) +
                                 " bytes, but file has " +
                                 std::to_string(fileSize) + " bytes.");
    }
    std::vector<uint8_t> volumeData(expectedSize);
    if (!inFile.read(reinterpret_cast<char *>(volumeData.data()),
                     expectedSize)) {
        inFile.close();
        throw std::runtime_error("Error: Failed to read data from file: " +
                                 filePath);
    }
    inFile.close();
    std::cout << "Successfully read " << expectedSize << " bytes." << std::endl;
    // --- End File Reading ---

    // --- Min/Max Calculation ---
    std::cout << "Computing Min/Max (uint8_t)..." << std::endl;
    size_t totalBlocks =
        static_cast<size_t>(GRID_DIM_X) * GRID_DIM_Y * GRID_DIM_Z;
    std::vector<MinMaxResult> results(
        totalBlocks);  // Vector of {uint8_t, uint8_t} structs

    for (int bz = 0; bz < GRID_DIM_Z; ++bz) {
        for (int by = 0; by < GRID_DIM_Y; ++by) {
            for (int bx = 0; bx < GRID_DIM_X; ++bx) {
                // Initialize with uint8_t limits
                uint8_t currentMin =
                    std::numeric_limits<uint8_t>::max();  // 255
                uint8_t currentMax = std::numeric_limits<uint8_t>::min();  // 0
                bool blockHasValidVoxel = false;

                for (int lz = 0; lz < BLOCK_DIM_Z; ++lz) {
                    for (int ly = 0; ly < BLOCK_DIM_Y; ++ly) {
                        for (int lx = 0; lx < BLOCK_DIM_X; ++lx) {
                            int gx = bx * BLOCK_DIM_X + lx;
                            int gy = by * BLOCK_DIM_Y + ly;
                            int gz = bz * BLOCK_DIM_Z + lz;

                            if (gx < VOLUME_DIM_X && gy < VOLUME_DIM_Y &&
                                gz < VOLUME_DIM_Z) {
                                size_t index =
                                    static_cast<size_t>(gz) * VOLUME_DIM_X *
                                        VOLUME_DIM_Y +
                                    static_cast<size_t>(gy) * VOLUME_DIM_X +
                                    static_cast<size_t>(gx);
                                uint8_t voxelValue =
                                    volumeData[index];  // Read uint8_t
                                currentMin = std::min(
                                    currentMin, voxelValue);  // Compare uint8_t
                                currentMax = std::max(
                                    currentMax, voxelValue);  // Compare uint8_t
                                blockHasValidVoxel = true;
                            }
                        }
                    }
                }

                size_t blockIndex =
                    static_cast<size_t>(bz) * GRID_DIM_X * GRID_DIM_Y +
                    static_cast<size_t>(by) * GRID_DIM_X +
                    static_cast<size_t>(bx);

                if (blockHasValidVoxel) {
                    // Store uint8_t results
                    results[blockIndex] = {currentMin, currentMax};
                    // std::cout << "Block " << blockIndex << ", Min: "  <<
                    // int(currentMin) << ", Max: " << (int)currentMax <<
                    // std::endl;
                } else {
                    results[blockIndex] = {0, 0};  // Default for empty blocks
                }
            }
        }
    }
    std::cout << "Min/Max computation finished." << std::endl;
    // --- End Min/Max Calculation ---

    return results;  // Return vector of {uint8_t, uint8_t} structs
}

inline int compareMinMaxResults(const std::vector<MinMaxResult>& gpuResults,
                                const std::vector<MinMaxResult>& cpuResults,
                                int maxErrorsToPrint = 10)
{
    std::cout << "\nComparing GPU vs CPU results..." << std::endl;

    // 1. Check sizes
    if (gpuResults.size() != cpuResults.size()) {
        std::cerr << "Error: GPU result count (" << gpuResults.size()
                  << ") does not match CPU result count (" << cpuResults.size()
                  << ")." << std::endl;
        return -1;  // Indicate error
    }

    if (cpuResults.empty()) {
        std::cout << "Result vectors are empty, nothing to compare."
                  << std::endl;
        return 0;  // No errors found
    }

    int mismatchCount = 0;
    int errorsPrinted = 0;
    size_t totalBlocks = cpuResults.size();  // Should match gpuResults.size()

    // 2. Compare element by element
    for (size_t i = 0; i < totalBlocks; ++i) {
        bool minMatch = (gpuResults[i].minVal == cpuResults[i].minVal);
        bool maxMatch = (gpuResults[i].maxVal == cpuResults[i].maxVal);

        if (!minMatch || !maxMatch) {
            mismatchCount++;
            if (errorsPrinted < maxErrorsToPrint) {
                // Calculate 3D block index from 1D index i
                // Ensure gridDimX/Y are non-zero to avoid division by zero
                int planeSize = GRID_DIM_X * GRID_DIM_Y;
                int bz = (planeSize > 0) ? (i / planeSize) : 0;
                int remainder = (planeSize > 0) ? (i % planeSize) : i;
                int by = (GRID_DIM_X > 0) ? (remainder / GRID_DIM_X) : 0;
                int bx =
                    (GRID_DIM_X > 0) ? (remainder % GRID_DIM_X) : remainder;

                std::cerr << "Mismatch found at Block [" << bx << ", " << by
                          << ", " << bz << "] (Index " << i
                          << "):" << std::endl;
                if (!minMatch) {
                    // Cast uint8_t to int for printing to avoid ASCII
                    // interpretation
                    std::cerr
                        << "  Min mismatch: GPU="
                        << static_cast<int>(gpuResults[i].minVal)
                        << ", CPU=" << static_cast<int>(cpuResults[i].minVal)
                        << std::endl;
                }
                if (!maxMatch) {
                    // Cast uint8_t to int for printing
                    std::cerr
                        << "  Max mismatch: GPU="
                        << static_cast<int>(gpuResults[i].maxVal)
                        << ", CPU=" << static_cast<int>(cpuResults[i].maxVal)
                        << std::endl;
                }
                errorsPrinted++;
            } else if (errorsPrinted == maxErrorsToPrint) {
                std::cerr << "... (further mismatch details suppressed)"
                          << std::endl;
                errorsPrinted++;  // Prevent printing this message again
            }
        }
    }

    // 3. Report results
    if (mismatchCount == 0) {
        std::cout << "Success: All " << totalBlocks << " block results match!"
                  << std::endl;
    } else {
        std::cout << "Comparison finished: Found " << mismatchCount
                  << " mismatches out of " << totalBlocks << " blocks."
                  << std::endl;
    }

    return mismatchCount;
}

inline uint32_t computeActiveBlockCountCPU(const std::vector<MinMaxResult>& minMaxResults, float isovalue) {
    std::cout << "\nCPU: Calculating active block count for isovalue " << isovalue << "..." << std::endl;
    uint32_t activeCount = 0;
    for (const auto& result : minMaxResults) {
        // Apply the EXACT SAME logic as the filtering shader
        bool blockIsActive = false;
        // Check if min and max are different to avoid trivial blocks (matches shader)
        if (result.minVal != result.maxVal) {
            // Check if isovalue is within the range [min, max]
            // Cast uint32_t min/max (which hold 0-255 range) to float for comparison
            blockIsActive = (isovalue >= static_cast<float>(result.minVal) &&
                             isovalue <= static_cast<float>(result.maxVal));
        }
        // If you used the simpler check in the shader, use it here too:
        // blockIsActive = (isovalue >= static_cast<float>(result.minVal) && isovalue <= static_cast<float>(result.maxVal));

        if (blockIsActive) {
            activeCount++;
        }
    }
    std::cout << "CPU: Active block count calculation finished. Found: " << activeCount << std::endl;
    return activeCount;
}