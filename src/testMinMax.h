#pragma once

#include <fstream>

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

 struct MinMaxResult {
    uint32_t minVal;
    uint32_t maxVal;
};


const int VOLUME_DIM_X = 256;
const int VOLUME_DIM_Y = 256;
const int VOLUME_DIM_Z = 256;
const int BLOCK_DIM_X = 4;
const int BLOCK_DIM_Y = 4;
const int BLOCK_DIM_Z = 4;
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

// Test function to analyze min-max computation for a specific page
inline void testStreamingMinMaxForPage(
    const std::vector<uint8_t>& volumeData,
    uint32_t volumeSizeX,
    uint32_t volumeSizeY,
    uint32_t volumeSizeZ,
    uint32_t blockSize,
    uint32_t pageX,
    uint32_t pageY,
    uint32_t pageZ,
    uint32_t pageSizeX,
    uint32_t pageSizeY,
    uint32_t pageSizeZ,
    float isovalue)
{
    std::cout << "\n=== Testing Streaming Min-Max for Page (" << pageX << "," << pageY << "," << pageZ << ") ===" << std::endl;
    
    // Calculate page boundaries in world coordinates
    uint32_t pageStartX = pageX * pageSizeX;
    uint32_t pageStartY = pageY * pageSizeY;
    uint32_t pageStartZ = pageZ * pageSizeZ;
    
    // Calculate number of blocks in this page
    uint32_t blocksPerPageX = pageSizeX / blockSize;
    uint32_t blocksPerPageY = pageSizeY / blockSize;
    uint32_t blocksPerPageZ = pageSizeZ / blockSize;
    
    std::cout << "Page starts at world (" << pageStartX << "," << pageStartY << "," << pageStartZ << ")" << std::endl;
    std::cout << "Blocks per page: " << blocksPerPageX << "x" << blocksPerPageY << "x" << blocksPerPageZ << std::endl;
    
    uint32_t activeBlocksInPage = 0;
    uint32_t blocksAffectedByBoundary = 0;
    
    // Process all blocks in this page
    for (uint32_t bz = 0; bz < blocksPerPageZ; bz++) {
        for (uint32_t by = 0; by < blocksPerPageY; by++) {
            for (uint32_t bx = 0; bx < blocksPerPageX; bx++) {
                // Calculate block's world coordinates
                uint32_t blockWorldX = pageStartX + bx * blockSize;
                uint32_t blockWorldY = pageStartY + by * blockSize;
                uint32_t blockWorldZ = pageStartZ + bz * blockSize;
                
                // Compute min-max for this block
                // The shader samples a 5x5x5 region for each 4x4x4 block
                // This includes the block itself plus a 1-voxel halo
                uint32_t minVal = 255;
                uint32_t maxVal = 0;
                uint32_t samplesFromOtherPages = 0;
                uint32_t totalSamples = 0;
                
                // Also compute min-max if we only sampled from current page
                uint32_t minValCurrentPageOnly = 255;
                uint32_t maxValCurrentPageOnly = 0;
                uint32_t samplesInCurrentPage = 0;
                
                // Sample the 5x5x5 region (block + 1 voxel halo)
                for (uint32_t dz = 0; dz <= blockSize; dz++) {
                    for (uint32_t dy = 0; dy <= blockSize; dy++) {
                        for (uint32_t dx = 0; dx <= blockSize; dx++) {
                            uint32_t worldX = blockWorldX + dx;
                            uint32_t worldY = blockWorldY + dy;
                            uint32_t worldZ = blockWorldZ + dz;
                            
                            // Check if out of volume bounds
                            if (worldX >= volumeSizeX || worldY >= volumeSizeY || worldZ >= volumeSizeZ) {
                                continue;
                            }
                            
                            // Check which page this voxel belongs to
                            uint32_t voxelPageX = worldX / pageSizeX;
                            uint32_t voxelPageY = worldY / pageSizeY;
                            uint32_t voxelPageZ = worldZ / pageSizeZ;
                            
                            // Sample the volume
                            uint32_t idx = worldZ * volumeSizeY * volumeSizeX + worldY * volumeSizeX + worldX;
                            uint32_t value = volumeData[idx];
                            
                            // Update global min-max
                            minVal = std::min(minVal, value);
                            maxVal = std::max(maxVal, value);
                            totalSamples++;
                            
                            // Track if sample is from current page
                            if (voxelPageX == pageX && voxelPageY == pageY && voxelPageZ == pageZ) {
                                minValCurrentPageOnly = std::min(minValCurrentPageOnly, value);
                                maxValCurrentPageOnly = std::max(maxValCurrentPageOnly, value);
                                samplesInCurrentPage++;
                            } else {
                                samplesFromOtherPages++;
                            }
                        }
                    }
                }
                
                // Check if this block is active with correct min-max
                bool isActiveCorrect = (minVal <= isovalue && maxVal >= isovalue);
                
                // Check if it would be active with page-only samples
                bool isActivePageOnly = false;
                if (samplesInCurrentPage > 0) {
                    isActivePageOnly = (minValCurrentPageOnly <= isovalue && maxValCurrentPageOnly >= isovalue);
                }
                
                // If the shader returns 0 for non-resident pages, simulate that
                uint32_t minValShaderBehavior = minValCurrentPageOnly;
                uint32_t maxValShaderBehavior = maxValCurrentPageOnly;
                if (samplesFromOtherPages > 0 && samplesInCurrentPage > 0) {
                    // Shader would return 0 for out-of-page samples, affecting min
                    minValShaderBehavior = 0;
                }
                bool isActiveShaderBehavior = (minValShaderBehavior <= isovalue && maxValShaderBehavior >= isovalue);
                
                if (isActiveCorrect) {
                    activeBlocksInPage++;
                }
                
                if (samplesFromOtherPages > 0) {
                    blocksAffectedByBoundary++;
                    
                    // Only print detailed info for blocks where the boundary matters
                    if (isActiveCorrect != isActivePageOnly || isActiveCorrect != isActiveShaderBehavior) {
                        std::cout << "\nBoundary Block (" << bx << "," << by << "," << bz << "):" << std::endl;
                        std::cout << "  World pos: (" << blockWorldX << "," << blockWorldY << "," << blockWorldZ << ")" << std::endl;
                        std::cout << "  Correct min/max: [" << minVal << ", " << maxVal << "] -> " 
                                  << (isActiveCorrect ? "ACTIVE" : "inactive") << std::endl;
                        std::cout << "  Page-only min/max: [" << minValCurrentPageOnly << ", " << maxValCurrentPageOnly << "] -> " 
                                  << (isActivePageOnly ? "ACTIVE" : "inactive") << std::endl;
                        std::cout << "  Shader behavior (0 for other pages): [" << minValShaderBehavior << ", " << maxValShaderBehavior << "] -> " 
                                  << (isActiveShaderBehavior ? "ACTIVE" : "inactive") << std::endl;
                        std::cout << "  Samples: " << samplesInCurrentPage << " in page, " << samplesFromOtherPages << " from other pages" << std::endl;
                    }
                }
            }
        }
    }
    
    std::cout << "\nPage Summary:" << std::endl;
    std::cout << "  Total blocks: " << (blocksPerPageX * blocksPerPageY * blocksPerPageZ) << std::endl;
    std::cout << "  Active blocks (correct): " << activeBlocksInPage << std::endl;
    std::cout << "  Blocks affected by page boundary: " << blocksAffectedByBoundary << std::endl;
}