#include "extractionTestUtils.h"
#include "blockFilteringTestUtils.h" // For mapUintBuffer
#include "mc_tables.h"               // For MarchingCubes::triTable
#include "vulkan_utils.h"

#include <vector>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <map> // Using map for edge->vertex to keep order (helps comparison slightly)
#define GLM_ENABLE_EXPERIMENTAL
#include <iomanip>

#include "glm/gtx/string_cast.hpp"

// --- Constants (match shader/C++ setup) ---
const int BLOCK_DIM_X_MAX = 8; // Max dimensions the process starts with
const int BLOCK_DIM_Y_MAX = 8;
const int BLOCK_DIM_Z_MAX = 8;
const int MIN_BLOCK_DIM = 2; // Smallest block size to recurse down to
const int MAX_MESHLET_VERTICES = 512;
const int MAX_MESHLET_PRIMITIVES = 512;

// --- Helper Structures ---
struct CPUVertex { /* ... Same as before ... */
    glm::vec3 pos; glm::vec3 norm;
    bool operator==(const CPUVertex& other) const { return pos == other.pos && norm == other.norm;}
};
struct VolumeEdge { /* ... Same as before ... */
    glm::ivec3 p1; int axis;
     bool operator==(const VolumeEdge& other) const { return p1 == other.p1 && axis == other.axis; }
     // Add less-than operator if using std::map instead of unordered_map
     bool operator<(const VolumeEdge& other) const {
         if (p1.x != other.p1.x) return p1.x < other.p1.x;
         if (p1.y != other.p1.y) return p1.y < other.p1.y;
         if (p1.z != other.p1.z) return p1.z < other.p1.z;
         return axis < other.axis;
     }
};
// Hash function for VolumeEdge (Needed if using unordered_map)
namespace std { /* ... Same std::hash<VolumeEdge> specialization ... */
    template <> struct hash<VolumeEdge> {
        std::size_t operator()(const VolumeEdge& e) const {
            size_t h1 = std::hash<int>{}(e.p1.x); size_t h2 = std::hash<int>{}(e.p1.y); size_t h3 = std::hash<int>{}(e.p1.z); size_t h4 = std::hash<int>{}(e.axis);
            size_t seed = 0; seed ^= h1 + 0x9e3779b9 + (seed << 6) + (seed >> 2); seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2); seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2); seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2); return seed;
        }
    };
}

// --- Helper Functions (Sampling, Vertex/Normal Calc, Single Case Estimate - unchanged) ---
inline uint8_t sampleVolume(const Volume& volume, const glm::ivec3& coord) { /* ... Same ... */
    if (coord.x < 0 || coord.x >= volume.volume_dims.x || coord.y < 0 || coord.y >= volume.volume_dims.y || coord.z < 0 || coord.z >= volume.volume_dims.z) return 0;
    size_t index = static_cast<size_t>(coord.z) * volume.volume_dims.x * volume.volume_dims.y + static_cast<size_t>(coord.y) * volume.volume_dims.x + static_cast<size_t>(coord.x);
    if (index >= volume.volume_data.size()) return 0; return volume.volume_data[index];
 }
inline glm::vec3 calculateVertexPosCPU(const glm::ivec3& p1_coord, const glm::ivec3& p2_coord, uint8_t val1_uint, uint8_t val2_uint, float isovalue) { /* ... Same ... */
    float val1 = static_cast<float>(val1_uint); float val2 = static_cast<float>(val2_uint); float denom = val2 - val1; if (std::abs(denom) < 1e-5f) return glm::vec3(p1_coord); float t = (isovalue - val1) / denom; return glm::mix(glm::vec3(p1_coord), glm::vec3(p2_coord), glm::clamp(t, 0.0f, 1.0f));
 }
inline glm::vec3 calculateVertexNormalCPU(const Volume& volume, const glm::vec3& pos) { /* ... Same ... */
     glm::ivec3 ipos = glm::ivec3(glm::round(pos)); float dx = (static_cast<float>(sampleVolume(volume, ipos + glm::ivec3(1, 0, 0))) - static_cast<float>(sampleVolume(volume, ipos - glm::ivec3(1, 0, 0)))); float dy = (static_cast<float>(sampleVolume(volume, ipos + glm::ivec3(0, 1, 0))) - static_cast<float>(sampleVolume(volume, ipos - glm::ivec3(0, 1, 0)))); float dz = (static_cast<float>(sampleVolume(volume, ipos + glm::ivec3(0, 0, 1))) - static_cast<float>(sampleVolume(volume, ipos - glm::ivec3(0, 0, 1)))); glm::vec3 grad(dx, dy, dz); if (glm::length(grad) < 1e-5f) return glm::vec3(0.0f, 1.0f, 0.0f); return -glm::normalize(grad);
 }
inline uint32_t estimateGeometryCPU(uint32_t mc_case) { /* ... Same ... */
    uint32_t primCount = 0; uint32_t vertCount = 0; int idx = 0; const int* table_entry = MarchingCubes::triTable[mc_case]; while (idx < 15 && table_entry[idx] != -1) { primCount++; idx += 3; } primCount /= 3; if (primCount > 0) { vertCount = primCount + 2; } return (primCount << 16) | vertCount;
 }

// --- NEW: Helper to estimate geometry for a given block/sub-block ---
struct BlockEstimate {
    uint32_t estVertexCount = 0;
    uint32_t estPrimCount = 0;
};

BlockEstimate estimateGeometryCPUForBlock(
    const glm::uvec3& blockOrigin,
    const glm::uvec3& blockDim,
    const Volume& volume,
    float isovalue)
{
    BlockEstimate estimate;
    for (uint32_t lz = 0; lz < blockDim.z; ++lz) {
        for (uint32_t ly = 0; ly < blockDim.y; ++ly) {
            for (uint32_t lx = 0; lx < blockDim.x; ++lx) {
                glm::ivec3 localCellCoord(lx, ly, lz);
                glm::ivec3 globalCellCoord = glm::ivec3(blockOrigin) + localCellCoord;
                uint32_t mc_case = 0;
                bool boundary = false;
                for (int i = 0; i < 8; ++i) {
                    glm::ivec3 cornerOffset((i & 1), (i & 2) >> 1, (i & 4) >> 2);
                    glm::ivec3 cornerCoord = globalCellCoord + cornerOffset;
                    if (cornerCoord.x >= volume.volume_dims.x || cornerCoord.y >= volume.volume_dims.y || cornerCoord.z >= volume.volume_dims.z || cornerCoord.x < 0 || cornerCoord.y < 0 || cornerCoord.z < 0) { boundary = true; break; }
                    if (static_cast<float>(sampleVolume(volume, cornerCoord)) >= isovalue) { mc_case |= (1 << i); }
                }
                if (boundary || mc_case == 0 || mc_case == 255) continue;
                uint32_t counts = estimateGeometryCPU(mc_case);
                estimate.estPrimCount += (counts >> 16);
                estimate.estVertexCount += (counts & 0xFFFF);
            }
        }
    }
    return estimate;
}


// --- Meshlet Generation for Sub-Block (Revised Warning) ---
void generateMeshletForSubBlockCPU(
    const glm::uvec3& subBlockOrigin, // Use the actual origin of the sub-block being processed
    const glm::uvec3& subBlockDim,
    const Volume& volume,
    float isovalue,
    CPUExtractionOutput& cpuOutput,
    uint32_t& globalVertexCounter,
    uint32_t& globalIndexCounter
) {
    std::vector<CPUVertex> localVertices;
    std::vector<uint32_t> localIndices;
    localVertices.reserve(MAX_MESHLET_VERTICES);
    localIndices.reserve(MAX_MESHLET_PRIMITIVES * 3);

    // Using std::map for potentially more deterministic iteration order than unordered_map
    std::map<VolumeEdge, uint32_t> uniqueVertexMap;

    bool vertexLimitHit = false; // Flag to track if limit was hit

    // Iterate through cells WITHIN this sub-block
    for (uint32_t lz = 0; lz < subBlockDim.z; ++lz) {
        for (uint32_t ly = 0; ly < subBlockDim.y; ++ly) {
            for (uint32_t lx = 0; lx < subBlockDim.x; ++lx) {
                glm::ivec3 localCellCoord(lx, ly, lz);
                glm::ivec3 globalCellCoord = glm::ivec3(subBlockOrigin) + localCellCoord; // Use subBlockOrigin

                uint32_t mc_case = 0;
                uint8_t cornerValues_uint[8];
                 bool boundary = false;
                for (int i = 0; i < 8; ++i) { /* ... Calculate mc_case, handle boundary ... */
                    glm::ivec3 cornerOffset((i & 1), (i & 2) >> 1, (i & 4) >> 2); glm::ivec3 cornerCoord = globalCellCoord + cornerOffset;
                    if (cornerCoord.x >= volume.volume_dims.x || cornerCoord.y >= volume.volume_dims.y || cornerCoord.z >= volume.volume_dims.z || cornerCoord.x < 0 || cornerCoord.y < 0 || cornerCoord.z < 0) { boundary = true; break; }
                    cornerValues_uint[i] = sampleVolume(volume, cornerCoord); if (static_cast<float>(cornerValues_uint[i]) >= isovalue) { mc_case |= (1 << i); }
                }
                if (boundary || mc_case == 0 || mc_case == 255) continue;

                const int* table_entry = MarchingCubes::triTable[mc_case];
                for (int v = 0; v < 12; v += 3) {
                    if (table_entry[v] == -1) break;

                    uint32_t triangleLocalIndices[3];
                    bool triangleValid = true;

                    for (int i = 0; i < 3; ++i) {
                        int edgeIndex = table_entry[v + i];
                        glm::ivec3 p1_offset, p2_offset; int corner1_idx, corner2_idx, axis; glm::ivec3 edgeP1_global = globalCellCoord;
                        // Define edge endpoints (same switch as before)
                         switch (edgeIndex) { /* ... Cases 0-11 setting offsets, indices, axis, edgeP1_global ... */
                            case 0:  p1_offset={0,0,0}; p2_offset={1,0,0}; corner1_idx=0; corner2_idx=1; axis=0; edgeP1_global+=p1_offset; break;
                            case 1:  p1_offset={1,0,0}; p2_offset={1,1,0}; corner1_idx=1; corner2_idx=2; axis=1; edgeP1_global+=p1_offset; break;
                            case 2:  p1_offset={0,1,0}; p2_offset={1,1,0}; corner1_idx=3; corner2_idx=2; axis=0; edgeP1_global+=p1_offset; break;
                            case 3:  p1_offset={0,0,0}; p2_offset={0,1,0}; corner1_idx=0; corner2_idx=3; axis=1; edgeP1_global+=p1_offset; break;
                            case 4:  p1_offset={0,0,1}; p2_offset={1,0,1}; corner1_idx=4; corner2_idx=5; axis=0; edgeP1_global+=p1_offset; break;
                            case 5:  p1_offset={1,0,1}; p2_offset={1,1,1}; corner1_idx=5; corner2_idx=6; axis=1; edgeP1_global+=p1_offset; break;
                            case 6:  p1_offset={0,1,1}; p2_offset={1,1,1}; corner1_idx=7; corner2_idx=6; axis=0; edgeP1_global+=p1_offset; break;
                            case 7:  p1_offset={0,0,1}; p2_offset={0,1,1}; corner1_idx=4; corner2_idx=7; axis=1; edgeP1_global+=p1_offset; break;
                            case 8:  p1_offset={0,0,0}; p2_offset={0,0,1}; corner1_idx=0; corner2_idx=4; axis=2; edgeP1_global+=p1_offset; break;
                            case 9:  p1_offset={1,0,0}; p2_offset={1,0,1}; corner1_idx=1; corner2_idx=5; axis=2; edgeP1_global+=p1_offset; break;
                            case 10: p1_offset={1,1,0}; p2_offset={1,1,1}; corner1_idx=2; corner2_idx=6; axis=2; edgeP1_global+=p1_offset; break;
                            case 11: p1_offset={0,1,0}; p2_offset={0,1,1}; corner1_idx=3; corner2_idx=7; axis=2; edgeP1_global+=p1_offset; break;
                            default: p1_offset={0,1,0}; p2_offset={0,1,1}; corner1_idx=3; corner2_idx=7; axis=2; edgeP1_global+=p1_offset; break;
                         }
                        VolumeEdge edgeKey = {edgeP1_global, axis};

                        auto it = uniqueVertexMap.find(edgeKey);
                        if (it != uniqueVertexMap.end()) {
                            triangleLocalIndices[i] = it->second;
                        } else {
                            if (localVertices.size() >= MAX_MESHLET_VERTICES) {
                                // Only print warning once per meshlet generation
                                if (!vertexLimitHit) {
                                    // std::cerr << "Warning: CPU Meshlet vertex limit (" << MAX_MESHLET_VERTICES
                                    //           << ") reached for sub-block starting at (" << subBlockOrigin.x << "," << subBlockOrigin.y << "," << subBlockOrigin.z << ")"
                                    //           << " dim (" << subBlockDim.x << "," << subBlockDim.y << "," << subBlockDim.z << ")." << std::endl;
                                    vertexLimitHit = true;
                                }
                                triangleValid = false;
                                break;
                            }
                            glm::ivec3 p1_global = globalCellCoord + p1_offset;
                            glm::ivec3 p2_global = globalCellCoord + p2_offset;
                            CPUVertex newVert;
                            newVert.pos = calculateVertexPosCPU(p1_global, p2_global, cornerValues_uint[corner1_idx], cornerValues_uint[corner2_idx], isovalue);
                            newVert.norm = calculateVertexNormalCPU(volume, newVert.pos);
                            uint32_t newLocalIndex = static_cast<uint32_t>(localVertices.size());
                            localVertices.push_back(newVert);
                            uniqueVertexMap[edgeKey] = newLocalIndex;
                            triangleLocalIndices[i] = newLocalIndex;
                        }
                    }

                    if (triangleValid && (localIndices.size() / 3) < MAX_MESHLET_PRIMITIVES) {
                        localIndices.push_back(triangleLocalIndices[0]);
                        localIndices.push_back(triangleLocalIndices[1]);
                        localIndices.push_back(triangleLocalIndices[2]);
                    } else if (!triangleValid) {
                        break; // Stop adding triangles for this cell if vertex limit hit
                    } else {
                         // Primitive limit reached
                         // Optional: Add warning similar to vertex limit
                         goto end_sub_block_processing; // Exit loops for this sub-block
                    }
                }
            }
        }
    }
end_sub_block_processing:;

    // Finalize Meshlet if geometry was generated
    if (!localVertices.empty() && !localIndices.empty()) {
        MeshletDescriptor desc;
        desc.vertexOffset = globalVertexCounter;
        desc.indexOffset = globalIndexCounter;
        desc.vertexCount = static_cast<uint32_t>(localVertices.size());
        desc.primitiveCount = static_cast<uint32_t>(localIndices.size() / 3);

        // Append local vertices/normals to global lists
        cpuOutput.vertices.insert(cpuOutput.vertices.end(), localVertices.size(), {}); // Efficiently resize/append
        cpuOutput.normals.insert(cpuOutput.normals.end(), localVertices.size(), {});
        for (size_t i = 0; i < localVertices.size(); ++i) {
            cpuOutput.vertices[globalVertexCounter + i] = localVertices[i].pos;
            cpuOutput.normals[globalVertexCounter + i] = localVertices[i].norm;
        }

        // Append adjusted indices to global list
        cpuOutput.indices.reserve(cpuOutput.indices.size() + localIndices.size());
        for (uint32_t localIndex : localIndices) {
            cpuOutput.indices.push_back(globalVertexCounter + localIndex);
        }

        cpuOutput.meshlets.push_back(desc);
        globalVertexCounter += desc.vertexCount;
        globalIndexCounter += desc.primitiveCount * 3;
    }
}

// --- NEW: Recursive Partitioning Function ---
void processBlockRecursive(
    const glm::uvec3& currentBlockOrigin,
    const glm::uvec3& currentBlockDim,
    const Volume& volume,
    float isovalue,
    CPUExtractionOutput& cpuOutput,
    uint32_t& globalVertexCounter,
    uint32_t& globalIndexCounter)
{
    // 1. Estimate geometry for the current block size
    BlockEstimate estimate = estimateGeometryCPUForBlock(currentBlockOrigin, currentBlockDim, volume, isovalue);

    // 2. Check if estimate fits within limits
    bool exceedsLimits = (estimate.estVertexCount > MAX_MESHLET_VERTICES || estimate.estPrimCount > MAX_MESHLET_PRIMITIVES);

    // 3. Decision:
    if (exceedsLimits) {
        // Check if we can split further
        if (currentBlockDim.x > MIN_BLOCK_DIM && currentBlockDim.y > MIN_BLOCK_DIM && currentBlockDim.z > MIN_BLOCK_DIM) {
            // Split into 8 sub-blocks and recurse
            glm::uvec3 subBlockDim = currentBlockDim / 2u; // Integer division
             // std::cout << "Splitting block at (" << currentBlockOrigin.x << "," << currentBlockOrigin.y << "," << currentBlockOrigin.z << ") dim (" << currentBlockDim.x << "," << currentBlockDim.y << "," << currentBlockDim.z << ")" << std::endl;
            for (uint32_t i = 0; i < 8; ++i) {
                glm::uvec3 subBlockOffset(
                    (i & 1) * subBlockDim.x,
                    ((i >> 1) & 1) * subBlockDim.y,
                    ((i >> 2) & 1) * subBlockDim.z
                );
                glm::uvec3 subBlockOrigin = currentBlockOrigin + subBlockOffset;
                processBlockRecursive(subBlockOrigin, subBlockDim, volume, isovalue,
                                      cpuOutput, globalVertexCounter, globalIndexCounter);
            }
        } else {
            // Cannot split further, but estimate exceeded limits.
            // Generate anyway, generateMeshletForSubBlockCPU will hit the limit and warn.
             // std::cout << "Generating meshlet for smallest block (estimate exceeded) at (" << currentBlockOrigin.x << "," << currentBlockOrigin.y << "," << currentBlockOrigin.z << ")" << std::endl;
            generateMeshletForSubBlockCPU(currentBlockOrigin, currentBlockDim, volume, isovalue,
                                          cpuOutput, globalVertexCounter, globalIndexCounter);
        }
    } else {
        // Estimate fits, or block is empty. Generate meshlet for this block size (if not empty).
        if (estimate.estVertexCount > 0 || estimate.estPrimCount > 0) {
             // std::cout << "Generating meshlet for block at (" << currentBlockOrigin.x << "," << currentBlockOrigin.y << "," << currentBlockOrigin.z << ") dim (" << currentBlockDim.x << "," << currentBlockDim.y << "," << currentBlockDim.z << ")" << std::endl;
            generateMeshletForSubBlockCPU(currentBlockOrigin, currentBlockDim, volume, isovalue,
                                          cpuOutput, globalVertexCounter, globalIndexCounter);
        }
        // else: Block estimate was empty, do nothing.
    }
}


// --- Main CPU Extraction Function (Uses Recursive Partitioning) ---
CPUExtractionOutput extractMeshletsCPU(
    VulkanContext& context,
    const Volume& volume,
    FilteringOutput& filteringOutput,
    float isovalue
) {
    std::cout << "\n--- Starting CPU Meshlet Extraction Simulation (Recursive Partitioning) ---" << std::endl;
    CPUExtractionOutput cpuOutput;
    uint32_t globalVertexCounter = 0;
    uint32_t globalIndexCounter = 0;

    if (filteringOutput.activeBlockCount == 0) {
        std::cout << "CPU: No active blocks, skipping." << std::endl;
        return cpuOutput;
    }

    // 1. Read back active block IDs (same as before)
    std::vector<uint32_t> activeBlockIDs;
    try { /* ... Read back using mapUintBuffer ... */
         VkDeviceSize activeBlockDataSize = filteringOutput.activeBlockCount * sizeof(uint32_t);
         if (filteringOutput.compactedBlockIdBuffer.size < activeBlockDataSize) { throw std::runtime_error("Compacted block ID buffer is smaller than expected based on active count."); }
         activeBlockIDs = mapUintBuffer( context.getDevice(), context.getMemoryProperties(), context.getCommandPool(), context.getQueue(), filteringOutput.compactedBlockIdBuffer, activeBlockDataSize, filteringOutput.activeBlockCount );
         if (activeBlockIDs.size() != filteringOutput.activeBlockCount) { std::cerr << "Warning: Number of read back IDs differs from GPU count. Using read back count: " << activeBlockIDs.size() << std::endl; }
    } catch (const std::runtime_error& e) { std::cerr << "CPU: Error reading back active block IDs: " << e.what() << std::endl; return cpuOutput; }

    // Pre-allocation (same as before)
    size_t expectedMaxVerts = static_cast<size_t>(activeBlockIDs.size()) * MAX_MESHLET_VERTICES * 8; // Worst case 8x split
    size_t expectedMaxIndices = static_cast<size_t>(activeBlockIDs.size()) * MAX_MESHLET_PRIMITIVES * 3 * 8;
    cpuOutput.vertices.reserve(expectedMaxVerts); cpuOutput.normals.reserve(expectedMaxVerts); cpuOutput.indices.reserve(expectedMaxIndices); cpuOutput.meshlets.reserve(activeBlockIDs.size() * 8);

    // Get block grid dimensions
    glm::uvec3 blockGridDim = (volume.volume_dims + glm::uvec3(BLOCK_DIM_X_MAX, BLOCK_DIM_Y_MAX, BLOCK_DIM_Z_MAX) - 1u)
                              / glm::uvec3(BLOCK_DIM_X_MAX, BLOCK_DIM_Y_MAX, BLOCK_DIM_Z_MAX);

    // 2. Iterate through active blocks and start recursive processing
    for (uint32_t blockIndex1D : activeBlockIDs) {
        glm::uvec3 blockCoord;
        blockCoord.x = blockIndex1D % blockGridDim.x;
        blockCoord.y = (blockIndex1D / blockGridDim.x) % blockGridDim.y;
        blockCoord.z = blockIndex1D / (blockGridDim.x * blockGridDim.y);
        glm::uvec3 blockOrigin = blockCoord * glm::uvec3(BLOCK_DIM_X_MAX, BLOCK_DIM_Y_MAX, BLOCK_DIM_Z_MAX); // Initial 8x8x8 origin

        // Start recursive partitioning for this active block
        processBlockRecursive(blockOrigin, glm::uvec3(BLOCK_DIM_X_MAX, BLOCK_DIM_Y_MAX, BLOCK_DIM_Z_MAX),
                              volume, isovalue,
                              cpuOutput, globalVertexCounter, globalIndexCounter);

    } // End loop over active blocks

    std::cout << "CPU: Finished processing " << activeBlockIDs.size() << " initial active blocks." << std::endl;
    std::cout << "CPU: Generated " << cpuOutput.meshlets.size() << " meshlets." << std::endl;
    std::cout << "CPU: Total Vertices: " << cpuOutput.vertices.size() << std::endl;
    std::cout << "CPU: Total Indices: " << cpuOutput.indices.size() << std::endl;

    return cpuOutput;
}

// Function to compare GPU results against CPU simulation
// Returns true if basic checks pass, false otherwise.
bool compareExtractionOutputs(
    VulkanContext& context,
    const ExtractionOutput& gpuOutput,      // Results from GPU Extraction
    const CPUExtractionOutput& cpuOutput) // Results from CPU Simulation
{
    std::cout << "\n--- Comparing GPU vs CPU Extraction Results ---" << std::endl;
    bool overallMatch = true;

    // --- 1. Read back GPU Counts ---
    uint32_t gpuVertexCount = 0;
    uint32_t gpuIndexCount = 0;
    uint32_t gpuMeshletCount = 0; // Number of descriptors written

    try {
        gpuVertexCount = mapCounterBuffer(context, gpuOutput.vertexBuffer);
        gpuIndexCount = mapCounterBuffer(context, gpuOutput.indexBuffer);
        gpuMeshletCount = mapCounterBuffer(context, gpuOutput.meshletDescriptorBuffer);

        std::cout << "GPU Readback Counts:" << std::endl;
        std::cout << "  - Vertex Count:    " << gpuVertexCount << std::endl;
        std::cout << "  - Index Count:     " << gpuIndexCount << std::endl;
        std::cout << "  - Meshlet Count:   " << gpuMeshletCount << std::endl; // Descriptors

    } catch (const std::runtime_error& e) {
        std::cerr << "Error reading back GPU counters: " << e.what() << std::endl;
        return false; // Cannot proceed without counts
    }

    // --- 2. Compare Counts ---
    std::cout << "\nCPU Generated Counts:" << std::endl;
    std::cout << "  - Vertex Count:    " << cpuOutput.vertices.size() << std::endl;
    std::cout << "  - Index Count:     " << cpuOutput.indices.size() << std::endl;
    std::cout << "  - Meshlet Count:   " << cpuOutput.meshlets.size() << std::endl;

    if (gpuVertexCount != cpuOutput.vertices.size()) {
        std::cerr << "MISMATCH: Vertex counts differ (GPU: " << gpuVertexCount << ", CPU: " << cpuOutput.vertices.size() << ")" << std::endl;
        overallMatch = false;
    }
    if (gpuIndexCount != cpuOutput.indices.size()) {
        std::cerr << "MISMATCH: Index counts differ (GPU: " << gpuIndexCount << ", CPU: " << cpuOutput.indices.size() << ")" << std::endl;
        overallMatch = false;
    }
     if (gpuMeshletCount != cpuOutput.meshlets.size()) {
        std::cerr << "MISMATCH: Meshlet descriptor counts differ (GPU: " << gpuMeshletCount << ", CPU: " << cpuOutput.meshlets.size() << ")" << std::endl;
        overallMatch = false;
    }

    if (!overallMatch) {
         std::cerr << "Count mismatch detected. Further comparison might be unreliable." << std::endl;
         // return false; // Option to exit early
    } else {
         std::cout << "Counts match!" << std::endl;
    }


    // --- 3. Read back GPU Descriptors and Compare ---
    std::vector<MeshletDescriptor> gpuMeshlets;
    try {
        gpuMeshlets = mapMeshletDescriptorBuffer(context, gpuOutput.meshletDescriptorBuffer, gpuMeshletCount);
        std::cout << "Read back " << gpuMeshlets.size() << " meshlet descriptors from GPU." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error reading back GPU meshlet descriptors: " << e.what() << std::endl;
        return false; // Cannot compare descriptors
    }

    uint32_t compareCount = std::min((uint32_t)cpuOutput.meshlets.size(), gpuMeshletCount);
    uint32_t descriptorMismatches = 0;
    std::cout << "\nComparing Meshlet Descriptors (up to " << compareCount << "):" << std::endl;
    for (uint32_t i = 0; i < compareCount; ++i) {
        const auto& cpuDesc = cpuOutput.meshlets[i];
        const auto& gpuDesc = gpuMeshlets[i]; // Direct comparison assumes order matches for now
        bool mismatch = false;

        if (cpuDesc.vertexCount != gpuDesc.vertexCount) { std::cerr << "  Desc[" << i << "] vertexCount mismatch (CPU=" << cpuDesc.vertexCount << ", GPU=" << gpuDesc.vertexCount << ")\n"; mismatch = true; }
        if (cpuDesc.primitiveCount != gpuDesc.primitiveCount) { std::cerr << "  Desc[" << i << "] primitiveCount mismatch (CPU=" << cpuDesc.primitiveCount << ", GPU=" << gpuDesc.primitiveCount << ")\n"; mismatch = true; }
        // Offsets *might* differ if allocation order differs slightly, but should be consistent if counts match
        if (cpuDesc.vertexOffset != gpuDesc.vertexOffset) { std::cerr << "  Desc[" << i << "] vertexOffset mismatch (CPU=" << cpuDesc.vertexOffset << ", GPU=" << gpuDesc.vertexOffset << ")\n"; mismatch = true; }
        if (cpuDesc.indexOffset != gpuDesc.indexOffset) { std::cerr << "  Desc[" << i << "] indexOffset mismatch (CPU=" << cpuDesc.indexOffset << ", GPU=" << gpuDesc.indexOffset << ")\n"; mismatch = true; }
        if (gpuDesc.vertexCount != 0) { std::cerr << " Non-zero vertex count descriptor " << gpuDesc.vertexCount << "\n";}
        if (mismatch) {
            descriptorMismatches++;
            overallMatch = false;
            if (descriptorMismatches > 20) { // Limit output
                 std::cerr << "  ... further descriptor mismatches suppressed." << std::endl;
                 break;
            }
        }
    }

    if (descriptorMismatches == 0 && compareCount > 0) {
         std::cout << "Meshlet descriptors match!" << std::endl;
    } else if (compareCount > 0) {
         std::cerr << descriptorMismatches << " Meshlet descriptor mismatches found." << std::endl;
    }


    // --- 4. Compare Vertices/Indices (Basic Check - Order Dependent!) ---
    // WARNING: This basic comparison assumes vertex/index order generated by CPU/GPU
    //          is IDENTICAL, which is often NOT the case due to parallelism.
    //          A more robust check would compare geometry meshlet by meshlet,
    //          potentially using spatial hashing or sorting, which is complex.
    //          Only perform this if descriptor comparison passes.

    if (overallMatch && gpuVertexCount > 0 && gpuIndexCount > 0) {
         std::cout << "\nAttempting basic Vertex/Index comparison (order-dependent, use with caution):" << std::endl;
         bool geometryMatch = true;
        try {
            std::vector<glm::vec3> gpuVertices = mapVec3Buffer(context, gpuOutput.vertexBuffer, gpuVertexCount);
            std::vector<uint32_t> gpuIndices = mapUintBuffer(
                context.getDevice(),
                context.getMemoryProperties(),
                context.getCommandPool(),
                context.getQueue(),
                gpuOutput.indexBuffer,
                gpuIndexCount * sizeof(uint32_t),
                gpuIndexCount
                ); // Read indices after counter

            // Compare Vertices (using epsilon)
            float epsilon = 1e-4f; // Tolerance for float comparison
            uint32_t vertexMismatches = 0;
            for (size_t i = 0; i < gpuVertexCount; ++i) {
                if (glm::distance(gpuVertices[i], cpuOutput.vertices[i]) > epsilon) {
                    vertexMismatches++;
                     if (vertexMismatches <= 10) std::cerr << "  Vertex[" << i << "] mismatch (GPU=" << glm::to_string(gpuVertices[i]) << ", CPU=" << glm::to_string(cpuOutput.vertices[i]) << ")\n";
                }
            }
             if (vertexMismatches > 10) std::cerr << "  ... further vertex mismatches suppressed.\n";
             if (vertexMismatches > 0) { std::cerr << vertexMismatches << " vertex mismatches found.\n"; geometryMatch = false;}
             else {std::cout << "Vertices appear to match (order-dependent check).\n"; }

             // Compare Indices (direct comparison)
            uint32_t indexMismatches = 0;
             for (size_t i = 0; i < gpuIndexCount; ++i) {
                  if (gpuIndices[i] != cpuOutput.indices[i]) {
                      indexMismatches++;
                       if (indexMismatches <= 10) std::cerr << "  Index[" << i << "] mismatch (GPU=" << gpuIndices[i] << ", CPU=" << cpuOutput.indices[i] << ")\n";
                  }
             }
             if (indexMismatches > 10) std::cerr << "  ... further index mismatches suppressed.\n";
             if (indexMismatches > 0) { std::cerr << indexMismatches << " index mismatches found.\n"; geometryMatch = false;}
             else {std::cout << "Indices appear to match (order-dependent check).\n"; }


        } catch (const std::runtime_error& e) {
             std::cerr << "Error during GPU geometry readback: " << e.what() << std::endl;
             geometryMatch = false;
        }
         overallMatch = overallMatch && geometryMatch; // Update overall match status
    } else if (overallMatch) {
        std::cout << "\nSkipping vertex/index comparison due to zero counts or prior mismatches." << std::endl;
    }


    std::cout << "\n--- Comparison Summary ---" << std::endl;
    if (overallMatch) {
        std::cout << "SUCCESS: CPU and GPU outputs appear to match based on performed checks." << std::endl;
    } else {
        std::cout << "FAILURE: Mismatches found between CPU and GPU outputs." << std::endl;
    }

    return overallMatch;
}

// --- UPDATED FUNCTION IMPLEMENTATION for GPU OBJ Export (Vertices & Faces) ---
bool writeGPUExtractionToOBJPrev(
    VulkanContext& context,
    const ExtractionOutput& gpuOutput, // Contains GPU buffer handles
    const std::string& filename
) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    std::cout << "Reading GPU extraction results for OBJ export to " << filename << "..." << std::endl;

    // 1. Read back actual counts from GPU buffers
    // NOTE: These counts from atomics might reflect ALLOCATED space, not necessarily FILLED space
    // if the mesh shader doesn't use atomics for its own counting.
    // It's better to rely on descriptor counts later if possible.
    uint32_t counterVertexCount = 0;
    uint32_t counterIndexCount = 0;
    uint32_t counterMeshletCount = 0; // Number of descriptors Task Shader allocated space for

    try {
        if(gpuOutput.vertexBuffer.buffer != VK_NULL_HANDLE)
            counterVertexCount = mapCounterBuffer(context, gpuOutput.vertexCountBuffer);
        if(gpuOutput.indexBuffer.buffer != VK_NULL_HANDLE)
            counterIndexCount = mapCounterBuffer(context, gpuOutput.indexCountBuffer);
        if(gpuOutput.meshletDescriptorBuffer.buffer != VK_NULL_HANDLE)
            counterMeshletCount = mapCounterBuffer(context, gpuOutput.meshletDescriptorCountBuffer);

        std::cout << "GPU Atomic Counter Readback:" << std::endl;
        std::cout << "  - Vertex Counter:    " << counterVertexCount << std::endl;
        std::cout << "  - Index Counter:     " << counterIndexCount << std::endl;
        std::cout << "  - Meshlet Counter:   " << counterMeshletCount << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error reading back GPU counters for OBJ export: " << e.what() << std::endl;
        outFile.close();
        return false;
    }

    // Use the meshlet counter to determine how many descriptors to read.
    uint32_t descriptorsToRead = counterMeshletCount;
    if (descriptorsToRead == 0) {
         std::cout << "No meshlets allocated by Task Shader. OBJ file will be empty." << std::endl;
         outFile << "# OBJ file generated by Meshtrex GPU extraction\n# Meshlets Allocated: 0\n";
         outFile.close(); return true;
    }

    // 2. Read back Meshlet Descriptors
    std::vector<MeshletDescriptor> gpuMeshletDescriptors;
    try {
        gpuMeshletDescriptors = mapMeshletDescriptorBuffer(context, gpuOutput.meshletDescriptorBuffer, descriptorsToRead);
        std::cout << "Read back " << gpuMeshletDescriptors.size() << " meshlet descriptors from GPU." << std::endl;
        if (gpuMeshletDescriptors.size() < descriptorsToRead) {
             std::cerr << "Warning: Read back fewer descriptors than indicated by counter." << std::endl;
             descriptorsToRead = static_cast<uint32_t>(gpuMeshletDescriptors.size()); // Adjust count
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error reading back GPU meshlet descriptors: " << e.what() << std::endl;
        outFile.close(); return false;
    }

    // 3. Calculate ACTUAL total vertex and index counts from the VALID descriptors
    uint32_t actualTotalVertexCount = 0;
    uint32_t actualTotalIndexCount = 0;
    uint32_t validMeshletCount = 0;
    uint32_t max_vertex_offset_needed = 0; // Track highest vertex index used
    uint32_t max_index_offset_needed = 0;  // Track highest index buffer position used

    for (const auto& desc : gpuMeshletDescriptors) {
        // Use the counts written by the Mesh Shader into the descriptor
        if (desc.vertexCount > 0 && desc.primitiveCount > 0) {
            actualTotalVertexCount += desc.vertexCount; // Sum actual vertices used by meshlets
            actualTotalIndexCount += desc.primitiveCount * 3; // Sum actual indices used by meshlets
            validMeshletCount++;
            // Track the maximum extent reached in the buffers based on descriptors
            max_vertex_offset_needed = std::max(max_vertex_offset_needed, desc.vertexOffset + desc.vertexCount);
            max_index_offset_needed = std::max(max_index_offset_needed, desc.indexOffset + desc.primitiveCount * 3);
        }
    }

    std::cout << "GPU Actual Geometry Counts (Summed from Valid Descriptors):" << std::endl;
    std::cout << "  - Vertex Count:    " << actualTotalVertexCount << std::endl;
    std::cout << "  - Index Count:     " << actualTotalIndexCount << std::endl;
    std::cout << "  - Valid Meshlets:  " << validMeshletCount << std::endl;
    std::cout << "  - Max Vertex Offset Needed: " << max_vertex_offset_needed << std::endl;
    std::cout << "  - Max Index Offset Needed:  " << max_index_offset_needed << std::endl;


    if (max_vertex_offset_needed == 0 || max_index_offset_needed == 0) {
        std::cout << "No valid geometry generated by GPU Mesh Shaders (according to descriptors). OBJ file will be empty or partial." << std::endl;
        // Write header and exit if desired
    }

    // 4. Read back vertex position data (up to the max offset needed)
    std::vector<glm::vec3> gpuVertexPositions;
    if (max_vertex_offset_needed > 0) {
        try {
            std::cout << "Reading up to vertex index " << (max_vertex_offset_needed - 1) << " from VertexBuffer." << std::endl;
            gpuVertexPositions = mapVec3Buffer(context, gpuOutput.vertexBuffer, max_vertex_offset_needed); // Read up to the max offset needed
            std::cout << "Read back " << gpuVertexPositions.size() << " vertex positions from GPU." << std::endl;
             if (gpuVertexPositions.size() < max_vertex_offset_needed) {
                 std::cerr << "Warning: Read back fewer vertices than expected based on descriptor offsets. Clamping OBJ indices." << std::endl;
                 // Adjust max count based on actual readback for safety in OBJ writing
                 max_vertex_offset_needed = static_cast<uint32_t>(gpuVertexPositions.size());
             }
        } catch (const std::runtime_error& e) {
            std::cerr << "Error reading back GPU vertex positions for OBJ export: " << e.what() << std::endl;
            outFile.close(); return false;
        }
    }

    // 5. Read back ALL index data (up to the max offset needed)
    std::vector<uint32_t> gpuGlobalIndices;
    if (max_index_offset_needed > 0) {
        try {
             std::cout << "Reading up to index " << (max_index_offset_needed - 1) << " from IndexBuffer." << std::endl;
             gpuGlobalIndices = mapUintBuffer(
                 context.getDevice(), context.getMemoryProperties(), context.getCommandPool(), context.getQueue(),
                 gpuOutput.indexBuffer,
                 max_index_offset_needed * sizeof(uint32_t), // Read up to max needed size
                 max_index_offset_needed                     // Number of index elements to read
             );
             std::cout << "Read back " << gpuGlobalIndices.size() << " total indices from GPU." << std::endl;
             if (gpuGlobalIndices.size() < max_index_offset_needed) {
                  std::cerr << "Warning: Read back fewer indices than expected based on descriptor offsets. Clamping face generation." << std::endl;
                  max_index_offset_needed = static_cast<uint32_t>(gpuGlobalIndices.size());
             }
        } catch (const std::runtime_error& e) {
            std::cerr << "Error reading back GPU index data for OBJ export: " << e.what() << std::endl;
            outFile.close(); return false;
        }
    }


    // --- Write to OBJ File ---
    outFile << std::fixed << std::setprecision(6);
    outFile << "# OBJ file generated by Meshtrex GPU extraction" << std::endl;
    outFile << "# Actual Vertices Written (Sum from Descriptors): " << actualTotalVertexCount << std::endl;
    outFile << "# Actual Indices Written (Sum from Descriptors): " << actualTotalIndexCount << std::endl;
    outFile << "# Normals: Not explicitly exported from global GPU buffers in this version." << std::endl;
    outFile << "# Meshlets with Geometry: " << validMeshletCount << " (Total Dispatched: " << descriptorsToRead << ")" << std::endl;

    // Write Vertices read from buffer
    if (!gpuVertexPositions.empty()) {
        outFile << "\n# Vertex Positions (v x y z)" << std::endl;
        // Write only up to the maximum index referenced by valid descriptors
        for (size_t i = 0; i < max_vertex_offset_needed; ++i) {
            const auto& v_pos = gpuVertexPositions[i];
            outFile << "v " << v_pos.x << " " << v_pos.y << " " << v_pos.z << std::endl;
        }
    }

    // Normals are not written

    // --- Write Faces using GLOBAL indices read from index buffer ---
    if (!gpuGlobalIndices.empty() && !gpuMeshletDescriptors.empty()) {
        outFile << "\n# Faces (f v1 v2 v3)" << std::endl;
        outFile << "g gpu_extracted_mesh" << std::endl;

        // Iterate through the *descriptors* to process faces meshlet by meshlet
        for (const auto& meshletDesc : gpuMeshletDescriptors) {
            // Skip meshlets that the Mesh Shader reported as having no geometry
            if (meshletDesc.primitiveCount == 0 || meshletDesc.vertexCount == 0) continue;

            // Iterate through the primitives (triangles) of this meshlet
            for (uint32_t prim = 0; prim < meshletDesc.primitiveCount; ++prim) {
                uint32_t index_buffer_base = meshletDesc.indexOffset + prim * 3;

                // Check if indices needed are within the bounds of the *read* index buffer
                if (index_buffer_base + 2 >= gpuGlobalIndices.size()) {
                    std::cerr << "Warning: Attempting to read indices out of bounds of read data for meshlet (desc offset "
                              << meshletDesc.indexOffset << ", prim " << prim << "). Max read index: "
                              << gpuGlobalIndices.size() - 1 << std::endl;
                    break; // Stop processing primitives for this meshlet
                }

                // Read the GLOBAL vertex indices written by the Mesh Shader
                uint32_t v1_global_idx = gpuGlobalIndices[index_buffer_base + 0];
                uint32_t v2_global_idx = gpuGlobalIndices[index_buffer_base + 1];
                uint32_t v3_global_idx = gpuGlobalIndices[index_buffer_base + 2];

                // Add 1 for OBJ's 1-based indexing
                uint32_t v1_global_obj = v1_global_idx + 1;
                uint32_t v2_global_obj = v2_global_idx + 1;
                uint32_t v3_global_obj = v3_global_idx + 1;

                // Check if indices are within the bounds of the vertices we actually read back
                // Use max_vertex_offset_needed as the upper bound (exclusive, count) for 0-based indices
                if (v1_global_idx >= max_vertex_offset_needed || v2_global_idx >= max_vertex_offset_needed || v3_global_idx >= max_vertex_offset_needed) {
                     std::cerr << "Warning: Face " << (index_buffer_base/3) << " uses out-of-bounds vertex index (max vertex index: "
                               << (max_vertex_offset_needed - 1) << "): "
                               << v1_global_idx << ", " << v2_global_idx << ", " << v3_global_idx << std::endl;
                     continue; // Skip writing this face
                }

                // Write faces without normals
                outFile << "f " << v1_global_obj << " " << v2_global_obj << " " << v3_global_obj << std::endl;
            }
        }
    }

    outFile.close();
    if (outFile.good()) {
        std::cout << "Successfully wrote GPU mesh (vertices and faces) to " << filename << std::endl;
        return true;
    } else {
        std::cerr << "Error writing GPU mesh to file " << filename << std::endl;
        return false;
    }
}

// Helper function to read a single uint32_t counter from a buffer
uint32_t readCounterFromBuffer(VulkanContext& context, const Buffer& counterBuffer) {
    if (counterBuffer.buffer == VK_NULL_HANDLE || counterBuffer.size < sizeof(uint32_t)) {
        std::cerr << "Warning: Invalid or too small counter buffer provided." << std::endl;
        return 0;
    }

    VkDevice device = context.getDevice();
    Buffer readbackBuffer = {};
    VkDeviceSize counterDataSize = sizeof(uint32_t);

    createBuffer(readbackBuffer, device, context.getMemoryProperties(),
                 counterDataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (readbackBuffer.buffer == VK_NULL_HANDLE || readbackBuffer.data == nullptr) {
        destroyBuffer(readbackBuffer, device); // Cleanup if partially created
        throw std::runtime_error("Failed to create or map readback buffer for counter.");
    }

    VkCommandBuffer cmd = beginSingleTimeCommands(device, context.getCommandPool());

    // Barrier: Ensure shader writes to the counter are finished before copy
    // Assuming the counter was last written by a TASK_SHADER_BIT_EXT
    VkBufferMemoryBarrier2 counterReadBarrier = bufferBarrier(
        counterBuffer.buffer,
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, // Source: Task shader atomic write
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,                  // Destination: Transfer read
        0, counterDataSize);
    pipelineBarrier(cmd, {}, 1, &counterReadBarrier, 0, {});

    VkBufferCopy region = {0, 0, counterDataSize};
    vkCmdCopyBuffer(cmd, counterBuffer.buffer, readbackBuffer.buffer, 1, &region);

    // Barrier: Ensure copy to readback buffer is complete before host access
    // (This is implicitly handled by endSingleTimeCommands waiting on the queue)

    endSingleTimeCommands(device, context.getCommandPool(), context.getQueue(), cmd);
    // vkDeviceWaitIdle(device); // Already done by endSingleTimeCommands

    uint32_t count = 0;
    memcpy(&count, readbackBuffer.data, sizeof(uint32_t));

    destroyBuffer(readbackBuffer, device);
    return count;
}

// Generic helper function to read an array of data from a GPU buffer
template <typename T>
std::vector<T> readDataBuffer(VulkanContext& context, const Buffer& dataBuffer, uint32_t elementCount) {
    if (elementCount == 0) {
        return {};
    }
    if (dataBuffer.buffer == VK_NULL_HANDLE) {
        std::cerr << "Warning: Invalid data buffer provided for readback." << std::endl;
        return {};
    }

    VkDevice device = context.getDevice();
    VkDeviceSize elementSize = sizeof(T);
    VkDeviceSize totalDataSize = elementCount * elementSize;

    if (dataBuffer.size < totalDataSize) {
         std::cerr << "Warning: Data buffer is smaller (" << dataBuffer.size
                   << " bytes) than expected based on element count (" << elementCount
                   << " elements * " << elementSize << " bytes/element = " << totalDataSize << " bytes)." << std::endl;
        // Optionally, adjust elementCount or throw error
        // For now, proceed with caution or return empty
        return {};
    }


    Buffer readbackBuffer = {};
    createBuffer(readbackBuffer, device, context.getMemoryProperties(),
                 totalDataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (readbackBuffer.buffer == VK_NULL_HANDLE || readbackBuffer.data == nullptr) {
        destroyBuffer(readbackBuffer, device);
        throw std::runtime_error("Failed to create or map readback buffer for data array.");
    }

    VkCommandBuffer cmd = beginSingleTimeCommands(device, context.getCommandPool());

    // Barrier: Ensure shader writes to the data buffer are finished before copy
    // Assuming data was last written by a MESH_SHADER_BIT_EXT
    VkBufferMemoryBarrier2 dataReadBarrier = bufferBarrier(
        dataBuffer.buffer,
        VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, // Source: Mesh shader write
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,                   // Destination: Transfer read
        0, totalDataSize); // Offset 0, as data starts from beginning of this buffer
    pipelineBarrier(cmd, {}, 1, &dataReadBarrier, 0, {});


    VkBufferCopy region = {0, 0, totalDataSize}; // Copy from offset 0
    vkCmdCopyBuffer(cmd, dataBuffer.buffer, readbackBuffer.buffer, 1, &region);

    endSingleTimeCommands(device, context.getCommandPool(), context.getQueue(), cmd);

    std::vector<T> hostData(elementCount);
    memcpy(hostData.data(), readbackBuffer.data, totalDataSize);

    destroyBuffer(readbackBuffer, device);
    return hostData;
}


void writeGPUExtractionToOBJ(
    VulkanContext& context,
    ExtractionOutput& extractionResult, // Pass by non-const ref if you intend to populate counts here
    const char* filePath)
{
    std::cout << "Attempting to write GPU extraction output to OBJ: " << filePath << std::endl;

    // 1. Read counts from their respective buffers
    // These counts are the ACTUAL number of elements written by the shaders.
    extractionResult.vertexCount = readCounterFromBuffer(context, extractionResult.vertexCountBuffer);
    extractionResult.indexCount = readCounterFromBuffer(context, extractionResult.indexCountBuffer);
    extractionResult.meshletCount = readCounterFromBuffer(context, extractionResult.meshletDescriptorCountBuffer);

    std::cout << "  Read from GPU: Vertices = " << extractionResult.vertexCount
              << ", Indices = " << extractionResult.indexCount
              << ", Meshlets = " << extractionResult.meshletCount << std::endl;

    if (extractionResult.vertexCount == 0 || extractionResult.indexCount == 0) {
        std::cout << "  No vertices or indices to write. OBJ file will be empty or minimal." << std::endl;
        // Create an empty file or just return
        std::ofstream outFile(filePath);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open OBJ file for writing: " << filePath << std::endl;
        } // outFile will close on scope exit
        return;
    }

    // 2. Read actual data from buffers
    std::vector<VertexData> vertices = readDataBuffer<VertexData>(context, extractionResult.vertexBuffer, extractionResult.vertexCount);
    std::vector<uint32_t> indices = readDataBuffer<uint32_t>(context, extractionResult.indexBuffer, extractionResult.indexCount);
    std::vector<MeshletDescriptor> meshlets; // Only needed if you want to structure OBJ by meshlets (e.g., with 'g' tags)
                                           // For a simple flat OBJ, we just need vertices and global indices.
                                           // If you want to verify meshlet descriptors, read them here:
    // meshlets = readDataBuffer<MeshletDescriptor>(context, extractionResult.meshletDescriptorBuffer, extractionResult.meshletCount);

    std::vector<uint32_t> correctIndices;
    if (vertices.empty() || indices.empty()) {
        std::cout << "  Failed to read vertex or index data from GPU. OBJ will be incomplete." << std::endl;
        return;
    }

    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open OBJ file for writing: " << filePath << std::endl;
        return;
    }

    // Set precision for floating point numbers
    outFile << std::fixed << std::setprecision(6);

    outFile << "# OBJ file generated from GPU extraction" << std::endl;
    outFile << "# Vertices: " << extractionResult.vertexCount << std::endl;
    outFile << "# Triangles: " << extractionResult.indexCount / 3 << std::endl;
    outFile << "# Meshlets: " << extractionResult.meshletCount << std::endl;

    // 3. Write Vertices (v) and Vertex Normals (vn)
    // Since VertexData pairs position and normal, their indices will align.
    size_t nonZero = 0;
    for (size_t i = 0; i < vertices.size(); ++i) {
        // Assuming VertexData.position and .normal are glm::vec4, use .xyz
        if (vertices[i].position.x != 0 && vertices[i].position.y != 0 && vertices[i].position.z != 0) {
            outFile << "v " << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
            nonZero++;
        }
        // outFile << "v " << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
    }
    for (size_t i = 0; i <= nonZero; i = i+3 ) {
        // Assuming VertexData.position and .normal are glm::vec4, use .xyz
        outFile << " f " << i + 1 << " " << i + 2 << " " << i + 3 << std::endl;

        // outFile << "v " << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
    }

    // 4. Write Faces (f)
    // The `indices` buffer contains GLOBAL vertex indices.
    // OBJ uses 1-based indexing.
    if (extractionResult.indexCount % 3 != 0) {
        std::cerr << "Warning: Index count (" << extractionResult.indexCount << ") is not a multiple of 3. Face data might be incorrect." << std::endl;
    }

    for (size_t i = 0; i < indices.size(); i += 3) {
        // OBJ indices are 1-based
        uint32_t v1_idx = indices[i+0] + 1;
        uint32_t v2_idx = indices[i+1] + 1;
        uint32_t v3_idx = indices[i+2] + 1;

        if (
            v1_idx != 1 && v2_idx != 1 && v3_idx != 1 &&
            vertices[v1_idx].position.x != 0 &&
            vertices[v1_idx].position.y != 0 &&
            vertices[v1_idx].position.z != 0 &&
            vertices[v2_idx].position.x != 0 &&
            vertices[v2_idx].position.y != 0 &&
            vertices[v2_idx].position.z != 0 &&
            vertices[v3_idx].position.x != 0 &&
            vertices[v3_idx].position.y != 0 &&
            vertices[v3_idx].position.z != 0
            )
            {
            // Format: f v1//vn1 v2//vn2 v3//vn3
            // Since vertex and normal data are paired, their indices are the same.
            // outFile << " f " << v1_idx << " " << v2_idx << " " << v3_idx << std::endl;
        }
    }

    outFile.close();
    std::cout << "Successfully wrote data to " << filePath << std::endl;
}


/////////////////////////////////////
// Generic helper function to read a specific chunk of data from a GPU buffer
template <typename T>
std::vector<T> readDataChunkFromBuffer(VulkanContext& context, const Buffer& dataBuffer, VkDeviceSize offset, uint32_t elementCount) {
    if (elementCount == 0) {
        return {};
    }
    if (dataBuffer.buffer == VK_NULL_HANDLE) {
        std::cerr << "Warning: Invalid data buffer provided for readDataChunkFromBuffer." << std::endl;
        return {};
    }

    VkDevice device = context.getDevice();
    VkDeviceSize elementSize = sizeof(T);
    VkDeviceSize dataChunkSize = elementCount * elementSize;

    if (dataBuffer.size < (offset + dataChunkSize)) {
         std::cerr << "ERROR in readDataChunkFromBuffer: Read request (offset " << offset << ", size " << dataChunkSize
                   << ") exceeds buffer actual size (" << dataBuffer.size << " bytes)." << std::endl;
        return {}; // Critical error
    }

    Buffer readbackBuffer = {};
    createBuffer(readbackBuffer, device, context.getMemoryProperties(),
                 dataChunkSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (readbackBuffer.buffer == VK_NULL_HANDLE || readbackBuffer.data == nullptr) {
        destroyBuffer(readbackBuffer, device);
        throw std::runtime_error("Failed to create or map readback buffer for data chunk.");
    }

    VkCommandBuffer cmd = beginSingleTimeCommands(device, context.getCommandPool());

    VkBufferMemoryBarrier2 dataReadBarrier = bufferBarrier(
        dataBuffer.buffer,
        VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        offset, dataChunkSize); // Use the provided offset and chunk size
    pipelineBarrier(cmd, {}, 1, &dataReadBarrier, 0, {});

    VkBufferCopy region = {offset, 0, dataChunkSize}; // Copy from the specified offset
    vkCmdCopyBuffer(cmd, dataBuffer.buffer, readbackBuffer.buffer, 1, &region);

    endSingleTimeCommands(device, context.getCommandPool(), context.getQueue(), cmd);

    std::vector<T> hostData(elementCount);
    memcpy(hostData.data(), readbackBuffer.data, dataChunkSize);

    destroyBuffer(readbackBuffer, device);
    return hostData;
}


void writeExtractionOutputToOBJ_Revised(
    VulkanContext& context,
    ExtractionOutput& extractionResult,
    const char* filePath)
{
    std::cout << "Attempting to write GPU extraction output to OBJ (Revised): " << filePath << std::endl;

    // 1. Read the number of *actually written* meshlet descriptors
    uint32_t actualMeshletCount = readCounterFromBuffer(context, extractionResult.meshletDescriptorCountBuffer);
    extractionResult.meshletCount = actualMeshletCount;

    std::cout << "  Actual Meshlet Descriptors generated by GPU: " << actualMeshletCount << std::endl;

    if (actualMeshletCount == 0) {
        std::cout << "  No meshlets generated. OBJ file will be empty." << std::endl;
        std::ofstream outFile(filePath);
        return;
    }

    // 2. Read ALL meshlet descriptors (even if some are empty)
    std::vector<MeshletDescriptor> meshletDescriptors =
        readDataChunkFromBuffer<MeshletDescriptor>(context, extractionResult.meshletDescriptorBuffer, 0, actualMeshletCount);

    if (meshletDescriptors.empty() && actualMeshletCount > 0) {
        std::cerr << "  Failed to read meshlet descriptors from GPU. Aborting OBJ write." << std::endl;
        return;
    }

    // --- CPU-side lists for compact geometry ---
    std::vector<VertexData> compactVertices;
    std::vector<glm::uvec3> compactFaces; // Store triplets of new, compact indices
    std::map<uint32_t, uint32_t> globalToLocalVertexIndexMap; // Map global GPU index to new compact CPU index
    uint32_t nextCompactVertexIndex = 0;

    uint32_t totalNonEmptyMeshlets = 0;
    uint32_t totalActualVerticesFromDesc = 0;
    uint32_t totalActualTrianglesFromDesc = 0;

    // 3. Iterate through each meshlet descriptor
    for (const auto& desc : meshletDescriptors) {
        if (desc.vertexCount == 0 || desc.primitiveCount == 0) {
            continue; // Skip empty meshlets
        }
        totalNonEmptyMeshlets++;
        totalActualVerticesFromDesc += desc.vertexCount;
        totalActualTrianglesFromDesc += desc.primitiveCount;

        // 4. Read this meshlet's vertex data
        std::vector<VertexData> meshletVertices =
            readDataChunkFromBuffer<VertexData>(context, extractionResult.vertexBuffer,
                                                desc.vertexOffset * sizeof(VertexData), // Byte offset
                                                desc.vertexCount);

        // 5. Read this meshlet's index data
        std::vector<uint32_t> meshletIndices =
            readDataChunkFromBuffer<uint32_t>(context, extractionResult.indexBuffer,
                                               desc.indexOffset * sizeof(uint32_t), // Byte offset
                                               desc.primitiveCount * 3);

        if (meshletVertices.empty() || meshletIndices.empty()) {
            std::cerr << "  Warning: Failed to read vertex or index data for a non-empty meshlet (VtxOffset: "
                      << desc.vertexOffset << ", IdxOffset: " << desc.indexOffset
                      << ", VtxCount: " << desc.vertexCount << ", PrimCount: " << desc.primitiveCount
                      << "). Skipping this meshlet." << std::endl;
            continue;
        }

        // 6. Process vertices and indices for this meshlet, re-indexing for the compact list
        std::vector<uint32_t> currentMeshletCompactedIndices;
        currentMeshletCompactedIndices.reserve(meshletIndices.size());

        for (uint32_t globalGpuIndex : meshletIndices) {
            // The indices from the GPU buffer (meshletIndices) are relative to the start of
            // the ENTIRE vertexBuffer (i.e., they are already global offsets if written correctly by shader).
            // However, the shader writes indices relative to its globalVtxBase.
            // So, globalGpuIndex here is actually the global index.

            auto it = globalToLocalVertexIndexMap.find(globalGpuIndex);
            if (it == globalToLocalVertexIndexMap.end()) {
                // Vertex not yet in our compact list, add it
                // We need to fetch the actual VertexData for this globalGpuIndex.
                // This requires reading from the correct position in the *original* meshletVertices chunk.
                // The globalGpuIndex is an offset into the *entire* vertexBuffer.
                // The meshletVertices vector is only `desc.vertexCount` long.
                // The indices in `meshletIndices` are already global offsets from the shader.
                // We need to find which vertex from `meshletVertices` this `globalGpuIndex` corresponds to.
                // This implies `globalGpuIndex` should be between `desc.vertexOffset` and `desc.vertexOffset + desc.vertexCount -1`.
                if (globalGpuIndex >= desc.vertexOffset && globalGpuIndex < (desc.vertexOffset + desc.vertexCount)) {
                    uint32_t localIndexWithinMeshletChunk = globalGpuIndex - desc.vertexOffset;
                    compactVertices.push_back(meshletVertices[localIndexWithinMeshletChunk]);
                    globalToLocalVertexIndexMap[globalGpuIndex] = nextCompactVertexIndex;
                    currentMeshletCompactedIndices.push_back(nextCompactVertexIndex);
                    nextCompactVertexIndex++;
                } else {
                    std::cerr << "  Error: Index " << globalGpuIndex << " is out of range for current meshlet's vertices (Offset: "
                              << desc.vertexOffset << ", Count: " << desc.vertexCount << "). Skipping index." << std::endl;
                    // Push a dummy to keep face structure, or handle error differently
                    currentMeshletCompactedIndices.push_back(0); // Or some error marker
                }
            } else {
                // Vertex already added, use its existing compact index
                currentMeshletCompactedIndices.push_back(it->second);
            }
        }

        // Add faces using the new compacted indices
        for (size_t i = 0; i < currentMeshletCompactedIndices.size(); i += 3) {
            if (i + 2 < currentMeshletCompactedIndices.size()) {
                compactFaces.push_back(glm::uvec3(currentMeshletCompactedIndices[i],
                                                  currentMeshletCompactedIndices[i+1],
                                                  currentMeshletCompactedIndices[i+2]));
            }
        }
    }

    std::cout << "  Processed " << totalNonEmptyMeshlets << " non-empty meshlets." << std::endl;
    std::cout << "  Total vertices from descriptors: " << totalActualVerticesFromDesc << std::endl;
    std::cout << "  Total triangles from descriptors: " << totalActualTrianglesFromDesc << std::endl;
    std::cout << "  Compact OBJ Vertices: " << compactVertices.size() << std::endl;
    std::cout << "  Compact OBJ Triangles: " << compactFaces.size() << std::endl;


    if (compactVertices.empty() || compactFaces.empty()) {
        std::cout << "  No geometry to write after processing meshlets. OBJ file will be minimal." << std::endl;
        std::ofstream outFile(filePath);
        return;
    }

    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open OBJ file for writing: " << filePath << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    outFile << "# OBJ file generated from GPU extraction (Revised)" << std::endl;
    outFile << "# Compact Vertices: " << compactVertices.size() << std::endl;
    outFile << "# Compact Triangles: " << compactFaces.size() << std::endl;
    outFile << "# Processed Meshlets: " << totalNonEmptyMeshlets << std::endl;

    for (const auto& vertex : compactVertices) {
        outFile << "v " << vertex.position.x << " " << vertex.position.y << " " << vertex.position.z << std::endl;
        outFile << "vn " << vertex.normal.x << " " << vertex.normal.y << " " << vertex.normal.z << std::endl;
    }

    for (const auto& face : compactFaces) {
        // OBJ indices are 1-based
        outFile << "f " << (face.x + 1) << "//" << (face.x + 1) << " "
                        << (face.y + 1) << "//" << (face.y + 1) << " "
                        << (face.z + 1) << "//" << (face.z + 1) << std::endl;
    }

    outFile.close();
    std::cout << "Successfully wrote compact data to " << filePath << std::endl;
}
