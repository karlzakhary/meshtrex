#include "extractionTestUtils.h"

#include <cmath>
#include <cstdint>
#include <functional>  // For std::hash
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "blockFilteringTestUtils.h"  // For mapUintBuffer
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include "mc_tables.h"  // For MarchingCubes::triTable
#include "vulkan_utils.h"  // For helper functions if needed (not strictly necessary here)

// --- Constants (match shader/C++ setup) ---
const int BLOCK_DIM_X_1 = 8;
const int BLOCK_DIM_Y_1 = 8;
const int BLOCK_DIM_Z_1 = 8;
const int MAX_MESHLET_VERTICES = 128;
const int MAX_MESHLET_PRIMITIVES = 256;

// --- Helper Structures ---

// Simple Vertex structure for CPU processing
struct CPUVertex {
    glm::vec3 pos;
    glm::vec3 norm;

    // Equality operator for potential use in maps if needed (though hashing edge is better)
    bool operator==(const CPUVertex& other) const {
        return pos == other.pos && norm == other.norm;
    }
};

// Represent an edge within the entire volume grid for hashing
struct VolumeEdge {
    glm::ivec3 p1; // Coordinate of the vertex with lower indices defining the edge
    int axis;      // 0 for X-edge, 1 for Y-edge, 2 for Z-edge

    bool operator==(const VolumeEdge& other) const {
        return p1 == other.p1 && axis == other.axis;
    }
};

// Hash function for VolumeEdge
namespace std {
    template <>
    struct hash<VolumeEdge> {
        std::size_t operator()(const VolumeEdge& e) const {
            // Combine hashes of coordinates and axis (simple example)
            size_t h1 = std::hash<int>{}(e.p1.x);
            size_t h2 = std::hash<int>{}(e.p1.y);
            size_t h3 = std::hash<int>{}(e.p1.z);
            size_t h4 = std::hash<int>{}(e.axis);
            // Combine hashes (boost::hash_combine style)
            size_t seed = 0;
            seed ^= h1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}


// --- Helper Functions ---

// Safely sample volume data (uint8) at given global coordinates
inline uint8_t sampleVolume(const Volume& volume, const glm::ivec3& coord) {
    if (coord.x < 0 || coord.x >= volume.volume_dims.x ||
        coord.y < 0 || coord.y >= volume.volume_dims.y ||
        coord.z < 0 || coord.z >= volume.volume_dims.z) {
        return 0; // Or some other boundary value
    }
    size_t index = static_cast<size_t>(coord.z) * volume.volume_dims.x * volume.volume_dims.y +
                   static_cast<size_t>(coord.y) * volume.volume_dims.x +
                   static_cast<size_t>(coord.x);
    if (index >= volume.volume_data.size()) {
         return 0; // Should not happen with bounds check, but safety
    }
    return volume.volume_data[index];
}

// Calculate vertex position via interpolation (matches shader logic)
inline glm::vec3 calculateVertexPosCPU(const glm::ivec3& p1_coord, const glm::ivec3& p2_coord,
                                  uint8_t val1_uint, uint8_t val2_uint, float isovalue) {
    float val1 = static_cast<float>(val1_uint);
    float val2 = static_cast<float>(val2_uint);
    float denom = val2 - val1;
    if (std::abs(denom) < 1e-5f) {
        return glm::vec3(p1_coord); // Avoid division by zero
    }
    float t = (isovalue - val1) / denom;
    return glm::mix(glm::vec3(p1_coord), glm::vec3(p2_coord), glm::clamp(t, 0.0f, 1.0f));
}

// Calculate vertex normal via central differences (matches shader logic)
inline glm::vec3 calculateVertexNormalCPU(const Volume& volume, const glm::vec3& pos) {
    glm::ivec3 ipos = glm::ivec3(glm::round(pos)); // Use integer coords near pos
    float dx = (static_cast<float>(sampleVolume(volume, ipos + glm::ivec3(1, 0, 0))) -
                static_cast<float>(sampleVolume(volume, ipos - glm::ivec3(1, 0, 0))));
    float dy = (static_cast<float>(sampleVolume(volume, ipos + glm::ivec3(0, 1, 0))) -
                static_cast<float>(sampleVolume(volume, ipos - glm::ivec3(0, 1, 0))));
    float dz = (static_cast<float>(sampleVolume(volume, ipos + glm::ivec3(0, 0, 1))) -
                static_cast<float>(sampleVolume(volume, ipos - glm::ivec3(0, 0, 1))));

    glm::vec3 grad(dx, dy, dz);
    if (glm::length(grad) < 1e-5f) {
        return glm::vec3(0.0f, 1.0f, 0.0f); // Default up vector
    }
    // MC normals point towards lower density (negative gradient)
    return -glm::normalize(grad);
}

// Mimics the Task Shader's estimateGeometry function
inline uint32_t estimateGeometryCPU(uint32_t mc_case) {
    uint32_t primCount = 0;
    uint32_t vertCount = 0;
    int idx = 0;
    const int* table_entry = MarchingCubes::triTable[mc_case];
    while (idx < 15 && table_entry[idx] != -1) {
        primCount++;
        idx += 3;
    }
    primCount /= 3;
    if (primCount > 0) {
       vertCount = primCount + 2; // Heuristic
    }
    return (primCount << 16) | vertCount;
}

// Mimics the Mesh Shader's logic for a single sub-block
void generateMeshletForSubBlockCPU(
    const glm::uvec3& blockOrigin, // Origin of the PARENT block
    const glm::uvec3& subBlockOffset, // Offset of this sub-block within parent
    const glm::uvec3& subBlockDim, // Dimensions of this sub-block (e.g., 8 or 4)
    const Volume& volume,
    float isovalue,
    CPUExtractionOutput& cpuOutput, // Reference to append global results
    uint32_t& globalVertexCounter,  // Current global vertex offset
    uint32_t& globalIndexCounter    // Current global index offset
) {
    // Local data for this meshlet
    std::vector<CPUVertex> localVertices;
    std::vector<uint32_t> localIndices; // Indices into localVertices
    localVertices.reserve(MAX_MESHLET_VERTICES);
    localIndices.reserve(MAX_MESHLET_PRIMITIVES * 3);

    // Map to store unique vertices generated *within this sub-block*
    // Key: Canonical representation of the edge in the volume grid
    // Value: Index into the 'localVertices' vector for this meshlet
    std::unordered_map<VolumeEdge, uint32_t> uniqueVertexMap;

    glm::uvec3 subBlockOrigin = blockOrigin + subBlockOffset;

    // Iterate through cells WITHIN this sub-block
    for (uint32_t lz = 0; lz < subBlockDim.z; ++lz) {
        for (uint32_t ly = 0; ly < subBlockDim.y; ++ly) {
            for (uint32_t lx = 0; lx < subBlockDim.x; ++lx) {
                glm::ivec3 localCellCoord(lx, ly, lz);
                glm::ivec3 globalCellCoord = glm::ivec3(subBlockOrigin) + localCellCoord;

                // Calculate MC case
                uint32_t mc_case = 0;
                uint8_t cornerValues_uint[8];
                 bool boundary = false;
                for (int i = 0; i < 8; ++i) {
                    glm::ivec3 cornerOffset( (i & 1), (i & 2) >> 1, (i & 4) >> 2 );
                    glm::ivec3 cornerCoord = globalCellCoord + cornerOffset;
                    // Check bounds carefully before sampling
                     if (cornerCoord.x >= volume.volume_dims.x ||
                         cornerCoord.y >= volume.volume_dims.y ||
                         cornerCoord.z >= volume.volume_dims.z ||
                         cornerCoord.x < 0 || cornerCoord.y < 0 || cornerCoord.z < 0) {
                          boundary = true;
                          break; // Skip cell if corner is out of bounds
                     }
                    cornerValues_uint[i] = sampleVolume(volume, cornerCoord);
                    if (static_cast<float>(cornerValues_uint[i]) >= isovalue) {
                        mc_case |= (1 << i);
                    }
                }

                if (boundary || mc_case == 0 || mc_case == 255) {
                    continue; // Skip inactive or boundary cells
                }

                // Generate triangles using triTable
                const int* table_entry = MarchingCubes::triTable[mc_case];
                for (int v = 0; v < 15; v += 3) {
                    if (table_entry[v] == -1) break; // End of list

                    uint32_t triangleLocalIndices[3];
                    bool triangleValid = true;

                    // Process 3 vertices of the triangle
                    for (int i = 0; i < 3; ++i) {
                        int edgeIndex = table_entry[v + i]; // 0-11

                        // Define edge endpoints relative to cell origin (localCellCoord)
                        // Based on standard edge numbering (same as shader)
                        glm::ivec3 p1_offset, p2_offset;
                        int corner1_idx, corner2_idx;
                        int axis; // 0=X, 1=Y, 2=Z
                        // Define the canonical lower-indexed vertex (p1) for the edge key
                        glm::ivec3 edgeP1_global = globalCellCoord;

                        switch (edgeIndex) {
                            case 0:  p1_offset = {0,0,0}; p2_offset = {1,0,0}; corner1_idx=0; corner2_idx=1; axis=0; edgeP1_global += p1_offset; break;
                            case 1:  p1_offset = {1,0,0}; p2_offset = {1,1,0}; corner1_idx=1; corner2_idx=2; axis=1; edgeP1_global += p1_offset; break;
                            case 2:  p1_offset = {0,1,0}; p2_offset = {1,1,0}; corner1_idx=3; corner2_idx=2; axis=0; edgeP1_global += p1_offset; break;
                            case 3:  p1_offset = {0,0,0}; p2_offset = {0,1,0}; corner1_idx=0; corner2_idx=3; axis=1; edgeP1_global += p1_offset; break;
                            case 4:  p1_offset = {0,0,1}; p2_offset = {1,0,1}; corner1_idx=4; corner2_idx=5; axis=0; edgeP1_global += p1_offset; break;
                            case 5:  p1_offset = {1,0,1}; p2_offset = {1,1,1}; corner1_idx=5; corner2_idx=6; axis=1; edgeP1_global += p1_offset; break;
                            case 6:  p1_offset = {0,1,1}; p2_offset = {1,1,1}; corner1_idx=7; corner2_idx=6; axis=0; edgeP1_global += p1_offset; break;
                            case 7:  p1_offset = {0,0,1}; p2_offset = {0,1,1}; corner1_idx=4; corner2_idx=7; axis=1; edgeP1_global += p1_offset; break;
                            case 8:  p1_offset = {0,0,0}; p2_offset = {0,0,1}; corner1_idx=0; corner2_idx=4; axis=2; edgeP1_global += p1_offset; break;
                            case 9:  p1_offset = {1,0,0}; p2_offset = {1,0,1}; corner1_idx=1; corner2_idx=5; axis=2; edgeP1_global += p1_offset; break;
                            case 10: p1_offset = {1,1,0}; p2_offset = {1,1,1}; corner1_idx=2; corner2_idx=6; axis=2; edgeP1_global += p1_offset; break;
                            case 11: p1_offset = {0,1,0}; p2_offset = {0,1,1}; corner1_idx=3; corner2_idx=7; axis=2; edgeP1_global += p1_offset; break;
                            default: throw std::runtime_error("Invalid edge index from triTable.");
                        }

                        VolumeEdge edgeKey = {edgeP1_global, axis};

                        // Check if vertex exists for this edge in this meshlet
                        auto it = uniqueVertexMap.find(edgeKey);
                        if (it != uniqueVertexMap.end()) {
                            // Vertex exists, use its local index
                            triangleLocalIndices[i] = it->second;
                        } else {
                            // Vertex is new for this meshlet, create it
                            if (localVertices.size() >= MAX_MESHLET_VERTICES) {
                                // Meshlet vertex limit reached
                                std::cerr << "Warning: Meshlet vertex limit (" << MAX_MESHLET_VERTICES << ") reached." << std::endl;
                                triangleValid = false;
                                break; // Stop processing this triangle
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
                    } // End loop over 3 triangle vertices

                    // If triangle is valid and doesn't exceed primitive limit, add its indices
                    if (triangleValid && (localIndices.size() / 3) < MAX_MESHLET_PRIMITIVES) {
                        localIndices.push_back(triangleLocalIndices[0]);
                        localIndices.push_back(triangleLocalIndices[1]);
                        localIndices.push_back(triangleLocalIndices[2]);
                    } else if (!triangleValid) {
                        // Stop processing triangles for this cell if vertex limit was hit
                        break;
                    } else {
                         // Primitive limit reached for this meshlet
                         std::cerr << "Warning: Meshlet primitive limit (" << MAX_MESHLET_PRIMITIVES << ") reached." << std::endl;
                         // Stop processing triangles for this cell and potentially subsequent cells
                         goto end_cell_processing; // Use goto for early exit from nested loops
                    }
                } // End loop over triangles for this case
            } // End loop over cells lx
        } // End loop ly
    } // End loop lz

end_cell_processing:; // Label for goto

    // --- Finalize Meshlet ---
    if (!localVertices.empty() && !localIndices.empty()) {
        MeshletDescriptor desc;
        desc.vertexOffset = globalVertexCounter;
        desc.indexOffset = globalIndexCounter; // This offset is into the *global* index buffer
        desc.vertexCount = static_cast<uint32_t>(localVertices.size());
        desc.primitiveCount = static_cast<uint32_t>(localIndices.size() / 3);

        // Append local vertices/normals to global lists
        cpuOutput.vertices.insert(cpuOutput.vertices.end(), localVertices.size(), glm::vec3()); // Resize first
        cpuOutput.normals.insert(cpuOutput.normals.end(), localVertices.size(), glm::vec3());
        for (size_t i = 0; i < localVertices.size(); ++i) {
            cpuOutput.vertices[globalVertexCounter + i] = localVertices[i].pos;
            cpuOutput.normals[globalVertexCounter + i] = localVertices[i].norm;
        }

        // Append local indices (adjusted to global vertex indices) to global list
        cpuOutput.indices.reserve(cpuOutput.indices.size() + localIndices.size());
        for (uint32_t localIndex : localIndices) {
            cpuOutput.indices.push_back(globalVertexCounter + localIndex); // Add base offset
        }

        // Add the descriptor
        cpuOutput.meshlets.push_back(desc);

        // Update global counters
        globalVertexCounter += desc.vertexCount;
        globalIndexCounter += desc.primitiveCount * 3;
    }
}


// --- Main CPU Extraction Function ---
CPUExtractionOutput extractMeshletsCPU(
    VulkanContext& context,
    const Volume& volume,
    FilteringOutput& filteringOutput,
    float isovalue
) {
    std::cout << "\n--- Starting CPU Meshlet Extraction Simulation ---" << std::endl;
    CPUExtractionOutput cpuOutput;
    uint32_t globalVertexCounter = 0;
    uint32_t globalIndexCounter = 0;

    if (filteringOutput.activeBlockCount == 0) {
        std::cout << "CPU: No active blocks, skipping." << std::endl;
        return cpuOutput;
    }

    // 1. Read back the active block IDs from the GPU buffer
    std::vector<uint32_t> activeBlockIDs;
    try {
        VkDeviceSize activeBlockDataSize = filteringOutput.activeBlockCount * sizeof(uint32_t);
        // Ensure buffer is large enough
        if (filteringOutput.compactedBlockIdBuffer.size < activeBlockDataSize) {
             throw std::runtime_error("Compacted block ID buffer is smaller than expected based on active count.");
        }
        activeBlockIDs = mapUintBuffer(
            context.getDevice(),
            context.getMemoryProperties(),
            context.getCommandPool(),
            context.getQueue(),
            filteringOutput.compactedBlockIdBuffer,
            activeBlockDataSize, // Read only the necessary amount
            filteringOutput.activeBlockCount
        );
        std::cout << "CPU: Read back " << activeBlockIDs.size() << " active block IDs." << std::endl;
        // It's possible mapUintBuffer adjusted the count if buffer was too small, re-check
        if (activeBlockIDs.size() != filteringOutput.activeBlockCount) {
             std::cerr << "Warning: Number of read back IDs differs from GPU count. Using read back count: " << activeBlockIDs.size() << std::endl;
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "CPU: Error reading back active block IDs: " << e.what() << std::endl;
        return cpuOutput; // Return empty on error
    }

    // Pre-allocate global vectors roughly (helps performance)
    size_t expectedMaxVerts = static_cast<size_t>(activeBlockIDs.size()) * MAX_MESHLET_VERTICES;
    size_t expectedMaxIndices = static_cast<size_t>(activeBlockIDs.size()) * MAX_MESHLET_PRIMITIVES * 3;
    cpuOutput.vertices.reserve(expectedMaxVerts);
    cpuOutput.normals.reserve(expectedMaxVerts);
    cpuOutput.indices.reserve(expectedMaxIndices);
    cpuOutput.meshlets.reserve(activeBlockIDs.size() * 2); // Account for potential splits

    // Get block grid dimensions (needed for coordinate conversion and estimates)
    glm::uvec3 blockGridDim = (volume.volume_dims + glm::uvec3(BLOCK_DIM_X_1, BLOCK_DIM_Y_1, BLOCK_DIM_Z_1) - 1u)
                              / glm::uvec3(BLOCK_DIM_X_1, BLOCK_DIM_Y_1, BLOCK_DIM_Z_1);

    // 2. Iterate through active blocks
    for (uint32_t blockIndex1D : activeBlockIDs) {
        // Convert 1D index to 3D coords
        glm::uvec3 blockCoord;
        blockCoord.x = blockIndex1D % blockGridDim.x;
        blockCoord.y = (blockIndex1D / blockGridDim.x) % blockGridDim.y;
        blockCoord.z = blockIndex1D / (blockGridDim.x * blockGridDim.y);
        glm::uvec3 blockOrigin = blockCoord * glm::uvec3(BLOCK_DIM_X_1, BLOCK_DIM_Y_1, BLOCK_DIM_Z_1);

        // 3. Mimic Task Shader Partitioning Decision
        uint32_t totalEstVertices = 0;
        uint32_t totalEstPrims = 0;
        // Estimate geometry for the whole 8x8x8 block
        for(int z=0; z<BLOCK_DIM_Z_1; ++z) {
            for(int y=0; y<BLOCK_DIM_Y_1; ++y) {
                for(int x=0; x<BLOCK_DIM_X_1; ++x) {
                    glm::ivec3 globalCellCoord = glm::ivec3(blockOrigin) + glm::ivec3(x,y,z);
                    uint32_t mc_case = 0;
                    // Calculate case (simplified, assumes corners are in bounds)
                    for(int i=0; i<8; ++i) {
                         glm::ivec3 cornerOffset( (i & 1), (i & 2) >> 1, (i & 4) >> 2 );
                         glm::ivec3 cornerCoord = globalCellCoord + cornerOffset;
                         if (static_cast<float>(sampleVolume(volume, cornerCoord)) >= isovalue) {
                             mc_case |= (1 << i);
                         }
                    }
                     if (mc_case != 0 && mc_case != 255) {
                         uint32_t counts = estimateGeometryCPU(mc_case);
                         totalEstPrims += (counts >> 16);
                         totalEstVertices += (counts & 0xFFFF);
                     }
                }
            }
        }

        bool splitBlock = (totalEstVertices > MAX_MESHLET_VERTICES || totalEstPrims > MAX_MESHLET_PRIMITIVES);

        // 4. Generate Meshlet(s) for the block/sub-blocks
        if (splitBlock) {
            // Split into 8 sub-blocks (4x4x4)
            glm::uvec3 subBlockDim(BLOCK_DIM_X_1 / 2, BLOCK_DIM_Y_1 / 2, BLOCK_DIM_Z_1 / 2);
            for (uint32_t i = 0; i < 8; ++i) {
                glm::uvec3 subBlockOffset(
                    (i & 1) * subBlockDim.x,
                    ((i >> 1) & 1) * subBlockDim.y,
                    ((i >> 2) & 1) * subBlockDim.z
                );
                generateMeshletForSubBlockCPU(blockOrigin, subBlockOffset, subBlockDim,
                                            volume, isovalue,
                                            cpuOutput, globalVertexCounter, globalIndexCounter);
            }
        } else if (totalEstVertices > 0) { // Check if block actually generated geometry estimate
            // Process the whole block as one meshlet
            generateMeshletForSubBlockCPU(blockOrigin, glm::uvec3(0), glm::uvec3(BLOCK_DIM_X_1, BLOCK_DIM_Y_1, BLOCK_DIM_Z_1),
                                        volume, isovalue,
                                        cpuOutput, globalVertexCounter, globalIndexCounter);
        }
    } // End loop over active blocks

    std::cout << "CPU: Finished processing " << activeBlockIDs.size() << " active blocks." << std::endl;
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