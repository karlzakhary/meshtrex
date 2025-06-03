#include "extractionTestUtils.h"
#include "blockFilteringTestUtils.h" // For mapUintBuffer
#include "mc_tables.h"               // For MarchingCubes::triTable
#include "meshoptimizer.h"
#include "vulkan_utils.h"
#include "extractionOutput.h"  // Add this at the top if not already present

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
const int BLOCK_DIM_X_MAX = 4; // Max dimensions the process starts with
const int BLOCK_DIM_Y_MAX = 4;
const int BLOCK_DIM_Z_MAX = 4;
const int MIN_BLOCK_DIM = 2; // Smallest block size to recurse down to
const int MAX_MESHLET_VERTICES = 256;
const int MAX_MESHLET_PRIMITIVES = 256;

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

// Function to perform welding and write to OBJ
void writeWeldedOutputToOBJ(
    const std::vector<VertexData>& rawVertices,
    const std::vector<glm::uvec3>& rawTriangles, // Triangles with indices into rawVertices
    const char* filePath,
    size_t totalProcessedMeshlets // For OBJ comments
) {
    if (rawVertices.empty() || rawTriangles.empty()) {
        std::cout << "  No raw geometry to weld. Creating empty OBJ file." << std::endl;
        std::ofstream outFile(filePath);
        outFile << "# Empty mesh - no raw geometry provided for welding" << std::endl;
        outFile.close();
        return;
    }

    std::cout << "Welding mesh using meshoptimizer..." << std::endl;
    std::cout << "  Raw vertices: " << rawVertices.size() << std::endl;
    std::cout << "  Raw triangles: " << rawTriangles.size() << std::endl;

    // 1. Prepare index buffer in meshoptimizer's expected format (flat array of uints)
    std::vector<unsigned int> flatIndices(rawTriangles.size() * 3);
    for (size_t i = 0; i < rawTriangles.size(); ++i) {
        flatIndices[i * 3 + 0] = rawTriangles[i].x;
        flatIndices[i * 3 + 1] = rawTriangles[i].y;
        flatIndices[i * 3 + 2] = rawTriangles[i].z;
    }

    // 2. Generate vertex remap table and unique vertex list
    std::vector<unsigned int> remap(rawVertices.size()); // To store the mapping from old to new indices
    
    // meshopt_generateVertexRemap works on raw vertex data (array of floats).
    // We need to provide a pointer to the vertex data and its stride.
    // For simplicity, we'll weld based on position only here.
    // For welding with multiple attributes (pos, normal, uv), you'd typically interleave them
    // or use meshopt_generateShadowIndexMap after remapping positions.

    // Weld based on position attribute.
    // The `sizeof(VertexData)` is the stride between consecutive vertices.
    // `offsetof(VertexData, position)` is the offset of the position member within the struct.
    size_t uniqueVertexCount = meshopt_generateVertexRemap(
        remap.data(),          // Output: remap table
        NULL,                  // indices = NULL (process all vertices from 0 to vertex_count-1)
        rawVertices.size(),    // index_count = vertex_count (signals to process all vertices)
        rawVertices.data(),    // Vertex data
        rawVertices.size(),    // Number of vertices
        sizeof(VertexData)     // Stride
    );
    // Note: The default behavior of meshopt_generateVertexRemap is to compare vertices bit-wise.
    // For float positions, if you need epsilon-based welding, you would typically
    // first quantize your vertex positions before calling this, or use a more complex
    // clustering approach before generating the final remap. Meshoptimizer itself
    // doesn't do epsilon-based float comparison directly in this function.
    // However, for OBJ viewers that merge based on textual float representation,
    // bit-wise identical floats will merge.

    std::cout << "  Unique vertices after remap: " << uniqueVertexCount << std::endl;

    // 3. Create new (welded) vertex buffer
    std::vector<VertexData> weldedVertices(uniqueVertexCount);
    meshopt_remapVertexBuffer(
        weldedVertices.data(), // Destination welded vertex buffer
        rawVertices.data(),    // Source raw vertex buffer
        rawVertices.size(),    // Number of raw vertices
        sizeof(VertexData),    // Stride of VertexData
        remap.data()           // The remap table generated above
    );

    // 4. Create new (welded) index buffer
    std::vector<unsigned int> weldedFlatIndices(flatIndices.size());
    meshopt_remapIndexBuffer(
        weldedFlatIndices.data(), // Destination welded index buffer
        flatIndices.data(),       // Source raw flat index buffer
        flatIndices.size(),       // Total number of indices (num_triangles * 3)
        remap.data()              // The remap table
    );

    // Convert weldedFlatIndices back to glm::uvec3 for OBJ writing convenience
    std::vector<glm::uvec3> weldedTriangles(weldedFlatIndices.size() / 3);
    for (size_t i = 0; i < weldedTriangles.size(); ++i) {
        weldedTriangles[i] = glm::uvec3(
            weldedFlatIndices[i * 3 + 0],
            weldedFlatIndices[i * 3 + 1],
            weldedFlatIndices[i * 3 + 2]
        );
    }

    // 5. Write the welded mesh to an OBJ file
    std::cout << "Writing welded OBJ file: " << filePath << std::endl;
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open OBJ file for writing: " << filePath << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    outFile << "# OBJ file generated from GPU extraction, welded with meshoptimizer" << std::endl;
    outFile << "# Processed meshlets (GPU): " << totalProcessedMeshlets << std::endl;
    outFile << "# Raw vertices (GPU): " << rawVertices.size() << std::endl;
    outFile << "# Welded (unique) vertices: " << weldedVertices.size() << std::endl;
    outFile << "# Triangles: " << weldedTriangles.size() << std::endl;

    // Write vertices
    for (const auto& vertex : weldedVertices) {
        outFile << "v " << vertex.position.x << " " << vertex.position.y << " " << vertex.position.z << std::endl;
    }

    // Write triangles (faces)
    for (const auto& triangle : weldedTriangles) {
        // OBJ indices are 1-based
        outFile << "f " << (triangle.x + 1) << "//" << (triangle.x + 1) << " "
                        << (triangle.y + 1) << "//" << (triangle.y + 1) << " "
                        << (triangle.z + 1) << "//" << (triangle.z + 1) << std::endl;
    }

    outFile.close();
    std::cout << "Successfully wrote welded OBJ file: " << filePath << std::endl;
}

// Improved OBJ export that works with the fixed subdivision approach
void writeExtractionOutputToOBJ_Revisedz(
    VulkanContext& context,
    ExtractionOutput& extractionResult,
    const char* filePath)
{
    std::cout << "Writing GPU extraction output to OBJ (Fixed Subdivision): " << filePath << std::endl;

    uint32_t actualMeshletCount = readCounterFromBuffer(context, extractionResult.meshletCountBuffer);
    extractionResult.meshletCount = actualMeshletCount;

    std::cout << "  Actual meshlets generated: " << actualMeshletCount << std::endl;

    if (actualMeshletCount == 0) {
        std::cout << "  No meshlets generated. Creating empty OBJ file." << std::endl;
        std::ofstream outFile(filePath);
        outFile << "# Empty mesh - no geometry generated" << std::endl;
        return;
    }

    // Read all meshlet descriptors
    std::vector<MeshletDescriptor> meshletDescriptors =
        readDataChunkFromBuffer<MeshletDescriptor>(context, extractionResult.meshletDescriptorBuffer, 0, actualMeshletCount);

    if (meshletDescriptors.empty()) {
        std::cerr << "  Failed to read meshlet descriptors. Aborting." << std::endl;
        return;
    }

    // STRATEGY: Read vertices and indices meshlet by meshlet to avoid gaps
    std::vector<VertexData> allVertices;
    std::vector<glm::uvec3> allTriangles;
    
    uint32_t vertexIndexOffset = 0; // Running offset for index mapping
    uint32_t totalVertices = 0;
    uint32_t totalTriangles = 0;
    uint32_t processedMeshlets = 0;

    std::cout << "  Processing meshlets..." << std::endl;

    for (uint32_t meshletIdx = 0; meshletIdx < meshletDescriptors.size(); meshletIdx++) {
        const auto& desc = meshletDescriptors[meshletIdx];
        
        if (desc.vertexCount == 0 || desc.primitiveCount == 0) {
            continue; // Skip empty meshlets
        }

        processedMeshlets++;

        // Read this meshlet's vertices (contiguous block)
        std::vector<VertexData> meshletVertices =
            readDataChunkFromBuffer<VertexData>(context, extractionResult.globalVertexBuffer,
                                                desc.vertexOffset * sizeof(VertexData),
                                                desc.vertexCount);

        // Read this meshlet's indices (contiguous block)
        std::vector<uint32_t> meshletIndices =
            readDataChunkFromBuffer<uint32_t>(context, extractionResult.globalIndexBuffer,
                                               desc.indexOffset * sizeof(uint32_t),
                                               desc.primitiveCount * 3);

        if (meshletVertices.size() != desc.vertexCount) {
            std::cerr << "  Warning: Meshlet " << meshletIdx << " vertex count mismatch. Expected: " 
                      << desc.vertexCount << ", Got: " << meshletVertices.size() << std::endl;
            continue;
        }

        if (meshletIndices.size() != desc.primitiveCount * 3) {
            std::cerr << "  Warning: Meshlet " << meshletIdx << " index count mismatch. Expected: " 
                      << (desc.primitiveCount * 3) << ", Got: " << meshletIndices.size() << std::endl;
            continue;
        }

        // Add vertices to global list
        allVertices.insert(allVertices.end(), meshletVertices.begin(), meshletVertices.end());

        // Process triangles with index remapping
        for (size_t i = 0; i < meshletIndices.size(); i += 3) {
            if (i + 2 < meshletIndices.size()) {
                uint32_t idx0 = meshletIndices[i];
                uint32_t idx1 = meshletIndices[i + 1];
                uint32_t idx2 = meshletIndices[i + 2];

                // CRITICAL: Indices should be relative to global vertex buffer
                // They should be in range [desc.vertexOffset, desc.vertexOffset + desc.vertexCount)
                if (idx0 >= desc.vertexOffset && idx0 < desc.vertexOffset + desc.vertexCount &&
                    idx1 >= desc.vertexOffset && idx1 < desc.vertexOffset + desc.vertexCount &&
                    idx2 >= desc.vertexOffset && idx2 < desc.vertexOffset + desc.vertexCount) {
                    
                    // Convert global indices to local indices for this OBJ export
                    uint32_t localIdx0 = (idx0 - desc.vertexOffset) + vertexIndexOffset;
                    uint32_t localIdx1 = (idx1 - desc.vertexOffset) + vertexIndexOffset;
                    uint32_t localIdx2 = (idx2 - desc.vertexOffset) + vertexIndexOffset;
                    
                    allTriangles.push_back(glm::uvec3(localIdx0, localIdx1, localIdx2));
                } else {
                    std::cerr << "  Error in meshlet " << meshletIdx << ": Indices (" 
                              << idx0 << "," << idx1 << "," << idx2 
                              << ") outside valid range [" << desc.vertexOffset 
                              << "," << (desc.vertexOffset + desc.vertexCount - 1) << "]" << std::endl;
                    
                    // Try to salvage by clamping indices
                    uint32_t clampedIdx0 = std::min(std::max(idx0, desc.vertexOffset), desc.vertexOffset + desc.vertexCount - 1);
                    uint32_t clampedIdx1 = std::min(std::max(idx1, desc.vertexOffset), desc.vertexOffset + desc.vertexCount - 1);
                    uint32_t clampedIdx2 = std::min(std::max(idx2, desc.vertexOffset), desc.vertexOffset + desc.vertexCount - 1);
                    
                    uint32_t localIdx0 = (clampedIdx0 - desc.vertexOffset) + vertexIndexOffset;
                    uint32_t localIdx1 = (clampedIdx1 - desc.vertexOffset) + vertexIndexOffset;
                    uint32_t localIdx2 = (clampedIdx2 - desc.vertexOffset) + vertexIndexOffset;
                    
                    allTriangles.push_back(glm::uvec3(localIdx0, localIdx1, localIdx2));
                }
            }
        }

        // Update running offset for next meshlet
        vertexIndexOffset += desc.vertexCount;
        totalVertices += desc.vertexCount;
        totalTriangles += desc.primitiveCount;
    }
    // writeWeldedOutputToOBJ(allVertices, allTriangles, getFullPath(ROOT_BUILD_PATH,"/welded_output.obj").c_str(), processedMeshlets);
    // return;
    std::cout << "  Processing complete:" << std::endl;
    std::cout << "    Processed meshlets: " << processedMeshlets << "/" << actualMeshletCount << std::endl;
    std::cout << "    Total vertices: " << totalVertices << " (collected: " << allVertices.size() << ")" << std::endl;
    std::cout << "    Total triangles: " << totalTriangles << " (collected: " << allTriangles.size() << ")" << std::endl;

    if (allVertices.empty() || allTriangles.empty()) {
        std::cout << "  No valid geometry collected. Creating minimal OBJ file." << std::endl;
        std::ofstream outFile(filePath);
        outFile << "# No valid geometry found" << std::endl;
        return;
    }

    // Write OBJ file
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open OBJ file for writing: " << filePath << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    outFile << "# OBJ file generated from GPU extraction (Fixed Subdivision)" << std::endl;
    outFile << "# Total vertices: " << allVertices.size() << std::endl;
    outFile << "# Total triangles: " << allTriangles.size() << std::endl;
    outFile << "# Processed meshlets: " << processedMeshlets << std::endl;

    // Write vertices and normals
    for (const auto& vertex : allVertices) {
        outFile << "v " << vertex.position.x << " " << vertex.position.y << " " << vertex.position.z << std::endl;
    }

    // Write triangles
    for (const auto& triangle : allTriangles) {
        // OBJ indices are 1-based
        outFile << "f " << (triangle.x + 1) << "//" << (triangle.x + 1) << " "
                        << (triangle.y + 1) << "//" << (triangle.y + 1) << " "
                        << (triangle.z + 1) << "//" << (triangle.z + 1) << std::endl;
    }

    outFile.close();
    std::cout << "Successfully wrote OBJ file: " << filePath << std::endl;
}

void writeGPUMeshToOBJ(VulkanContext& context, ExtractionOutput& extractionOutput, const char* filename) {
    // Read meshlet count from GPU
    uint32_t meshletCount = readCounterFromBuffer(context, extractionOutput.meshletCountBuffer);
    
    if (meshletCount == 0) {
        std::cerr << "No meshlets to write to OBJ file" << std::endl;
        return;
    }

    // Read meshlet descriptors
    std::vector<MeshletDescriptor> descriptors = readDataBuffer<MeshletDescriptor>(
        context,
        extractionOutput.meshletDescriptorBuffer,
        meshletCount
    );

    // Open OBJ file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write header
    outFile << "# Marching Cubes Mesh\n";
    outFile << "# Meshlets: " << meshletCount << "\n\n";

    uint32_t totalVertices = 0;
    uint32_t totalTriangles = 0;
    uint32_t vertexIndexOffset = 0;  // For remapping vertex indices
    
    // First pass: Count total vertices and triangles
    for (const auto& desc : descriptors) {
        totalVertices += desc.vertexCount;
        totalTriangles += desc.primitiveCount;
    }

    outFile << "# Total Vertices: " << totalVertices << "\n";
    outFile << "# Total Triangles: " << totalTriangles << "\n\n";

    // Process each meshlet
    for (size_t meshletIdx = 0; meshletIdx < descriptors.size(); meshletIdx++) {
        const auto& desc = descriptors[meshletIdx];
        
        if (desc.vertexCount == 0 || desc.primitiveCount == 0) continue;

        // // Debug output
        // std::cout << "Meshlet " << meshletIdx << ": "
        //           << "vertexOffset=" << desc.vertexOffset 
        //           << ", vertexCount=" << desc.vertexCount
        //           << ", indexOffset=" << desc.primitiveOffset  // Note: this is actually index offset
        //           << ", primitiveCount=" << desc.primitiveCount << std::endl;

        // Read vertices for this meshlet
        // vertexOffset is in vertices, need to convert to bytes
        std::vector<VertexData> meshletVertices = readDataChunkFromBuffer<VertexData>(
            context,
            extractionOutput.globalVertexBuffer,
            desc.vertexOffset * sizeof(VertexData),  // Convert vertex offset to bytes
            desc.vertexCount
        );

        // Write vertices
        for (const auto& vertex : meshletVertices) {
            outFile << "v " << vertex.position.x << " " << vertex.position.y << " " << vertex.position.z << "\n";
        }

        // Read indices for this meshlet
        // primitiveOffset is already in indices (not primitives!), just convert to bytes
        std::vector<uint32_t> meshletIndices = readDataChunkFromBuffer<uint32_t>(
            context,
            extractionOutput.globalIndexBuffer,
            desc.indexOffset * sizeof(uint32_t),  // primitiveOffset is actually index offset
            desc.primitiveCount * 3  // Read 3 indices per primitive
        );

        // Write faces with adjusted indices (OBJ uses 1-based indexing)
        for (size_t i = 0; i < meshletIndices.size(); i += 3) {
            // The indices in the buffer are global, need to make them relative to this meshlet
            uint32_t idx0 = meshletIndices[i] - desc.vertexOffset + vertexIndexOffset + 1;
            uint32_t idx1 = meshletIndices[i + 1] - desc.vertexOffset + vertexIndexOffset + 1;
            uint32_t idx2 = meshletIndices[i + 2] - desc.vertexOffset + vertexIndexOffset + 1;
            
            outFile << "f "
                << idx0 << "//" << idx0 << " "
                << idx1 << "//" << idx1 << " "
                << idx2 << "//" << idx2 << "\n";
        }

        vertexIndexOffset += desc.vertexCount;
    }

    outFile.close();
    std::cout << "Mesh written to " << filename << std::endl;
    std::cout << "Meshlets: " << meshletCount << ", Total Vertices: " << totalVertices << ", Total Triangles: " << totalTriangles << std::endl;
}

void writeExtractionOutputToOBJ_Revised_pmb(
    VulkanContext& context,
    ExtractionOutput& extractionResult,
    const char* filePath)
{
    std::cout << "Writing GPU extraction output to OBJ (Corrected PMB): " << filePath << std::endl;

    // Read actual meshlet count from GPU
    uint32_t actualMeshletCount = readCounterFromBuffer(context, extractionResult.meshletCountBuffer);
    extractionResult.meshletCount = actualMeshletCount;

    std::cout << "  Actual meshlets generated: " << actualMeshletCount << std::endl;

    if (actualMeshletCount == 0) {
        std::cout << "  No meshlets generated. Creating empty OBJ file." << std::endl;
        std::ofstream outFile(filePath);
        outFile << "# Empty mesh - no geometry generated" << std::endl;
        return;
    }

    // Read all meshlet descriptors
    std::vector<MeshletDescriptor> meshletDescriptors =
        readDataChunkFromBuffer<MeshletDescriptor>(context, extractionResult.meshletDescriptorBuffer, 0, actualMeshletCount);

    if (meshletDescriptors.empty()) {
        std::cerr << "  Failed to read meshlet descriptors. Aborting." << std::endl;
        return;
    }

    // For inter-block deduplication, we'll need to track unique vertices
    std::unordered_map<std::string, uint32_t> vertexMap;  // vertex key -> final index
    std::vector<VertexData> uniqueVertices;
    std::vector<glm::uvec3> allTriangles;
    
    uint32_t totalVertices = 0;
    uint32_t totalTriangles = 0;
    uint32_t processedMeshlets = 0;
    uint32_t duplicateVertices = 0;

    std::cout << "  Processing meshlets with deduplication..." << std::endl;

    // First pass: Read all vertices and build unique vertex list
    for (uint32_t meshletIdx = 0; meshletIdx < meshletDescriptors.size(); meshletIdx++) {
        const auto& desc = meshletDescriptors[meshletIdx];
        
        if (desc.vertexCount == 0 || desc.primitiveCount == 0) {
            continue; // Skip empty meshlets
        }

        processedMeshlets++;

        // Read this meshlet's vertices
        std::vector<VertexData> meshletVertices =
            readDataChunkFromBuffer<VertexData>(context, extractionResult.globalVertexBuffer,
                                                desc.vertexOffset * sizeof(VertexData),
                                                desc.vertexCount);

        if (meshletVertices.size() != desc.vertexCount) {
            std::cerr << "  Warning: Meshlet " << meshletIdx << " vertex count mismatch. Expected: " 
                      << desc.vertexCount << ", Got: " << meshletVertices.size() << std::endl;
            continue;
        }

        // Process vertices for deduplication
        for (const auto& vertex : meshletVertices) {
            // Create a key for the vertex (position with some precision)
            char keyBuffer[128];
            snprintf(keyBuffer, sizeof(keyBuffer), "%.6f,%.6f,%.6f", 
                     vertex.position.x, vertex.position.y, vertex.position.z);
            std::string key(keyBuffer);

            auto it = vertexMap.find(key);
            if (it == vertexMap.end()) {
                // New unique vertex
                uint32_t newIndex = static_cast<uint32_t>(uniqueVertices.size());
                vertexMap[key] = newIndex;
                uniqueVertices.push_back(vertex);
            } else {
                duplicateVertices++;
            }
        }
        
        totalVertices += desc.vertexCount;
    }

    std::cout << "  Vertex deduplication: " << totalVertices << " total, " 
              << uniqueVertices.size() << " unique, " 
              << duplicateVertices << " duplicates removed" << std::endl;

    // Second pass: Read indices and remap to unique vertices
    for (uint32_t meshletIdx = 0; meshletIdx < meshletDescriptors.size(); meshletIdx++) {
        const auto& desc = meshletDescriptors[meshletIdx];
        
        if (desc.vertexCount == 0 || desc.primitiveCount == 0) {
            continue;
        }

        // Read vertices again for mapping
        std::vector<VertexData> meshletVertices =
            readDataChunkFromBuffer<VertexData>(context, extractionResult.globalVertexBuffer,
                                                desc.vertexOffset * sizeof(VertexData),
                                                desc.vertexCount);

        // Create local to global vertex mapping
        std::vector<uint32_t> localToGlobal(desc.vertexCount);
        for (uint32_t i = 0; i < desc.vertexCount; i++) {
            char keyBuffer[128];
            snprintf(keyBuffer, sizeof(keyBuffer), "%.6f,%.6f,%.6f", 
                     meshletVertices[i].position.x, 
                     meshletVertices[i].position.y, 
                     meshletVertices[i].position.z);
            std::string key(keyBuffer);
            localToGlobal[i] = vertexMap[key];
        }

        // Read indices - note that indexOffset is already in indices, not primitives
        std::vector<uint32_t> meshletIndices =
            readDataChunkFromBuffer<uint32_t>(context, extractionResult.globalIndexBuffer,
                                               desc.indexOffset * sizeof(uint32_t),
                                               desc.primitiveCount * 3);

        if (meshletIndices.size() != desc.primitiveCount * 3) {
            std::cerr << "  Warning: Meshlet " << meshletIdx << " index count mismatch. Expected: " 
                      << (desc.primitiveCount * 3) << ", Got: " << meshletIndices.size() << std::endl;
            continue;
        }

        // Process triangles with remapping
        for (size_t i = 0; i < meshletIndices.size(); i += 3) {
            if (i + 2 < meshletIndices.size()) {
                uint32_t idx0 = meshletIndices[i];
                uint32_t idx1 = meshletIndices[i + 1];
                uint32_t idx2 = meshletIndices[i + 2];

                // Indices should already be global, but we need to make them relative to this meshlet
                uint32_t localIdx0 = idx0 - desc.vertexOffset;
                uint32_t localIdx1 = idx1 - desc.vertexOffset;
                uint32_t localIdx2 = idx2 - desc.vertexOffset;

                // Validate local indices
                if (localIdx0 < desc.vertexCount && 
                    localIdx1 < desc.vertexCount && 
                    localIdx2 < desc.vertexCount) {
                    
                    // Map to deduplicated global indices
                    uint32_t globalIdx0 = localToGlobal[localIdx0];
                    uint32_t globalIdx1 = localToGlobal[localIdx1];
                    uint32_t globalIdx2 = localToGlobal[localIdx2];
                    
                    allTriangles.push_back(glm::uvec3(globalIdx0, globalIdx1, globalIdx2));
                } else {
                    std::cerr << "  Error in meshlet " << meshletIdx << ": Local indices (" 
                              << localIdx0 << "," << localIdx1 << "," << localIdx2 
                              << ") outside valid range [0," << (desc.vertexCount - 1) << "]" << std::endl;
                }
            }
        }

        totalTriangles += desc.primitiveCount;
    }

    std::cout << "  Processing complete:" << std::endl;
    std::cout << "    Processed meshlets: " << processedMeshlets << "/" << actualMeshletCount << std::endl;
    std::cout << "    Unique vertices: " << uniqueVertices.size() << std::endl;
    std::cout << "    Total triangles: " << totalTriangles << " (collected: " << allTriangles.size() << ")" << std::endl;

    if (uniqueVertices.empty() || allTriangles.empty()) {
        std::cout << "  No valid geometry collected. Creating minimal OBJ file." << std::endl;
        std::ofstream outFile(filePath);
        outFile << "# No valid geometry found" << std::endl;
        return;
    }

    // Write OBJ file
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open OBJ file for writing: " << filePath << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    outFile << "# OBJ file generated from GPU extraction (PMB Marching Cubes)" << std::endl;
    outFile << "# Total vertices: " << uniqueVertices.size() << " (deduplicated)" << std::endl;
    outFile << "# Total triangles: " << allTriangles.size() << std::endl;
    outFile << "# Processed meshlets: " << processedMeshlets << std::endl;

    // Write vertices
    for (const auto& vertex : uniqueVertices) {
        outFile << "v " << vertex.position.x << " " << vertex.position.y << " " << vertex.position.z << std::endl;
    }

    // Write triangles
    for (const auto& triangle : allTriangles) {
        // OBJ indices are 1-based
        outFile << "f " << (triangle.x + 1) << "//" << (triangle.x + 1) << " "
                        << (triangle.y + 1) << "//" << (triangle.y + 1) << " "
                        << (triangle.z + 1) << "//" << (triangle.z + 1) << std::endl;
    }

    outFile.close();
    std::cout << "Successfully wrote OBJ file: " << filePath << std::endl;
}

void writeExtractionOutputToOBJ_Revised(
    VulkanContext& context,
    ExtractionOutput& extractionResult,
    const char* filePath)
{
    std::cout << "Attempting to write GPU extraction output to OBJ (Revised): " << filePath << std::endl;

    // 1. Read the number of *actually written* meshlet descriptors
    uint32_t actualMeshletCount = readCounterFromBuffer(context, extractionResult.meshletCountBuffer);
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
            readDataChunkFromBuffer<VertexData>(context, extractionResult.globalVertexBuffer,
                                                desc.vertexOffset * sizeof(VertexData), // Byte offset
                                                desc.vertexCount);

        // 5. Read this meshlet's index data
        std::vector<uint32_t> meshletIndices =
            readDataChunkFromBuffer<uint32_t>(context, extractionResult.globalIndexBuffer,
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

    // for (const auto& vertex : compactVertices) {
    //     outFile << "v " << vertex.position.x << " " << vertex.position.y << " " << vertex.position.z << std::endl;
    //     outFile << "vn " << vertex.normal.x << " " << vertex.normal.y << " " << vertex.normal.z << std::endl;
    // }

    for (const auto& face : compactFaces) {
        // OBJ indices are 1-based
        outFile << "f " << (face.x + 1) << "//" << (face.x + 1) << " "
                        << (face.y + 1) << "//" << (face.y + 1) << " "
                        << (face.z + 1) << "//" << (face.z + 1) << std::endl;
    }

    outFile.close();
    std::cout << "Successfully wrote compact data to " << filePath << std::endl;
}

void writeExtractionOutputToOBJ_Revisedd(
    VulkanContext& context,
    ExtractionOutput& extractionResult,
    const char* filePath) {
    try {
        // Step 1: Read meshlet count
        uint32_t meshletCount = readCounterFromBuffer(context, extractionResult.meshletCountBuffer);
        
        if (meshletCount == 0) {
            std::cerr << "Warning: No meshlets generated" << std::endl;
            return;
        }
        
        std::cout << "Extracting mesh from " << meshletCount << " meshlets" << std::endl;
        
        // Step 2: Read all meshlet descriptors
        std::vector<MeshletDescriptor> meshlets = 
            readDataBuffer<MeshletDescriptor>(context, extractionResult.meshletDescriptorBuffer, meshletCount);
        
        // Step 3: Calculate total unique vertices and indices needed
        uint32_t maxVertexEnd = 0;
        uint32_t maxIndexEnd = 0;
        uint32_t totalPrimitives = 0;
        
        for (const auto& meshlet : meshlets) {
            maxVertexEnd = std::max(maxVertexEnd, meshlet.vertexOffset + meshlet.vertexCount);
            maxIndexEnd = std::max(maxIndexEnd, meshlet.indexOffset + meshlet.primitiveCount * 3);
            totalPrimitives += meshlet.primitiveCount;
        }
        
        std::cout << "Total vertices to read: " << maxVertexEnd << std::endl;
        std::cout << "Total indices to read: " << maxIndexEnd << std::endl;
        std::cout << "Total primitives: " << totalPrimitives << std::endl;
        
        // Step 4: Read only the used vertices and indices
        std::vector<VertexData> vertices = readDataBuffer<VertexData>(context, extractionResult.globalVertexBuffer, maxVertexEnd);
        std::vector<uint32_t> indices = readDataBuffer<uint32_t>(context, extractionResult.globalIndexBuffer, maxIndexEnd);
        
        // Step 5: Write to OBJ file
        std::ofstream objFile(filePath);
        if (!objFile.is_open()) {
            throw std::runtime_error("Failed to open file: " + std::string(filePath));
        }
        
        // Write header
        objFile << "# Extracted isosurface mesh from meshlets\n";
        objFile << "# Meshlets: " << meshletCount << "\n";
        objFile << "# Total vertices: " << maxVertexEnd << "\n";
        objFile << "# Total triangles: " << totalPrimitives << "\n";
        objFile << "# Note: This mesh contains duplicate vertices at block boundaries\n\n";
        
        objFile << std::fixed << std::setprecision(6);
        
        // Step 6: Write all vertices
        for (uint32_t i = 0; i < maxVertexEnd; ++i) {
            objFile << "v " << vertices[i].position[0] << " " 
                           << vertices[i].position[1] << " " 
                           << vertices[i].position[2] << "\n";
        }
        
        objFile << "\n# Faces organized by meshlet\n";
        
        // Step 7: Write faces organized by meshlet
        for (size_t m = 0; m < meshlets.size(); ++m) {
            const auto& meshlet = meshlets[m];
            
            if (meshlet.primitiveCount == 0) continue;
            
            objFile << "# Meshlet " << m << " (vertices: " << meshlet.vertexCount 
                    << ", triangles: " << meshlet.primitiveCount << ")\n";
            
            // Each meshlet's indices are already in global vertex space
            for (uint32_t p = 0; p < meshlet.primitiveCount; ++p) {
                uint32_t baseIdx = meshlet.indexOffset + p * 3;
                
                // OBJ uses 1-based indexing
                objFile << "f " << (indices[baseIdx] + 1) << " " 
                               << (indices[baseIdx + 1] + 1) << " " 
                               << (indices[baseIdx + 2] + 1) << "\n";
            }
            objFile << "\n";
        }
        
        objFile.close();
        std::cout << "Successfully wrote mesh to " << filePath << std::endl;
        
        // Analyze vertex usage and duplicates
        // analyzeVertexUsageAndDuplicates(meshlets, vertices, indices);
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting mesh: " << e.what() << std::endl;
    }
}

// Enhanced analysis function
void analyzeVertexUsageAndDuplicates(const std::vector<MeshletDescriptor>& meshlets,
                                     const std::vector<VertexData>& vertices,
                                     const std::vector<uint32_t>& indices) {
    // Track vertex usage
    std::vector<int> vertexUsageCount(vertices.size(), 0);
    
    for (const auto& meshlet : meshlets) {
        for (uint32_t p = 0; p < meshlet.primitiveCount; ++p) {
            uint32_t baseIdx = meshlet.indexOffset + p * 3;
            for (int i = 0; i < 3; ++i) {
                if (indices[baseIdx + i] < vertices.size()) {
                    vertexUsageCount[indices[baseIdx + i]]++;
                }
            }
        }
    }
    
    // Count unused vertices
    int unusedCount = 0;
    int multiUseCount = 0;
    int maxUse = 0;
    
    for (int count : vertexUsageCount) {
        if (count == 0) unusedCount++;
        if (count > 1) multiUseCount++;
        maxUse = std::max(maxUse, count);
    }
    
    std::cout << "\nVertex usage statistics:\n"
              << "  Unused vertices: " << unusedCount << "\n"
              << "  Vertices used multiple times: " << multiUseCount << "\n"
              << "  Maximum vertex reuse: " << maxUse << std::endl;
    
    // Find duplicate vertices (same position)
    const float EPSILON = 1e-6f;
    std::map<std::tuple<int, int, int>, std::vector<uint32_t>> vertexMap;
    
    for (uint32_t i = 0; i < vertices.size(); ++i) {
        if (vertexUsageCount[i] == 0) continue; // Skip unused vertices
        
        // Quantize to grid for duplicate detection
        int qx = static_cast<int>(vertices[i].position[0] / EPSILON);
        int qy = static_cast<int>(vertices[i].position[1] / EPSILON);
        int qz = static_cast<int>(vertices[i].position[2] / EPSILON);
        
        vertexMap[{qx, qy, qz}].push_back(i);
    }
    
    int duplicateGroups = 0;
    int totalDuplicates = 0;
    float maxDuplicateDistance = 0.0f;
    
    for (const auto& [key, group] : vertexMap) {
        if (group.size() > 1) {
            duplicateGroups++;
            totalDuplicates += group.size() - 1;
            
            // Check actual distances within group
            for (size_t i = 0; i < group.size(); ++i) {
                for (size_t j = i + 1; j < group.size(); ++j) {
                    float dx = vertices[group[i]].position[0] - vertices[group[j]].position[0];
                    float dy = vertices[group[i]].position[1] - vertices[group[j]].position[1];
                    float dz = vertices[group[i]].position[2] - vertices[group[j]].position[2];
                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    maxDuplicateDistance = std::max(maxDuplicateDistance, dist);
                }
            }
        }
    }
    
    std::cout << "\nDuplicate vertex analysis:\n"
              << "  Duplicate groups: " << duplicateGroups << "\n"
              << "  Total duplicate vertices: " << totalDuplicates << "\n"
              << "  Max distance between duplicates: " << maxDuplicateDistance << std::endl;
}
