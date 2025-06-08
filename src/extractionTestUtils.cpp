#include "extractionTestUtils.h"
#include "blockFilteringTestUtils.h" // For mapUintBuffer
#include "mc_tables.h"               // For MarchingCubes::triTable
#include "meshoptimizer.h"
#include "vulkan_utils.h"

#include <set>
#include <vector>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <map> // Using map for edge->vertex to keep order (helps comparison slightly)
#include <unordered_map> // For vertex deduplication
#define GLM_ENABLE_EXPERIMENTAL
#include <iomanip>
#include <fstream>

#include "glm/gtx/string_cast.hpp"

// --- Constants (match shader/C++ setup) ---
const int BLOCK_DIM_X_MAX = 4; // Max dimensions the process starts with
const int BLOCK_DIM_Y_MAX = 4;
const int BLOCK_DIM_Z_MAX = 4;

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

void writeGPUExtractionToOBJ(
    VulkanContext& context,
    ExtractionOutput& extractionResult,
    const char* filePath) {
    
    // Read counters from GPU to get actual counts
    uint32_t vertexCount = readCounterFromBuffer(context, extractionResult.vertexCountBuffer);
    uint32_t indexCount = readCounterFromBuffer(context, extractionResult.indexCountBuffer);
    
    std::cout << "Writing OBJ file with " << vertexCount << " vertices and " 
              << indexCount / 3 << " triangles to " << filePath << std::endl;
    
    if (vertexCount == 0 || indexCount == 0) {
        std::cerr << "Warning: No geometry to export (vertex count: " << vertexCount 
                  << ", index count: " << indexCount << ")" << std::endl;
        return;
    }
    
    // Read vertex data from GPU
    std::vector<VertexData> vertices = readDataChunkFromBuffer<VertexData>(
        context, 
        extractionResult.vertexBuffer, 
        0,
        vertexCount
    );
    
    // Read index data from GPU
    std::vector<uint32_t> indices = readDataChunkFromBuffer<uint32_t>(
        context, 
        extractionResult.indexBuffer, 
        0,
        indexCount
    );
    
    // Deduplicate vertices
    std::cout << "Deduplicating vertices..." << std::endl;
    
    // Hash function for VertexData comparison
    struct VertexHash {
        std::size_t operator()(const VertexData& vertex) const {
            // Simple hash combining position and normal
            auto h1 = std::hash<float>{}(vertex.position.x);
            auto h2 = std::hash<float>{}(vertex.position.y);
            auto h3 = std::hash<float>{}(vertex.position.z);
            auto h4 = std::hash<float>{}(vertex.normal.x);
            auto h5 = std::hash<float>{}(vertex.normal.y);
            auto h6 = std::hash<float>{}(vertex.normal.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
        }
    };
    
    // Equality function for VertexData comparison
    struct VertexEqual {
        bool operator()(const VertexData& a, const VertexData& b) const {
            const float epsilon = 1e-6f;
            return (std::abs(a.position.x - b.position.x) < epsilon &&
                    std::abs(a.position.y - b.position.y) < epsilon &&
                    std::abs(a.position.z - b.position.z) < epsilon &&
                    std::abs(a.normal.x - b.normal.x) < epsilon &&
                    std::abs(a.normal.y - b.normal.y) < epsilon &&
                    std::abs(a.normal.z - b.normal.z) < epsilon);
        }
    };
    
    // Map from original vertex to deduplicated vertex index
    std::unordered_map<VertexData, uint32_t, VertexHash, VertexEqual> vertexMap;
    std::vector<VertexData> uniqueVertices;
    std::vector<uint32_t> remappedIndices(indexCount);
    
    // Build unique vertex list and remap indices
    for (uint32_t i = 0; i < indexCount; ++i) {
        uint32_t originalIndex = indices[i];
        if (originalIndex >= vertexCount) {
            std::cerr << "Warning: Invalid vertex index " << originalIndex 
                      << " (max: " << vertexCount - 1 << ")" << std::endl;
            continue;
        }
        
        const VertexData& vertex = vertices[originalIndex];
        
        auto it = vertexMap.find(vertex);
        if (it != vertexMap.end()) {
            // Vertex already exists, use existing index
            remappedIndices[i] = it->second;
        } else {
            // New unique vertex
            uint32_t newIndex = static_cast<uint32_t>(uniqueVertices.size());
            vertexMap[vertex] = newIndex;
            uniqueVertices.push_back(vertex);
            remappedIndices[i] = newIndex;
        }
    }
    
    // Remove degenerate triangles (triangles with duplicate vertices)
    std::vector<uint32_t> validIndices;
    uint32_t degenerateCount = 0;
    
    for (uint32_t i = 0; i < indexCount; i += 3) {
        uint32_t v1 = remappedIndices[i];
        uint32_t v2 = remappedIndices[i + 1];
        uint32_t v3 = remappedIndices[i + 2];
        
        // Check for degenerate triangle (any two vertices are the same)
        if (v1 != v2 && v2 != v3 && v1 != v3) {
            validIndices.push_back(v1);
            validIndices.push_back(v2);
            validIndices.push_back(v3);
        } else {
            degenerateCount++;
        }
    }
    
    uint32_t finalVertexCount = static_cast<uint32_t>(uniqueVertices.size());
    uint32_t finalTriangleCount = static_cast<uint32_t>(validIndices.size() / 3);
    
    std::cout << "Deduplication results:" << std::endl;
    std::cout << "  Original vertices: " << vertexCount << " -> Unique vertices: " << finalVertexCount 
              << " (reduction: " << std::fixed << std::setprecision(1) 
              << (100.0f * (vertexCount - finalVertexCount) / vertexCount) << "%)" << std::endl;
    std::cout << "  Original triangles: " << indexCount / 3 << " -> Valid triangles: " << finalTriangleCount;
    if (degenerateCount > 0) {
        std::cout << " (removed " << degenerateCount << " degenerate triangles)";
    }
    std::cout << std::endl;
    
    // Open output file
    std::ofstream objFile(filePath);
    if (!objFile.is_open()) {
        throw std::runtime_error("Failed to open OBJ file for writing: " + std::string(filePath));
    }
    
    // Write header comment
    objFile << "# Generated by MeshTrex GPU Extraction (Deduplicated)\n";
    objFile << "# Original: " << vertexCount << " vertices, " << indexCount / 3 << " triangles\n";
    objFile << "# Deduplicated: " << finalVertexCount << " vertices, " << finalTriangleCount << " triangles\n\n";
    
    // Write unique vertices
    objFile << "# Vertex positions\n";
    for (uint32_t i = 0; i < finalVertexCount; ++i) {
        const glm::vec4& pos = uniqueVertices[i].position;
        objFile << "v " << pos.x << " " << pos.y << " " << pos.z << "\n";
    }
    objFile << "\n";
    
    // Write vertex normals
    objFile << "# Vertex normals\n";
    for (uint32_t i = 0; i < finalVertexCount; ++i) {
        const glm::vec4& normal = uniqueVertices[i].normal;
        objFile << "vn " << normal.x << " " << normal.y << " " << normal.z << "\n";
    }
    objFile << "\n";
    
    // Write faces (triangles)
    objFile << "# Faces\n";
    for (uint32_t i = 0; i < validIndices.size(); i += 3) {
        // OBJ uses 1-based indexing, so add 1 to each index
        uint32_t v1 = validIndices[i] + 1;
        uint32_t v2 = validIndices[i + 1] + 1;
        uint32_t v3 = validIndices[i + 2] + 1;
        
        objFile << "f " << v1 << "//" << v1 << " " << v2 << "//" << v2 << " " << v3 << "//" << v3 << "\n";
    }
    
    objFile.close();
    
    std::cout << "Successfully wrote deduplicated OBJ file: " << filePath << std::endl;
}