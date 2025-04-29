#pragma once

#include "common.h"
#include "volume.h"
#include "filteringOutput.h"
#include "extractionOutput.h" // For MeshletDescriptor struct
#include "vulkan_context.h"   // For reading back GPU buffer

#include <vector>
#include <cstdint>
#include <glm/glm.hpp> // Assuming glm is used via common.h or glmMath.h

// Structure to hold the CPU-generated geometry and meshlets
struct CPUExtractionOutput {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<uint32_t> indices; // Global indices referencing vertices vector
    std::vector<MeshletDescriptor> meshlets;
};

// Main function to mimic Task/Mesh extraction on the CPU
CPUExtractionOutput extractMeshletsCPU(
    VulkanContext& context,           // Needed to read back active block buffer
    const Volume& volume,             // Raw volume data access
    FilteringOutput& filteringOutput, // Contains active block count & buffer handle
    float isovalue
);