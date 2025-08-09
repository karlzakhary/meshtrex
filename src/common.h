#pragma once

#include <volk.h>

#include <string>
#include <vector>
#include "glmMath.h"
#include "mc_tables.h"

#define VK_CHECK(call) \
	do \
	{ \
		VkResult result_ = call; \
		assert(result_ == VK_SUCCESS); \
	} while (0)

#define VK_CHECK_SWAPCHAIN(call) \
	do \
	{ \
		VkResult result_ = call; \
		assert(result_ == VK_SUCCESS || result_ == VK_SUBOPTIMAL_KHR || result_ == VK_ERROR_OUT_OF_DATE_KHR); \
	} while (0)

template <typename T, size_t Size>
char (*countof_helper(T (&_Array)[Size]))[Size];

#define COUNTOF(array) (sizeof(*countof_helper(array)) + 0)


struct alignas(16) PushConstants {
    glm::uvec4 volumeDim;     // Offset 0, Size 16. Shader uses .xyz
    glm::uvec4 blockDim;      // Offset 16, Size 16. Shader uses .xyz
    glm::uvec4 blockGridDim;  // Offset 32, Size 16. Shader uses .xyz
    float isovalue;           // Offset 48, Size 4.
};

struct alignas(16) DensityPushConstants {
    glm::uvec4 volumeDim;     // Offset 0, Size 16. Shader uses .xyz
    glm::uvec4 blockDim;      // Offset 16, Size 16. Shader uses .xyz
    glm::uvec4 blockGridDim;  // Offset 32, Size 16. Shader uses .xyz
    glm::vec4 voxelSize;      // Offset 48, Size 16. For proper alignment
    glm::vec4 origin;         // Offset 64, Size 16. Volume origin
    float isovalue;           // Offset 80, Size 4.
    int activeBlockCount;     // Offset 84, Size 4.
    // Density-specific offsets
    uint32_t globalVertexOffset;   // Offset 88, Size 4.
    uint32_t globalIndexOffset;    // Offset 92, Size 4.
    uint32_t globalMeshletOffset;  // Offset 96, Size 4.
    uint32_t densityClass;         // Offset 100, Size 4. 0=sparse, 1=medium, 2=dense
    uint32_t blockOffset;          // Offset 104, Size 4. Offset into activeBlockIDs
    uint32_t _padding;             // Offset 108, Size 4. Total size = 112 (multiple of 16)
};

struct SceneUniforms { // Data changing per frame (usually)
    glm::mat4 viewProjectionMatrix;
    glm::mat4 modelMatrix; // Or pass via push constants if it changes per object
    glm::vec3 cameraPos_world;
    // Add other frequently changing params if any
};

struct LightingUniforms { // Data changing less frequently
    glm::vec4 lightPosition_world; // Using vec4 for alignment (w=1 for point, 0 for directional)
    glm::vec4 lightColor;        // Using vec4 for alignment
    float ambientIntensity;
    // Add padding if needed for std140/std430 alignment
    // Add other light properties, material properties etc.
};

inline std::string getFullPath(const char *base, const char *path)
{
    std::string fullPath = base;
    std::string::size_type pos = fullPath.find_last_of("/\\");
    if (pos == std::string::npos)
        fullPath = "";
    // else
    //     fullPath = fullPath.substr(0, pos + 1);
    fullPath += path;

    return fullPath;
}