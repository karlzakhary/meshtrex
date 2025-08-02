#pragma once

#include "minMaxOutput.h"
#include "filteringOutput.h"
#include "vulkan_context.h"
#include "volume.h"
#include <glm/glm.hpp>

class GPUProfiler;

struct TransientExtractionPushConstants {
    glm::mat4 viewProj;
    glm::vec4 frustumPlanes[6]; // Frustum planes for culling
};

// Transient extraction that renders directly without persisting geometry
void extractAndRenderTransient(
    VulkanContext& vulkanContext,
    MinMaxOutput& minMaxOutput,
    FilteringOutput& filterOutput,
    PushConstants& pushConstants,
    const TransientExtractionPushConstants& renderConstants,
    VkCommandBuffer commandBuffer,
    VkFormat colorFormat,
    VkFormat depthFormat,
    GPUProfiler* profiler = nullptr,
    bool pmb = true
);

// Extract frustum planes from view-projection matrix
void extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4 frustumPlanes[6]);

// Cleanup static resources
void cleanupTransientExtractionResources(VkDevice device);