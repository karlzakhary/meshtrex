#pragma once

#include "common.h"
#include "volume.h"
#include "filteringOutput.h"
#include "extractionOutput.h" // For MeshletDescriptor struct
#include "vulkan_context.h"   // For reading back GPU buffer

#include <vector>
#include <cstdint>
#include <glm/glm.hpp> // Assuming glm is used via common.h or glmMath.h


void writeGPUExtractionToOBJ(
    VulkanContext& context,
    ExtractionOutput& extractionResult,
    const char* filePath);