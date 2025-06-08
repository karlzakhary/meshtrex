#pragma once

#include "extractionOutput.h"
#include "filteringOutput.h"
#include "vulkan_context.h"

ExtractionOutput extractMeshletDescriptors(VulkanContext &vulkanContext, FilteringOutput &filterOutput, PushConstants& pushConstants);
