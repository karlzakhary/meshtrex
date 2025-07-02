#pragma once

#include "extractionOutput.h"
#include "minMaxOutput.h"
#include "filteringOutput.h"
#include "vulkan_context.h"

ExtractionOutput extractMeshletDescriptors(VulkanContext& vulkanContext, MinMaxOutput& minMaxOutput, FilteringOutput& filterOutput, PushConstants& pushConstants);
