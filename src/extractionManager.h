#pragma once

#include "extractionOutput.h"
#include "minMaxOutput.h"
#include "filteringOutput.h"
#include "vulkan_context.h"

class GPUProfiler;

ExtractionOutput extractMeshletDescriptors(VulkanContext& vulkanContext, MinMaxOutput& minMaxOutput, FilteringOutput& filterOutput, PushConstants& pushConstants,
                                          VkCommandBuffer externalCmd = VK_NULL_HANDLE, GPUProfiler* profiler = nullptr);
