#pragma once

#include "extractionOutput.h"
#include "minMaxOutput.h"
#include "filteringOutput.h"
#include "vulkan_context.h"
#include "volume.h"

class GPUProfiler;

// Original extraction function
ExtractionOutput extractMeshletDescriptors(VulkanContext& vulkanContext, MinMaxOutput& minMaxOutput, FilteringOutput& filterOutput, PushConstants& pushConstants,
                                          VkCommandBuffer externalCmd = VK_NULL_HANDLE, GPUProfiler* profiler = nullptr, bool pmb = false);

// New extraction function with density-based dispatch support
ExtractionOutput extractMeshletDescriptorsWithDensity(VulkanContext& vulkanContext, MinMaxOutput& minMaxOutput, FilteringOutput& filterOutput, PushConstants& pushConstants,
                                                      const Volume& volume, bool useDensityDispatch,
                                                      VkCommandBuffer externalCmd = VK_NULL_HANDLE, GPUProfiler* profiler = nullptr, bool pmb = false);
