#pragma once

#include "filteringOutput.h"
#include "minMaxOutput.h"

class VulkanContext;
class GPUProfiler;

FilteringOutput filterActiveBlocks(VulkanContext &vulkanContext, MinMaxOutput &minMaxOutput, PushConstants& pushConstants,
                                   VkCommandBuffer externalCmd = VK_NULL_HANDLE, GPUProfiler* profiler = nullptr);

// Read back the active block count from GPU after command buffer submission
void readActiveBlockCount(VulkanContext &vulkanContext, FilteringOutput &filteringOutput);
