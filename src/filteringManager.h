#pragma once

#include "filteringOutput.h"
#include "minMaxOutput.h"
#include "streamingSystem.h" // For PageCoord

class VulkanContext;

FilteringOutput filterActiveBlocks(VulkanContext &vulkanContext, MinMaxOutput &minMaxOutput, PushConstants& pushConstants);

// Streaming version that works with sparse atlas
FilteringOutput filterStreamingActiveBlocks(VulkanContext &vulkanContext, 
                                          MinMaxOutput &minMaxOutput,
                                          const Buffer& pageTableBuffer,
                                          const PageCoord& pageCoord,
                                          PushConstants& pushConstants);
