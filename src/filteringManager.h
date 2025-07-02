#pragma once

#include "filteringOutput.h"
#include "minMaxOutput.h"

class VulkanContext;

FilteringOutput filterActiveBlocks(VulkanContext &vulkanContext, MinMaxOutput &minMaxOutput, PushConstants& pushConstants);
