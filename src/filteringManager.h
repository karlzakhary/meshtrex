#pragma once

#include "filteringOutput.h"

class VulkanContext;

FilteringOutput filterActiveBlocks(VulkanContext &vulkanContext, const char* volumePath);
