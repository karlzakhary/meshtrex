#pragma once

#include "filteringOutput.h"
#include "volume.h"

class VulkanContext;

FilteringOutput filterActiveBlocks(VulkanContext &vulkanContext, Volume volume, PushConstants& pushConstants);
