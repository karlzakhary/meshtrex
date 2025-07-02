#include "filteringManager.h"
#include "common.h"

#include "vulkan_context.h"
#include "volume.h"
#include "resources.h"
#include "buffer.h"
#include "image.h"
#include "vulkan_utils.h"
#include "minMaxPass.h"
#include "minMaxOutput.h"
#include "blockFilteringTestUtils.h"
#include "activeBlockFilteringPass.h"
#include <cstring>
#include <iostream>
#include <string>

// --- Modified Main Orchestrating Function ---
// Returns a struct containing handles to persistent resources
MinMaxOutput computeMinMaxMip(VulkanContext &context, Volume volume, PushConstants& pushConstants);