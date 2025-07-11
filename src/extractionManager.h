#pragma once

#include "extractionOutput.h"
#include "minMaxOutput.h"
#include "filteringOutput.h"
#include "vulkan_context.h"
#include "streamingSystem.h" // For PageCoord
#include "persistentGeometryExtraction.h" // For persistent buffers

ExtractionOutput extractMeshletDescriptors(VulkanContext& vulkanContext, MinMaxOutput& minMaxOutput, FilteringOutput& filterOutput, PushConstants& pushConstants);

// Streaming version that writes to persistent global buffers
ExtractionOutput extractStreamingMeshletDescriptors(VulkanContext& vulkanContext, 
                                                  MinMaxOutput& minMaxOutput,
                                                  FilteringOutput& filterOutput,
                                                  VkImageView volumeAtlasView,
                                                  VkDescriptorSet pageTableSet,
                                                  const PageCoord& pageCoord,
                                                  PushConstants& pushConstants,
                                                  PersistentGeometryBuffers& persistentBuffers);
