#pragma once

#include "common.h"
#include "extractionOutput.h"
#include "renderingPipeline.h"

// Updates the descriptor sets for the main rendering pipeline
void updateRenderingDescriptors(
    VkDevice device,
    RenderingPipeline& renderingPipelineState,
    uint32_t currentFrame,
    const Buffer& sceneUboBuffer,
    const ExtractionOutput& extractionOutput
);

// Records the commands needed to render the extracted meshlets
void recordRenderingCommands(
    VkCommandBuffer commandBuffer,
    RenderingPipeline& renderingPipelineState,
    uint32_t currentFrame,
    const ExtractionOutput& extractionOutput,
    VkExtent2D swapchainExtent,
    VkImageView colorAttachmentView,
    VkImageView depthAttachmentView
);