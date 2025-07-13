#pragma once

#include "common.h"
#include "renderingPipeline.h"
#include "extractionOutput.h"
#include "extractionPipeline.h"
#include "filteringOutput.h"
#include "minMaxOutput.h"

void recordRenderingCommands(
    VkCommandBuffer commandBuffer,
    VkDevice device,
    RenderingPipeline& renderingPipelineState,
    const ExtractionOutput& extractionOutput, // Input: Geometry buffers
    VkExtent2D swapchainExtent,
    VkImageView colorAttachmentView,
    VkImageView depthAttachmentView,
    VkImageLayout colorAttachmentLayout,
    VkImageLayout depthAttachmentLayout
);

void updateExtractionDescriptors(
    VkDevice device,
    ExtractionPipeline& extractionPipelineState,
    const MinMaxOutput& minMaxOutput,
    const FilteringOutput& filteringResult, // Input: Active block list/count
    const Buffer& uboIsovalue,              // Input: UBO containing current isovalue etc.
    const Buffer& marchingCubesTriTable,    // Input: MC Table
    const ExtractionOutput& extractionOutput // Output: Target Vtx/Idx/Indirect buffers
);

void recordExtractionDispatch(
    VkCommandBuffer commandBuffer,
    ExtractionPipeline& extractionPipelineState,
    const FilteringOutput& filteringResult,
    const Buffer& indirectDrawBuffer  // Add indirect draw buffer parameter
);

void updateRenderingDescriptors(
    VkDevice device,
    RenderingPipeline& renderingPipelineState,
    const Buffer& sceneUboBuffer, // UBO containing MVP, lighting etc.
    const ExtractionOutput& extractionOutput   // Contains MeshletDesc, VB, IB
    // Add texture bindings if needed
);