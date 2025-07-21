#pragma once

#include "common.h"
#include "image.h"
#include "buffer.h"
#include "resources.h" // Defines Image and Buffer structs
#include "temporaryResources.h"
#include <cstdint>

// Structure to hold the results and persistent resources from the filtering process
struct FilteringOutput {
    Buffer compactedBlockIdBuffer{}; // Buffer containing IDs of active blocks
    Buffer activeBlockCountBuffer{}; // Buffer containing the count of active blocks (atomic counter)
    // Buffer indirectDispatchBuffer{}; // Buffer for indirect mesh task dispatch (VkDrawMeshTasksIndirectCommandEXT)
    uint32_t activeBlockCount = 0; // The actual count read back from the buffer (0 when using GPU-driven)
    
    // Temporary resources that need cleanup after command buffer submission
    // Only populated when using external command buffer
    TemporaryResources tempResources;

    // Add handles needed for destruction by the caller
    // If Image/Buffer structs don't store these, add them here.
    // VkDevice device = VK_NULL_HANDLE;

     // Default constructor (optional, but can be useful)
    FilteringOutput() = default;

    // Move constructor (important for transferring ownership)
    FilteringOutput(FilteringOutput&& other) noexcept :
        compactedBlockIdBuffer(std::move(other.compactedBlockIdBuffer)),
        activeBlockCountBuffer(std::move(other.activeBlockCountBuffer)),
        // indirectDispatchBuffer(std::move(other.indirectDispatchBuffer)),
        activeBlockCount(other.activeBlockCount),
        tempResources(std::move(other.tempResources))
        // Move device/allocator if added
         {
            // Reset other's handles if Image/Buffer move semantics don't already
            other.activeBlockCount = 0;
         }

    // Move assignment operator (optional, but good practice)
    FilteringOutput& operator=(FilteringOutput&& other) noexcept {
        if (this != &other) {
             // Need to destroy existing resources owned by *this* first
             // This depends heavily on how Image/Buffer manage resources.
             // If they are RAII and require explicit device/allocator,
             // destruction here is complex without those handles.
             // Simplest if Image/Buffer handle moves cleanly or if caller
             // ensures *this* is empty before move-assigning.

             compactedBlockIdBuffer = std::move(other.compactedBlockIdBuffer);
             activeBlockCountBuffer = std::move(other.activeBlockCountBuffer);
             // indirectDispatchBuffer = std::move(other.indirectDispatchBuffer);
             activeBlockCount = other.activeBlockCount;
             tempResources = std::move(other.tempResources);
             // Move device/allocator if added

             other.activeBlockCount = 0; // Reset scalar type
        }
        return *this;
    }


    // Prevent copying if Image/Buffer are not copyable
    FilteringOutput(const FilteringOutput&) = delete;
    FilteringOutput& operator=(const FilteringOutput&) = delete;

    // Add a cleanup method if preferred over manual destruction by the caller
    void cleanup(VkDevice device) const {
        destroyBuffer(compactedBlockIdBuffer, device);
        destroyBuffer(activeBlockCountBuffer, device);
        // destroyBuffer(indirectDispatchBuffer, device);
        // Reset members to indicate they are cleaned up
    }
};