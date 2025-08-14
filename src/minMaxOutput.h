#pragma once

#include "common.h"
#include "image.h"
#include "buffer.h"
#include "resources.h" // Defines Image and Buffer structs
#include "temporaryResources.h"
#include <cstdint>

struct MinMaxOutput {
    Image volumeImage{};
    Image minMaxImage{};
    std::vector<VkImageView> minMaxMipViews;
    VkSampler minMaxSampler = VK_NULL_HANDLE;
    TemporaryResources tempResources;
    
    MinMaxOutput() = default;
    // Move constructor (important for transferring ownership)
    MinMaxOutput(MinMaxOutput&& other) noexcept :
        volumeImage(std::move(other.volumeImage)),
        minMaxImage(std::move(other.minMaxImage)),
        minMaxMipViews(std::move(other.minMaxMipViews)),
        minMaxSampler(other.minMaxSampler),
        tempResources(std::move(other.tempResources))
    {
        other.minMaxSampler = VK_NULL_HANDLE;
    }
    // Move assignment operator (optional, but good practice)
    MinMaxOutput& operator=(MinMaxOutput&& other) noexcept {
        if (this != &other) {
             // Need to destroy existing resources owned by *this* first
             // This depends heavily on how Image/Buffer manage resources.
             // If they are RAII and require explicit device/allocator,
             // destruction here is complex without those handles.
             // Simplest if Image/Buffer handle moves cleanly or if caller
             // ensures *this* is empty before move-assigning.

             volumeImage = std::move(other.volumeImage);
             minMaxImage = std::move(other.minMaxImage);
             minMaxMipViews = std::move(other.minMaxMipViews);
             minMaxSampler = other.minMaxSampler;
             tempResources = std::move(other.tempResources);
             other.minMaxSampler = VK_NULL_HANDLE;
        }
        return *this;
    }


    // Prevent copying if Image/Buffer are not copyable
    MinMaxOutput(const MinMaxOutput&) = delete;
    MinMaxOutput& operator=(const MinMaxOutput&) = delete;

    // Add a cleanup method if preferred over manual destruction by the caller
    void cleanup(VkDevice device) const {
        destroyImage(volumeImage, device);
        destroyImage(minMaxImage, device);
        for (auto view : minMaxMipViews) {
            vkDestroyImageView(device, view, nullptr);
        }
        if (minMaxSampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, minMaxSampler, nullptr);
        }
        // Reset members to indicate they are cleaned up
    }
};