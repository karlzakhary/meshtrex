#pragma once

#include "common.h"
#include "image.h"
#include "buffer.h"
#include "resources.h" // Defines Image and Buffer structs
#include <cstdint>

struct MinMaxOutput {
    Image volumeImage{};             // Handle to the uploaded volume data image
    Image minMaxImage{};             // Handle to the intermediate min/max image (optional need)
    std::vector<VkImageView> minMaxMipViews;
    VkImageView minMaxFull{};

    // Add handles needed for destruction by the caller
    // If Image/Buffer structs don't store these, add them here.
    // VkDevice device = VK_NULL_HANDLE;

     // Default constructor (optional, but can be useful)
    MinMaxOutput() = default;

    // Move constructor (important for transferring ownership)
    MinMaxOutput(MinMaxOutput&& other) noexcept :
        volumeImage(std::move(other.volumeImage)),
        minMaxImage(std::move(other.minMaxImage)),
        minMaxMipViews(std::move(other.minMaxMipViews)),
        minMaxFull(std::move(other.minMaxFull))
    {}
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
             minMaxFull = std::move(other.minMaxFull);
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
        vkDestroyImageView(device, minMaxFull, nullptr);
        for (auto view : minMaxMipViews) {
            vkDestroyImageView(device, view, nullptr);
        }
        // Reset members to indicate they are cleaned up
    }
};