#pragma once

#include "buffer.h"
#include "common.h"
#include <vector>

// Structure to track temporary resources that need cleanup after command buffer submission
struct TemporaryResources {
    std::vector<Buffer> buffers;
    std::vector<VkSampler> samplers;
    std::vector<VkPipeline> pipelines;
    std::vector<VkPipelineLayout> pipelineLayouts;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkDescriptorPool> descriptorPools;
    std::vector<VkShaderModule> shaderModules;
    VkDevice device = VK_NULL_HANDLE;
    
    void addBuffer(const Buffer& buffer) {
        buffers.push_back(buffer);
    }
    
    void addSampler(VkSampler sampler) {
        samplers.push_back(sampler);
    }
    
    void addPipeline(VkPipeline pipeline) {
        pipelines.push_back(pipeline);
    }
    
    void addPipelineLayout(VkPipelineLayout layout) {
        pipelineLayouts.push_back(layout);
    }

    void addDescriptorSet(VkDescriptorSet set) {
        descriptorSets.push_back(set);
    }
    
    void addDescriptorSetLayout(VkDescriptorSetLayout layout) {
        descriptorSetLayouts.push_back(layout);
    }
    
    void addDescriptorPool(VkDescriptorPool pool) {
        descriptorPools.push_back(pool);
    }
    
    void addShaderModule(VkShaderModule module) {
        shaderModules.push_back(module);
    }
    
    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            // Destroy in reverse order of dependency
            for (const auto& pipeline : pipelines) {
                vkDestroyPipeline(device, pipeline, nullptr);
            }
            pipelines.clear();
            
            for (const auto& layout : pipelineLayouts) {
                vkDestroyPipelineLayout(device, layout, nullptr);
            }
            pipelineLayouts.clear();

            for (const auto& pool : descriptorPools) {
                vkDestroyDescriptorPool(device, pool, nullptr);
            }
            descriptorPools.clear();
            
            for (const auto& layout : descriptorSetLayouts) {
                vkDestroyDescriptorSetLayout(device, layout, nullptr);
            }
            descriptorSetLayouts.clear();
            
            for (const auto& module : shaderModules) {
                vkDestroyShaderModule(device, module, nullptr);
            }
            shaderModules.clear();
            
            for (const auto& sampler : samplers) {
                vkDestroySampler(device, sampler, nullptr);
            }
            samplers.clear();
            
            for (const auto& buffer : buffers) {
                destroyBuffer(buffer, device);
            }
            buffers.clear();
        }
    }
    
    // Move constructor
    TemporaryResources(TemporaryResources&& other) noexcept
        : buffers(std::move(other.buffers)),
          samplers(std::move(other.samplers)),
          pipelines(std::move(other.pipelines)),
          pipelineLayouts(std::move(other.pipelineLayouts)),
          descriptorSetLayouts(std::move(other.descriptorSetLayouts)),
          descriptorPools(std::move(other.descriptorPools)),
          shaderModules(std::move(other.shaderModules)),
          device(other.device) {
        // Clear the other object so its destructor won't clean up
        other.device = VK_NULL_HANDLE;
    }
    
    // Move assignment operator
    TemporaryResources& operator=(TemporaryResources&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            cleanup();
            
            // Move resources from other
            buffers = std::move(other.buffers);
            samplers = std::move(other.samplers);
            pipelines = std::move(other.pipelines);
            pipelineLayouts = std::move(other.pipelineLayouts);
            descriptorSetLayouts = std::move(other.descriptorSetLayouts);
            descriptorPools = std::move(other.descriptorPools);
            shaderModules = std::move(other.shaderModules);
            device = other.device;
            
            // Clear the other object
            other.device = VK_NULL_HANDLE;
        }
        return *this;
    }
    
    // Delete copy constructor and copy assignment to prevent accidental copies
    TemporaryResources(const TemporaryResources&) = delete;
    TemporaryResources& operator=(const TemporaryResources&) = delete;
    
    // Default constructor
    TemporaryResources() = default;
    
    ~TemporaryResources() {
        cleanup();
    }
};