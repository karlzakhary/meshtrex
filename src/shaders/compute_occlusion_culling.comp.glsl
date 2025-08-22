#version 460 core
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// Compute-based occlusion culling using Hi-Z pyramid
// This shader produces a complete PVS for the current frame only
// No temporal logic - that's handled by the PVS difference shader

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Min-max octree input (3D texture with mip levels) - using usampler3D for unsigned int format
layout(set = 0, binding = 0) uniform usampler3D minMaxTexture;

// Hi-Z pyramid sampler
layout(set = 0, binding = 1) uniform sampler2D hiZPyramid;

// Previous frame's visibility bitfield for temporal coherence (1 bit per block)
layout(set = 0, binding = 2) restrict readonly buffer PreviousVisibilityBitfield {
    uint bits[];  // Each uint stores visibility for 32 blocks
} previousVisibility;

// Output: Current frame's visibility bitfield (1 bit per block)
layout(set = 0, binding = 3) restrict buffer CurrentVisibilityBitfield {
    uint bits[];  // Each uint stores visibility for 32 blocks
} currentVisibility;

// Note: binding 4 is still needed for descriptor set compatibility but not used
layout(set = 0, binding = 4) restrict buffer Unused {
    uint dummy;
} unused;

// Uniforms
layout(set = 0, binding = 5) uniform UniformBuffer {
    mat4 viewProj;
    mat4 prevViewProj;
    vec4 frustumPlanes[6];  // Frustum planes in world space
    vec3 volumeMin;
    float blockSize;
    vec3 volumeMax;
    float isovalue;
    ivec3 volumeDimensions;
    uint totalBlocks;
    vec2 screenSize;
    uint hiZLevels;
    uint previousPVSCount;
} ubo;

// No push constants needed - we process all blocks directly

// Test if a box is inside the view frustum by projecting and checking screen bounds
bool isInFrustum(vec3 minPos, vec3 maxPos) {
    // Check box against all 6 frustum planes
    for (int i = 0; i < 6; i++) {
        vec3 normal = ubo.frustumPlanes[i].xyz;
        float dist = ubo.frustumPlanes[i].w;

        // Find the corner of the box that is "most positive"
        // in the direction of the plane normal.
        vec3 p_vertex = minPos;
        if (normal.x > 0) p_vertex.x = maxPos.x;
        if (normal.y > 0) p_vertex.y = maxPos.y;
        if (normal.z > 0) p_vertex.z = maxPos.z;

        // If this corner is outside the plane, the entire box is outside.
        if (dot(normal, p_vertex) + dist < 0.0) {
            return false; // Box is culled
        }
    }
    return true; // Box is at least partially inside
}

// Project a 3D bounding box to screen space and get depth range
// Reversed-Z: We need the maximum depth (closest point) for occlusion testing
void projectToScreen(vec3 minPos, vec3 maxPos, out vec2 screenMin, out vec2 screenMax, out float minZ) {
    screenMin = vec2(1.0);
    screenMax = vec2(0.0);
    minZ = 0.0;  // Reversed-Z: Initialize to far plane
    
    // Test all 8 corners of the bounding box
    for (int i = 0; i < 8; i++) {
        vec3 corner = vec3(
            (i & 1) != 0 ? maxPos.x : minPos.x,
            (i & 2) != 0 ? maxPos.y : minPos.y,
            (i & 4) != 0 ? maxPos.z : minPos.z
        );
        
        vec4 clipPos = ubo.viewProj * vec4(corner, 1.0);
        if (clipPos.w > 0.0) {
            vec3 ndc = clipPos.xyz / clipPos.w;
            vec2 screen = ndc.xy * 0.5 + 0.5;  // Convert from NDC [-1,1] to UV [0,1]
            
            screenMin = min(screenMin, screen);
            screenMax = max(screenMax, screen);
            minZ = max(minZ, ndc.z);  // Reversed-Z: Get maximum (closest) depth
        }
    }
}

// Select appropriate Hi-Z level based on screen space size
uint selectHiZLevel(vec2 screenMin, vec2 screenMax) {
    vec2 screenSize = (screenMax - screenMin) * ubo.screenSize;
    float maxSize = max(screenSize.x, screenSize.y);
    
    // Select level where one texel covers approximately the screen region
    uint level = uint(max(0.0, log2(maxSize)));
    return min(level, ubo.hiZLevels - 1);
}

// Test occlusion against Hi-Z pyramid
bool isOccludedByHiZ(vec2 screenMin, vec2 screenMax, float minZ) {
    // --- (Keep all of your initial checks the same) ---
    screenMin = clamp(screenMin, vec2(0.0), vec2(1.0));
    screenMax = clamp(screenMax, vec2(0.0), vec2(1.0));
    
    if (screenMin.x >= screenMax.x || screenMin.y >= screenMax.y) {
        return true;
    }
    
    vec2 coverage = screenMax - screenMin;
    if (coverage.x > 0.1 || coverage.y > 0.1) {
        return false;
    }
    
    if (minZ <= 0.0) {
        return false;
    }

    uint level = selectHiZLevel(screenMin, screenMax);
    float lod = float(level);

    // 1. Sample the four corners of the screen-space bounding box
    float d0 = textureLod(hiZPyramid, screenMin, lod).r;
    float d1 = textureLod(hiZPyramid, vec2(screenMax.x, screenMin.y), lod).r;
    float d2 = textureLod(hiZPyramid, screenMax, lod).r;
    float d3 = textureLod(hiZPyramid, vec2(screenMin.x, screenMax.y), lod).r;

    // 2. Find the closest occluder. With a reversed-Z max-pyramid, this is the MAXIMUM of the sampled depths.
    float closestOccluderDepth = max(max(d0, d1), max(d2, d3));
    
    return minZ < closestOccluderDepth - 0.00002;
}

void main() {
    uint blockIndex = gl_GlobalInvocationID.x;
    
    // Check bounds
    if (blockIndex >= ubo.totalBlocks) {
        return;
    }
    
    // TEMPORAL COHERENCE: Check if this block was in previous PVS using bitfield
    bool wasInPreviousPVS = false;
    if (ubo.previousPVSCount > 0) {
        uint bitfieldIdx = blockIndex / 32;
        uint bitIdx = blockIndex % 32;
        wasInPreviousPVS = (previousVisibility.bits[bitfieldIdx] & (1u << bitIdx)) != 0;
    }
    
    // 1. Min-Max Test
    ivec3 numBlocks = (ubo.volumeDimensions + 7) / 8;
    ivec3 blockCoord;
    blockCoord.z = int(blockIndex) / (numBlocks.x * numBlocks.y);
    blockCoord.y = (int(blockIndex) % (numBlocks.x * numBlocks.y)) / numBlocks.x;
    blockCoord.x = int(blockIndex) % numBlocks.x;
    
    // Sample min-max values using texelFetch for exact sampling
    uvec2 minMax = texelFetch(minMaxTexture, blockCoord, 0).rg;
    // ubo.isovalue is normalized (0-1), but min-max values are in 0-255 range
    float scaledIsovalue = ubo.isovalue * 255.0;
        
    // Block is active if isovalue is within [min, max] range
    // Cull if isovalue is outside the range
    if (scaledIsovalue < float(minMax.x) || scaledIsovalue > float(minMax.y)) {
        return;
    }
    
    // 2. Frustum Culling
    vec3 blockMin = ubo.volumeMin + vec3(blockCoord) * ubo.blockSize;
    vec3 blockMax = blockMin + vec3(ubo.blockSize);
    
    if (!isInFrustum(blockMin, blockMax)) {
        return;
    }
    
    // 3. HI-Z OCCLUSION TEST
    // Only test if we have a valid Hi-Z pyramid from previous frame
    if (ubo.previousPVSCount > 0) {
        vec2 screenMin, screenMax;
        float minZ;
        projectToScreen(blockMin, blockMax, screenMin, screenMax, minZ);

        if (minZ <= 0.0 || screenMin.x > screenMax.x || screenMin.y > screenMax.y) {
            // Conservative: don't cull if projection failed
        } else {
            // Use consistent Hi-Z test for all blocks
            // The temporal coherence is handled by the two-pass rendering system
            uint level = selectHiZLevel(screenMin, screenMax);
            float lod = float(level);

            // Sample four corners for robust occlusion test
            float d0 = textureLod(hiZPyramid, screenMin, lod).r;
            float d1 = textureLod(hiZPyramid, vec2(screenMax.x, screenMin.y), lod).r;
            float d2 = textureLod(hiZPyramid, screenMax, lod).r;
            float d3 = textureLod(hiZPyramid, vec2(screenMin.x, screenMax.y), lod).r;
            
            float closestOccluderDepth = max(max(d0, d1), max(d2, d3));
            
            // Use a consistent bias for all blocks to prevent artifacts
            float bias = 0.00002;
            bool isOccluded = minZ < closestOccluderDepth - bias;
            
            if (isOccluded) {
                return; // Cull the block
            }
        }
    }

    // If the block survived all tests, mark it as visible for the next frame.
    uint bitfieldIdx = blockIndex / 32;
    uint bitIdx = blockIndex % 32;
    atomicOr(currentVisibility.bits[bitfieldIdx], 1u << bitIdx);
}