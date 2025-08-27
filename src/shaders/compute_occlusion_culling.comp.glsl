#version 460 core
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_arithmetic: enable

// Use DEBUG_ENABLED from CMake (defaults to 0 if not defined)
#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 0
#endif

// Enable debug metrics collection only in debug builds
#define DEBUG_OCCLUSION DEBUG_ENABLED

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

// Uniforms
layout(set = 0, binding = 4) uniform UniformBuffer {
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

// Simple debug buffer for Hi-Z occlusion statistics
#ifdef DEBUG_OCCLUSION
layout(set = 0, binding = 5) coherent buffer DebugStats {
    uint hiZTests;      // Total blocks that reached Hi-Z test
    uint hiZOccluded;   // Total blocks occluded by Hi-Z
    uint atomicOps;     // Total atomic operations performed
} debugStats;
#endif

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
    screenMin = vec2(2.0);
    screenMax = vec2(-1.0);
    minZ = 0.0;  // Reversed-Z: Initialize to far plane (will find maximum/closest)
    
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
bool isOccludedByHiZ(vec2 screenMin, vec2 screenMax, float minZ, vec3 blockMin, vec3 blockMax) {
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
    ivec2 sizeL = textureSize(hiZPyramid, int(level));
    vec2  centerUV = clamp((screenMin + screenMax) * 0.5, 0.0, 1.0);
    ivec2 coord = clamp(ivec2(centerUV * vec2(sizeL)), ivec2(0), sizeL - ivec2(1));
    float hizMin = texelFetch(hiZPyramid, coord, int(level)).r;
    bool occluded = (minZ  + 1e-5 <= hizMin); 
    return occluded;
}

void main() {
    uint blockIndex = gl_GlobalInvocationID.x;

    // --- per-invocation bookkeeping (no early returns; we reduce at the end) ---
    bool inRange  = (blockIndex < ubo.totalBlocks);
    bool alive    = inRange;     // survives minmax + frustum
    bool visible  = false;       // final decision for this invocation

    // 1) Compute 3D block coord only if needed
    ivec3 numBlocks = (ubo.volumeDimensions + 7) / 8;
    ivec3 blockCoord = ivec3(0);

    if (alive) {
        blockCoord.z = int(blockIndex) / (numBlocks.x * numBlocks.y);
        blockCoord.y = (int(blockIndex) % (numBlocks.x * numBlocks.y)) / numBlocks.x;
        blockCoord.x = int(blockIndex) % numBlocks.x;

        // Min-max test
        uvec2 minMax = texelFetch(minMaxTexture, blockCoord, 0).rg;
        float scaledIsovalue = ubo.isovalue * 255.0;
        if (scaledIsovalue < float(minMax.x) || scaledIsovalue > float(minMax.y)) {
            alive = false;
        }
    }

    // 2) Frustum test
    vec3 blockMin = vec3(0.0);
    vec3 blockMax = vec3(0.0);
    if (alive) {
        blockMin = ubo.volumeMin + vec3(blockCoord) * ubo.blockSize;
        blockMax = blockMin + vec3(ubo.blockSize);
        if (!isInFrustum(blockMin, blockMax)) {
            alive = false;
        }
    }

    if (alive) {
        if (ubo.previousPVSCount > 0u) {
            vec2 screenMin, screenMax;
            float minZ;
            projectToScreen(blockMin, blockMax, screenMin, screenMax, minZ);

            bool skipHiZ = false;
            const float nearPlaneThreshold = 0.99; // reversed-Z
            if (minZ > nearPlaneThreshold) {
                skipHiZ = true;        // too close → visible
            } else if (minZ <= 0.0 || screenMin.x >= screenMax.x || screenMin.y >= screenMax.y) {
                skipHiZ = true;        // conservative fallback
            }

            if (skipHiZ) {
                visible = true;
            } else {
#ifdef DEBUG_OCCLUSION
                atomicAdd(debugStats.hiZTests, 1);
#endif
                bool occluded = isOccludedByHiZ(screenMin, screenMax, minZ, blockMin, blockMax);
                visible = !occluded;
#ifdef DEBUG_OCCLUSION
                if (occluded) {
                    atomicAdd(debugStats.hiZOccluded, 1);
                }
#endif
            }
        } else {
            // No Hi-Z yet (bootstrap): visible if it passed minmax+frustum
            visible = true;
        }
    }

    // --- Workgroup aggregation: one atomic per 32 lanes ---

    uint lane = gl_SubgroupInvocationID & 31u;
    uint tile = gl_LocalInvocationID.x >> 5;    // 0..7 within WG

    // Each lane contributes its bit (or 0)
    uint myBit = visible ? (1u << lane) : 0u;

    // OR within clusters of 32 lanes
    uint mask  = subgroupOr(myBit);

    // Lane 0 of each 32-lane cluster writes one global atomic
    if (lane == 0u) {
        uint groupBase = gl_WorkGroupID.x * gl_WorkGroupSize.x; // WG*256
        uint baseWord  = (groupBase >> 5) + tile;
        if (mask != 0u) {
            atomicOr(currentVisibility.bits[baseWord], mask);
#ifdef DEBUG_OCCLUSION
            atomicAdd(debugStats.atomicOps, 1);
#endif
        }
    }
}