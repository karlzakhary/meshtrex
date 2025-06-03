#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

// Push constants containing dimensions (matches C++ struct)
layout(push_constant) uniform PushConstants {
    uvec4 volumeDim;    // Actual dimensions of the input volume texture
    uvec4 blockDim;     // Block dimensions (e.g., 8,8,8)
    uvec4 blockStride;  // Stride for overlap (e.g., 7,7,7) - CRITICAL FOR SEAMLESS!
    uvec4 blockGridDim; // Grid dimensions = ceil(volumeDim / blockStride)
    float isovalue;     // The target isovalue
} pc;

// --- Layout Definitions ---

// Define the local workgroup size (should match pc.blockDim)
layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Binding 0: Input scalar field data using a 3D image texture
layout(binding = 0, r8ui) uniform readonly uimage3D volume;

// Binding 1: Output min/max pairs using a 3D image texture
// Dimensions match the block grid dimensions (calculated using stride)
layout(binding = 1, rg32ui) uniform writeonly uimage3D minMaxOutputVolume;

// --- Shared Memory ---
const uint totalInvocations = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
shared uvec2 s_subgroupMinMax[totalInvocations / 32];

void main() {
    // Calculate the 3D index of the current workgroup (block)
    const uvec3 workGroupID = gl_WorkGroupID;

    // Calculate thread's local invocation ID within the workgroup
    const uvec3 localInvocationID = gl_LocalInvocationID;
    const uint localInvocationIndex = gl_LocalInvocationIndex;

    // Calculate subgroup ID and invocation ID within the subgroup
    const uint subgroupId = gl_SubgroupID;
    const uint subgroupInvocationId = gl_SubgroupInvocationID;

    // =================================================================
    // CRITICAL CHANGE: Use blockStride for block positioning!
    // This creates the overlap between adjacent blocks
    // =================================================================
    
    // Calculate the starting position of this block in the volume
    // OLD: workGroupID * pc.blockDim.xyz (creates gaps)
    // NEW: workGroupID * pc.blockStride.xyz (creates overlap)
    const uvec3 blockOriginInVolume = workGroupID * pc.blockStride.xyz;
    
    // Calculate the global 3D coordinates for this thread's voxel
    const ivec3 imageCoord = ivec3(blockOriginInVolume + localInvocationID);
    
    // =================================================================
    // Example of how this creates overlap:
    // Block 0: samples voxels [0,0,0] to [7,7,7]
    // Block 1: samples voxels [7,0,0] to [14,7,7] (overlap at x=7!)
    // Without stride, Block 1 would start at [8,0,0] (gap!)
    // =================================================================

    // Initialize min/max values for this invocation
    uint invocationMin = 0xFFFFFFFFu; // UINT_MAX
    uint invocationMax = 0u;          // UINT_MIN

    // Check boundary conditions
    if (all(lessThan(imageCoord, ivec3(pc.volumeDim.xyz)))) {
        // Read the scalar value
        const uint scalarValue = imageLoad(volume, imageCoord).x;
        
        // Update invocation's local min/max
        invocationMin = scalarValue;
        invocationMax = scalarValue;
    }
    // If outside bounds, the initial values will be ignored in reduction

    // --- Step 1: Subgroup-level reduction using subgroupShuffleDown ---
    for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
        uint downMin = subgroupShuffleDown(invocationMin, offset);
        uint downMax = subgroupShuffleDown(invocationMax, offset);
        invocationMin = min(invocationMin, downMin);
        invocationMax = max(invocationMax, downMax);
    }

    // --- Step 2: Write partial results to shared memory ---
    if (subgroupInvocationId == 0) {
        if (subgroupId < (totalInvocations / 32)) {
            s_subgroupMinMax[subgroupId] = uvec2(invocationMin, invocationMax);
        }
    }

    barrier();

    // --- Step 3: Final reduction using the first subgroup ---
    if (subgroupId == 0) {
        uvec2 subgroupResult = uvec2(0xFFFFFFFFu, 0u);

        uint sharedMemIndex = subgroupInvocationId;
        uint numSubgroups = totalInvocations / gl_SubgroupSize;
        
        if(sharedMemIndex < numSubgroups && sharedMemIndex < (totalInvocations / 32)) {
            subgroupResult = s_subgroupMinMax[sharedMemIndex];
        }

        // Final reduction within the first subgroup
        for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
            uint downMin = subgroupShuffleDown(subgroupResult.x, offset);
            uint downMax = subgroupShuffleDown(subgroupResult.y, offset);
            subgroupResult.x = min(subgroupResult.x, downMin);
            subgroupResult.y = max(subgroupResult.y, downMax);
        }

        // --- Step 4: Write final block result ---
        if (subgroupInvocationId == 0) {
            // Write to the grid position (not the volume position)
            imageStore(minMaxOutputVolume, ivec3(workGroupID), uvec4(subgroupResult, 0, 0));
        }
    }
}