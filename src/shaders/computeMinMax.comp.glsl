#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require // For subgroupMin/Max (alternative)
#extension GL_KHR_shader_subgroup_shuffle : require // For basic subgroupShuffle
#extension GL_KHR_shader_subgroup_shuffle_relative : require // Needed for subgroupShuffleDown/Up

// Push constants containing dimensions (matches C++ struct)
layout(push_constant) uniform PushConstants {
    uvec4 volumeDim;    // Actual dimensions of the input volume texture
    uvec4 blockDim;     // Should match local_size (e.g., 8,8,8)
    uvec4 blockGridDim; // Dimensions of the dispatch grid (and the output image)
    float isovalue;     // The target isovalue
} pc;

// --- Layout Definitions ---

// Define the local workgroup size (should match pc.blockDim)
layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Binding 0: Input scalar field data using a 3D image texture
// Assumes r8ui format for unsigned 8-bit integer data [0, 255].
layout(binding = 0, r8ui) uniform readonly uimage3D volume;

// Binding 1: Output min/max pairs using a 3D image texture
// Format rg32ui stores two 32-bit unsigned integers (matching uvec2).
// Dimensions match the block grid dimensions.
layout(binding = 1, rg32ui) uniform writeonly uimage3D minMaxOutputVolume;

// --- Shared Memory ---
// Shared memory for intermediate min/max values per subgroup.
// Uses uvec2 to store uint min/max values.
const uint totalInvocations = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
// Allocate based on a minimum subgroup size of 32 (adjust if needed for target hardware)
// Ensure this allocation is sufficient for max subgroups per workgroup (totalInvocations / minSubgroupSize)
shared uvec2 s_subgroupMinMax[totalInvocations / 32];


void main() {
    // Calculate the 3D index of the current workgroup (block)
    // This will be used as the coordinate to write the final result to the output image.
    const uvec3 workGroupID = gl_WorkGroupID;

    // Calculate thread's local invocation ID within the workgroup
    const uvec3 localInvocationID = gl_LocalInvocationID;
    const uint localInvocationIndex = gl_LocalInvocationIndex;

    // Calculate subgroup ID and invocation ID within the subgroup
    const uint subgroupId = gl_SubgroupID;
    const uint subgroupInvocationId = gl_SubgroupInvocationID;

    // Calculate the global 3D integer coordinates for reading the input volume
    // Use ivec3 for imageLoad coordinates
    const ivec3 imageCoord = ivec3(workGroupID * gl_WorkGroupSize + localInvocationID);

    // Initialize min/max values for this invocation (using uint for r8ui input)
    uint invocationMin = 0xFFFFFFFFu; // UINT_MAX
    uint invocationMax = 0u;          // UINT_MIN (0 for uint)

    // Check boundary conditions using volume dimensions from push constants
    if (imageCoord.x < imageSize(volume).x &&
    imageCoord.y < imageSize(volume).y &&
    imageCoord.z < imageSize(volume).z) {

        // Read the scalar value using imageLoad (fetches from r8ui format as uint)
        const uint scalarValue = imageLoad(volume, imageCoord).x;

        // Update invocation's local min/max
        invocationMin = scalarValue;
        invocationMax = scalarValue;
    }
    // If outside bounds, the initial UINT_MAX/0 values will be ignored in reduction.


    // --- Step 1: Subgroup-level reduction using subgroupShuffleDown ---
    // Reduce uint values within the subgroup.
    for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
        uint downMin = subgroupShuffleDown(invocationMin, offset);
        uint downMax = subgroupShuffleDown(invocationMax, offset);
        invocationMin = min(invocationMin, downMin);
        invocationMax = max(invocationMax, downMax);
    }
    // After this loop, the invocation with subgroupInvocationId == 0 holds the subgroup's min/max.


    // --- Step 2: Write partial results (subgroup min/max) to shared memory ---
    if (subgroupInvocationId == 0) {
        // Ensure index is within bounds of the shared memory array
        if (subgroupId < (totalInvocations / 32)) { // Check against allocated size
            s_subgroupMinMax[subgroupId] = uvec2(invocationMin, invocationMax); // Store uvec2
        }
    }

    // Synchronize all invocations within the workgroup
    barrier();


    // --- Step 3: Final reduction using the first subgroup ---
    if (subgroupId == 0) {
        // Load the appropriate subgroup's min/max result (uvec2) from shared memory
        uvec2 subgroupResult = uvec2(0xFFFFFFFFu, 0u); // Initialize safely for uint min/max

        uint sharedMemIndex = subgroupInvocationId;
        uint numSubgroups = totalInvocations / gl_SubgroupSize;
        // Check bounds against actual subgroup count and allocated shared memory size
        if(sharedMemIndex < numSubgroups && sharedMemIndex < (totalInvocations / 32) ) {
            subgroupResult = s_subgroupMinMax[sharedMemIndex];
        }

        // Perform the final reduction *within the first subgroup*.
        // Requires GL_KHR_shader_subgroup_shuffle_relative.
        for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
            uint downMin = subgroupShuffleDown(subgroupResult.x, offset);
            uint downMax = subgroupShuffleDown(subgroupResult.y, offset);
            subgroupResult.x = min(subgroupResult.x, downMin);
            subgroupResult.y = max(subgroupResult.y, downMax);
        }

        // --- Step 4: Write final block result to Output Image ---
        // The first invocation of the first subgroup (subgroupInvocationId == 0)
        // writes the final uint min/max pair for the workgroup (block) to the output image.
        if (subgroupInvocationId == 0) {
            // Write the uvec2 result to the output image at the 3D coordinate
            // corresponding to this workgroup's ID.
            // imageStore requires ivec3 coordinates and uvec4 value for rg32ui format.
            imageStore(minMaxOutputVolume, ivec3(workGroupID), uvec4(subgroupResult, 0, 0));
        }
    }
}
