#version 450 // Minimum version for subgroup operations

// Required extensions for subgroup operations (check device support)
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require // For subgroupMin/Max (alternative)
#extension GL_KHR_shader_subgroup_shuffle : require // For subgroupShuffleDown
#extension GL_KHR_shader_subgroup_shuffle_relative : require // Needed for subgroupShuffleDown/Up
// --- Configuration Constants ---
// These should ideally be passed via specialization constants, UBOs, or push constants
// for flexibility, but are defined here for direct translation.

// Local workgroup size (matches CUDA block dimensions)
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 8

// Define the size of the scalar field volume (adjust to your data)
// This should match the dimensions of the input image3D
#define VOLUME_DIM_X 256
#define VOLUME_DIM_Y 256
#define VOLUME_DIM_Z 256

// Define grid dimensions (number of workgroups) - needed for global index calculation
// These would typically come from a UBO or push constants in a real app.
// Example values if the volume is 256^3 and block is 8^3:
#define GRID_DIM_X (VOLUME_DIM_X / BLOCK_DIM_X) // 32
#define GRID_DIM_Y (VOLUME_DIM_Y / BLOCK_DIM_Y) // 32
#define GRID_DIM_Z (VOLUME_DIM_Z / BLOCK_DIM_Z) // 32

// --- Layout Definitions ---

// Define the local workgroup size
layout (local_size_x = BLOCK_DIM_X, local_size_y = BLOCK_DIM_Y, local_size_z = BLOCK_DIM_Z) in;

// Input scalar field data using a 3D image texture
// Format r8ui contains unsigned 8-bit integer data [0, 255].
layout(binding = 0, r8ui) uniform readonly uimage3D volume; // Use r8ui for unsigned int access

// Output min/max pairs for each block (using SSBO)
// Stores normalized float values [0.0, 1.0] as vec2.
layout(binding = 1, std430) buffer MinMaxBuffer { // Renamed buffer, uses vec2
    vec2 minMax[];
};

// --- Shared Memory ---
// Shared memory for intermediate min/max values per subgroup (warp).
// Still uses uvec2 internally as calculations are based on uint input.
const uint totalInvocations = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
// Allocate based on a minimum subgroup size of 32 (adjust if needed)
shared uvec2 s_subgroupMinMax[totalInvocations / 32];


void main() {
    // Calculate the 3D index of the current workgroup (block)
    const uvec3 workGroupID = gl_WorkGroupID; // Equivalent to blockIdx

    // Calculate the 1D global index for the output array
    const uint globalBlockIdx = workGroupID.z * (GRID_DIM_X * GRID_DIM_Y) +
    workGroupID.y * GRID_DIM_X +
    workGroupID.x;

    // Calculate thread's local invocation ID within the workgroup
    const uvec3 localInvocationID = gl_LocalInvocationID; // Equivalent to threadIdx
    const uint localInvocationIndex = gl_LocalInvocationIndex; // 1D version of threadIdx

    // Calculate subgroup ID and invocation ID within the subgroup
    const uint subgroupId = gl_SubgroupID;
    const uint subgroupInvocationId = gl_SubgroupInvocationID; // Equivalent to laneId

    // Calculate the global 3D integer coordinates for imageLoad
    const ivec3 imageCoord = ivec3(workGroupID * gl_WorkGroupSize + localInvocationID);

    // Initialize min/max values for this invocation (using uint for r8ui)
    uint invocationMin = 0xFFFFFFFFu; // UINT_MAX
    uint invocationMax = 0u;          // UINT_MIN

    // Check boundary conditions: ensure the coordinates are within the image dimensions
    if (imageCoord.x < imageSize(volume).x &&
    imageCoord.y < imageSize(volume).y &&
    imageCoord.z < imageSize(volume).z) {

        // Read the scalar value using imageLoad (fetches from r8ui format as uint)
        const uint scalarValue = imageLoad(volume, imageCoord).x;

        // Update invocation's local min/max
        invocationMin = scalarValue;
        invocationMax = scalarValue;
    }

    // --- Step 1: Subgroup-level reduction using subgroupShuffleDown ---
    // Reduce uint values within the subgroup
    for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
        uint downMin = subgroupShuffleDown(invocationMin, offset);
        uint downMax = subgroupShuffleDown(invocationMax, offset);
        invocationMin = min(invocationMin, downMin);
        invocationMax = max(invocationMax, downMax);
    }

    // --- Step 2: Write partial results (subgroup min/max) to shared memory ---
    if (subgroupInvocationId == 0) {
        if (subgroupId < (totalInvocations / 32)) {
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
        if(sharedMemIndex < numSubgroups && sharedMemIndex < (totalInvocations / 32) ) {
            subgroupResult = s_subgroupMinMax[sharedMemIndex];
        }

        // Perform the final reduction *within the first subgroup*.
        for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
            uint downMin = subgroupShuffleDown(subgroupResult.x, offset);
            uint downMax = subgroupShuffleDown(subgroupResult.y, offset);
            subgroupResult.x = min(subgroupResult.x, downMin);
            subgroupResult.y = max(subgroupResult.y, downMax);
        }

        // --- Step 4: Normalize and Write final block result ---
        // The first invocation of the first subgroup (subgroupInvocationId == 0)
        // normalizes the uint min/max result and writes it to the output buffer.
        if (subgroupInvocationId == 0) {
            // Normalize the uint values [0, 255] to float [0.0, 1.0]
            vec2 normalizedResult = vec2(subgroupResult) / 255.0f;

            // Write the normalized vec2 result to the output buffer
            minMax[globalBlockIdx] = normalizedResult;
        }
    }
}