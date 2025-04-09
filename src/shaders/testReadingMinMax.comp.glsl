#version 450
#extension GL_KHR_shader_subgroup_basic : require // Included for completeness, not strictly needed by this simple shader

// --- Push Constants ---
// Only need grid dimensions to know how many texels/blocks to read
// Push constants containing dimensions (matches C++ struct)
layout(push_constant) uniform PushConstants {
    uvec3 volumeDim;    // Actual dimensions of the input volume texture
    uvec3 blockDim;     // Should match local_size (e.g., 8,8,8)
    uvec3 blockGridDim; // Dimensions of the dispatch grid (and the output image)
    float isovalue;     // The target isovalue
} pc;

// --- Bindings ---

// Binding 0: Input Min/Max data image (produced by the first compute pass)
// Format must match what was written (e.g., rg32ui)
layout(binding = 0, rg32ui) uniform readonly uimage3D minMaxInputVolume;

// Binding 1: Output buffer to store the values read from the image
// Each element stores the uvec2 read from the corresponding image coordinate
layout(binding = 1, std430) buffer OutputValues {
    uvec2 values[]; // Matches the uvec2 format read from rg32ui image
};

// --- Workgroup Setup ---
// Launch 1D, matching the filtering shader's launch structure if desired
layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

void main() {
    // Calculate the 1D index of the block/texel this invocation reads
    uint globalBlockID1D = gl_GlobalInvocationID.x;

    // Calculate total number of texels/blocks in the input image
    uint totalBlocks = pc.blockGridDim.x * pc.blockGridDim.y * pc.blockGridDim.z;

    // --- Bounds Check ---
//    if (globalBlockID1D >= totalBlocks) {
//        return;
//    }

    // --- Read Input Image ---
    // Calculate 3D coordinate corresponding to the 1D index
    uvec3 blockCoord;
    uint planeSize = pc.blockGridDim.x * pc.blockGridDim.y;
    blockCoord.z = (planeSize > 0) ? (globalBlockID1D / planeSize) : 0;
    uint remainder = (planeSize > 0) ? (globalBlockID1D % planeSize) : globalBlockID1D;
    blockCoord.y = (pc.blockGridDim.x > 0) ? (remainder / pc.blockGridDim.x) : 0;
    blockCoord.x = (pc.blockGridDim.x > 0) ? (remainder % pc.blockGridDim.x) : remainder;

    // Read the uvec2 value from the input image
    uvec2 readValue = imageLoad(minMaxInputVolume, ivec3(blockCoord)).xy;

    // --- Write Output Buffer ---
    // Write the value read directly to the output buffer at the corresponding index
    values[globalBlockID1D] = readValue;
}