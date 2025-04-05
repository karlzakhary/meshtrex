#version 460

#extension GL_EXT_shader_atomic_float2: require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Binding 0: input 3D volume (e.g., VK_FORMAT_R8_UNORM)
layout(binding = 0, r8) uniform readonly image3D volume;

// Binding 1: output buffer, 1 vec2(min, max) per block
layout(std430, binding = 1) buffer MinMaxBuffer {
    vec2 minMax[];
};

// Push constants for dimensions
layout(push_constant) uniform PushConstants {
    ivec3 volumeDim;     // e.g. 256, 256, 225
    ivec3 blockDim;      // e.g. 8, 8, 8
    ivec3 blockGridDim;  // e.g. 32, 32, 29
} pc;

// Shared memory for parallel reduction
shared float s_min;
shared float s_max;

void main() {
    uvec3 localID = gl_LocalInvocationID;
    uvec3 blockID = gl_WorkGroupID;

    uvec3 blockOffset = blockID * pc.blockDim;
    uvec3 voxelCoord = blockOffset + localID;

    // Handle out-of-bound threads
    if (any(greaterThanEqual(voxelCoord, uvec3(pc.volumeDim)))) {
        return;
    }

    float v = imageLoad(volume, ivec3(voxelCoord)).r;

    // First thread initializes shared memory
    if (gl_LocalInvocationIndex == 0) {
        s_min = v;
        s_max = v;
    }
    memoryBarrierShared();
    barrier();

    // Parallel min/max update
    atomicMin(s_min, v);
    atomicMax(s_max, v);

    barrier();

    // Write result
    if (gl_LocalInvocationIndex == 0) {
        uint flatBlockIndex = blockID.z * pc.blockGridDim.x * pc.blockGridDim.y +
        blockID.y * pc.blockGridDim.x + blockID.x;
        minMax[flatBlockIndex] = vec2(s_min, s_max);
    }
}