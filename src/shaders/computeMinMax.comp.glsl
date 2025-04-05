#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Binding 0: volume data (VK_FORMAT_R8_UNORM → normalized to [0,1])
layout(binding = 0, r8) uniform readonly image3D volume;

// Binding 1: output buffer storing vec2(min, max) per block
layout(binding = 1, std430) buffer MinMaxBuffer {
    vec2 minMax[];
};

// Push constants
layout(push_constant) uniform PushConstants {
    ivec3 volumeDim;     // e.g. (256, 256, 225)
    ivec3 blockDim;      // e.g. (8, 8, 8)
    ivec3 blockGridDim;  // e.g. (32, 32, 29)
} pc;

// Shared memory to store local thread values
shared float sharedMin[512];  // 8x8x8 = 512 threads
shared float sharedMax[512];

void main() {
    uvec3 localID = gl_LocalInvocationID;
    uvec3 groupID = gl_WorkGroupID;
    uint localIndex = gl_LocalInvocationIndex;

    // Compute voxel coordinate
    uvec3 voxelCoord = groupID * pc.blockDim + localID;

    // Load and clamp
    float value = 0.25 + gl_LocalInvocationIndex * 0.001;
//    if (all(lessThan(voxelCoord, uvec3(pc.volumeDim)))) {
//        value = imageLoad(volume, ivec3(voxelCoord)).r;
//    }

    // Store each thread's value in shared memory
    sharedMin[localIndex] = value;
    sharedMax[localIndex] = value;
    memoryBarrierShared();
    barrier();

    // Perform parallel reduction for min and max
    for (uint offset = 256; offset > 0; offset >>= 1) {
        if (localIndex < offset && (localIndex + offset) < 512) {
            sharedMin[localIndex] = min(sharedMin[localIndex], sharedMin[localIndex + offset]);
            sharedMax[localIndex] = max(sharedMax[localIndex], sharedMax[localIndex + offset]);
        }
        memoryBarrierShared();
        barrier();
    }

    // Thread 0 writes the final result
    if (localIndex == 0) {
        uint flatIndex = groupID.z * pc.blockGridDim.x * pc.blockGridDim.y +
        groupID.y * pc.blockGridDim.x + groupID.x;
        minMax[flatIndex] = vec2(sharedMin[0], sharedMax[0]);
    }
}