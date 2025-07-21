#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

// Push constants containing dimensions
layout(push_constant) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} pc;

// Define the local workgroup size
// Optimized: 64 threads for better utilization with 3x3x3 blocks (4x4x4 = 64 voxels)
layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Binding 0: Input scalar field data
layout(binding = 0, r8ui) uniform readonly uimage3D volume;

// Binding 1: Output min/max pairs
layout(binding = 1, rg32ui) uniform writeonly uimage3D minMaxOutputVolume;

// Shared memory for intermediate min/max values per subgroup
const uint totalInvocations = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;

shared uvec2 s_subgroupMinMax[totalInvocations / 32]; // Assuming min subgroup size of 32

void main() {
    const uvec3 workGroupID = gl_WorkGroupID;
    const uvec3 localInvocationID = gl_LocalInvocationID;
    const uint localInvocationIndex = gl_LocalInvocationIndex;
    const uint subgroupId = gl_SubgroupID;
    const uint subgroupInvocationId = gl_SubgroupInvocationID;
    
    // Calculate the starting position of this block of CELLS in the volume
    const ivec3 blockStart = ivec3(workGroupID * pc.blockDim.xyz);
    
    // Initialize min/max values
    uint invocationMin = 0xFFFFFFFFu;
    uint invocationMax = 0u;
    
    // Calculate total voxels needed for this block (blockDim + 1 in each dimension for marching cubes)
    const uvec3 voxelsPerBlock = pc.blockDim.xyz + uvec3(1);
    const uint totalVoxelsNeeded = voxelsPerBlock.x * voxelsPerBlock.y * voxelsPerBlock.z;
    
    // Each thread will process multiple voxels to cover all needed voxels
    const uint voxelsPerThread = (totalVoxelsNeeded + totalInvocations - 1) / totalInvocations;
    
    // OPTIMIZATION 1: Pre-calculate constants to avoid repeated division
    const uint voxelStrideY = voxelsPerBlock.x;
    const uint voxelStrideZ = voxelsPerBlock.x * voxelsPerBlock.y;
    
    // Process voxels assigned to this thread
    for (uint i = 0; i < voxelsPerThread; ++i) {
        uint voxelIndex = localInvocationIndex * voxelsPerThread + i;
        
        if (voxelIndex < totalVoxelsNeeded) {
            // OPTIMIZATION 2: Use bit operations where possible for power-of-2 dimensions
            // Convert linear index to 3D coordinates within the block
            uvec3 localVoxel;
            localVoxel.z = voxelIndex / voxelStrideZ;
            uint temp = voxelIndex - (localVoxel.z * voxelStrideZ);
            localVoxel.y = temp / voxelStrideY;
            localVoxel.x = temp - (localVoxel.y * voxelStrideY);
            
            // Calculate global voxel coordinate
            ivec3 voxelCoord = blockStart + ivec3(localVoxel);
            
            // OPTIMIZATION 3: Single bounds check for common case
            if (all(lessThan(voxelCoord, ivec3(pc.volumeDim.xyz)))) {
                uint value = imageLoad(volume, voxelCoord).x;
                invocationMin = min(invocationMin, value);
                invocationMax = max(invocationMax, value);
            }
        }
    }
    
    // --- Step 1: Subgroup-level reduction ---
    // OPTIMIZATION 4: Use native subgroup operations if available (usually faster)
    #ifdef GL_KHR_shader_subgroup_arithmetic
        invocationMin = subgroupMin(invocationMin);
        invocationMax = subgroupMax(invocationMax);
    #else
        // Fallback to manual reduction
        for (uint offset = gl_SubgroupSize / 2; offset > 0; offset /= 2) {
            uint downMin = subgroupShuffleDown(invocationMin, offset);
            uint downMax = subgroupShuffleDown(invocationMax, offset);
            invocationMin = min(invocationMin, downMin);
            invocationMax = max(invocationMax, downMax);
        }
    #endif
    
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
            imageStore(minMaxOutputVolume, ivec3(workGroupID), uvec4(subgroupResult, 0, 0));
        }
    }
}