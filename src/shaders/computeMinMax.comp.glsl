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
layout (local_size_x = 8, local_size_y = 8, local_size_z = 2) in;

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
    
    // Strategy: Each thread checks exactly 2 voxels to cover all 125 voxels
    // 64 threads × 2 voxels = 128 > 125 needed
    
    // First voxel: Regular 4x4x4 grid
    {
        ivec3 voxelCoord = blockStart + ivec3(localInvocationID);
        if (all(lessThan(voxelCoord, ivec3(pc.volumeDim.xyz)))) {
            uint value = imageLoad(volume, voxelCoord).x;
            invocationMin = min(invocationMin, value);
            invocationMax = max(invocationMax, value);
        }
    }
    
    // Second voxel: Handle the boundary layer (position 4 in each dimension)
    // We need to cover 64 additional voxels (128 - 64 = 64)
    if (localInvocationIndex < 61) {
        ivec3 extraVoxel;
        
        if (localInvocationIndex < 16) {
            // Handle x=4 face: (4, y, z) where y,z ∈ [0,3]
            uint y = localInvocationIndex % pc.blockDim.x;
            uint z = localInvocationIndex / pc.blockDim.z;
            extraVoxel = blockStart + ivec3(pc.blockDim.x, y, z);
        }
        else if (localInvocationIndex < 32) {
            // Handle y=4 face: (x, 4, z) where x ∈ [0,3], z ∈ [0,3]
            uint idx = localInvocationIndex - 16;
            uint x = idx % pc.blockDim.x;
            uint z = idx / pc.blockDim.z;
            extraVoxel = blockStart + ivec3(x, pc.blockDim.y, z);
        }
        else if (localInvocationIndex < 48) {
            // Handle z=4 face: (x, y, 4) where x,y ∈ [0,3]
            uint idx = localInvocationIndex - 32;
            uint x = idx % pc.blockDim.x;
            uint y = idx / pc.blockDim.y;
            extraVoxel = blockStart + ivec3(x, y, pc.blockDim.z);
        }
        else if (localInvocationIndex < 52) {
            // Handle x=4, y=4 edge: (4, 4, z) where z ∈ [0,3]
            uint z = localInvocationIndex - 48;
            extraVoxel = blockStart + ivec3(pc.blockDim.x, pc.blockDim.y, z);
        }
        else if (localInvocationIndex < 56) {
            // Handle x=4, z=4 edge: (4, y, 4) where y ∈ [0,3]
            uint y = localInvocationIndex - 52;
            extraVoxel = blockStart + ivec3(pc.blockDim.x, y, pc.blockDim.z);
        }
        else if (localInvocationIndex < 60) {
            // Handle y=4, z=4 edge: (x, 4, 4) where x ∈ [0,3]
            uint x = localInvocationIndex - 56;
            extraVoxel = blockStart + ivec3(x, pc.blockDim.y, pc.blockDim.z);
        }
        else {
            // Handle the corner: (4, 4, 4)
            extraVoxel = blockStart + ivec3(pc.blockDim.xyz);
        }
        
        if (all(lessThan(extraVoxel, ivec3(pc.volumeDim.xyz)))) {
            uint value = imageLoad(volume, extraVoxel).x;
            invocationMin = min(invocationMin, value);
            invocationMax = max(invocationMax, value);
        }
    }
    
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
            imageStore(minMaxOutputVolume, ivec3(workGroupID), uvec4(subgroupResult, 0, 0));
        }
    }
}