#version 450
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 32) in;

// Push constants
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} pc;

// Bindings
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount { 
    uint count; 
} activeBlockCount;

layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs { 
    uint blockIds[]; 
} compactedBlocks;

// Task payload shared between task and mesh shaders
struct TaskPayload {
    uint blockIds[32]; // Up to 32 blocks per task shader workgroup
};
taskPayloadSharedEXT TaskPayload payload;

void main() {
    uint taskID = gl_GlobalInvocationID.x;
    uint blocksPerTask = 1; // Process 1 block per mesh shader for simplicity
    uint startBlockIdx = taskID * blocksPerTask;
    
    uint meshCount = 0;
    
    // Check if we have blocks to process
    if (startBlockIdx < activeBlockCount.count) {
        // Determine how many blocks this task will process
        uint endBlockIdx = min(startBlockIdx + blocksPerTask, activeBlockCount.count);
        meshCount = endBlockIdx - startBlockIdx;
        
        // Copy block IDs to task payload
        for (uint i = 0; i < meshCount; ++i) {
            payload.blockIds[i] = compactedBlocks.blockIds[startBlockIdx + i];
        }
    }
    
    // Dispatch mesh shaders (one per block)
    EmitMeshTasksEXT(meshCount, 1, 1);
}