#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_debug_printf: require

layout(local_size_x = 32) in;

// Input: Meshlet descriptors from the extraction stage
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};
layout(set = 0, binding = 1, std430) buffer MeshletDescriptorBuffer { MeshletDescriptor descriptors[]; } meshlets;


// **Input: The total count of meshlets generated**
layout(set=0, binding = 4, std430) readonly buffer MeshletDescriptorCount {
    uint meshletCounter;
} meshletCount;

// Define the payload to pass to the mesh shader
taskPayloadSharedEXT struct TaskPayload{
  uint meshletID;
} payload;

void main() {
    uint currentMeshletID = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    // Check against the actual number of meshlets from the counter buffer
    if (currentMeshletID < meshletCount.meshletCounter) {
        payload.meshletID = currentMeshletID;
        EmitMeshTasksEXT(1, 1, 1);
    }
}