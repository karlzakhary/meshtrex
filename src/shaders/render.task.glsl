#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_debug_printf : enable

// Task shader for rendering pipeline
// Dispatches mesh shader workgroups based on meshlet descriptors

// Task workgroup size - processes multiple meshlets
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// Maximum mesh shader workgroups this task shader can emit
const uint MAX_MESH_WORKGROUPS = 32;

// Meshlet descriptor structure matching extraction output
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

// Binding 0: Scene UBO (for potential frustum culling)
layout(set = 0, binding = 0, std140) uniform SceneUBO {
    mat4 viewProjectionMatrix;
    mat4 modelMatrix;
    vec3 cameraPos_world;
    float pad;
} scene;

// Binding 1: Meshlet descriptor buffer from extraction
layout(set = 0, binding = 1, std430) readonly buffer MeshletDescriptors {
    MeshletDescriptor descriptors[];
} meshlets;

// Task shader output payload to mesh shader
struct TaskPayload {
    uint meshletIndices[MAX_MESH_WORKGROUPS];
    uint meshletCount;
};

taskPayloadSharedEXT TaskPayload taskPayload;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationID.x;
    
    // Initialize shared payload
    if (tid == 0) {
        taskPayload.meshletCount = 0;
    }
    barrier();
    
    // Each thread checks one meshlet
    // In a real implementation, you'd do frustum culling here
    if (gid < meshlets.descriptors.length()) {
        MeshletDescriptor meshlet = meshlets.descriptors[gid];
        
        // Simple visibility check - skip empty meshlets
        bool visible = (meshlet.vertexCount > 0 && meshlet.primitiveCount > 0);
        
        // TODO: Add frustum culling using scene.viewProjectionMatrix
        // For now, all non-empty meshlets are considered visible
        
        if (visible) {
            // Atomically add this meshlet to the output list
            uint slot = atomicAdd(taskPayload.meshletCount, 1);
            if (slot < MAX_MESH_WORKGROUPS) {
                taskPayload.meshletIndices[slot] = gid;
            }
        }
    }
    
    barrier();
    
    // Thread 0 emits the mesh shader workgroups
    if (tid == 0) {
        uint meshWorkgroups = min(taskPayload.meshletCount, MAX_MESH_WORKGROUPS);
        debugPrintfEXT("Task shader: Dispatching %u mesh workgroups from %u visible meshlets", meshWorkgroups, taskPayload.meshletCount);
        EmitMeshTasksEXT(meshWorkgroups, 1, 1);
    }
}