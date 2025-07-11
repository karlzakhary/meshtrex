#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable

// Mesh shader for rendering culled persistent geometry
// Reads from global buffers and outputs final triangles

#define MAX_VERTICES_PER_MESHLET 64
#define MAX_PRIMITIVES_PER_MESHLET 64
#define MAX_MESHLETS_PER_WORKGROUP 8

// --- Output Layout ---
layout(triangles) out;
layout(max_vertices = MAX_VERTICES_PER_MESHLET, max_primitives = MAX_PRIMITIVES_PER_MESHLET) out;

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

struct VertexData {
    vec4 position;
    vec4 normal;
};

struct CameraData {
    mat4 viewMatrix;
    mat4 projMatrix;
    mat4 viewProjMatrix;
    vec4 frustumPlanes[6];
    vec3 cameraPosition;
    float padding;
};

taskPayloadSharedEXT struct TaskPayload {
    uint meshletIndices[MAX_MESHLETS_PER_WORKGROUP];
    uint meshletCount;
} payload;

// --- Push Constants ---
layout(push_constant) uniform PushConstants {
    uint totalMeshlets;
    uint meshletsPerWorkgroup;
    float cullDistance;
    uint enableFrustumCulling;
    uint enableBackfaceCulling;
    uint enableOcclusionCulling;
    uint padding[2];
} pc;

// --- Descriptor Bindings ---
layout(set = 0, binding = 0, std140) uniform CameraBuffer {
    CameraData camera;
};

layout(set = 0, binding = 1, std430) readonly buffer MeshletDescriptorBuffer {
    MeshletDescriptor meshlets[];
};

layout(set = 0, binding = 2, std430) readonly buffer VertexBuffer {
    VertexData vertices[];
};

layout(set = 0, binding = 3, std430) readonly buffer IndexBuffer {
    uint indices[];
};

// --- Output Attributes ---
layout(location = 0) out VertexOutput {
    vec3 worldPos;
    vec3 normal;
    vec3 viewPos;
} vertOut[];

layout(location = 3) out flat uint primitiveID[];

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint meshletIdx = gl_WorkGroupID.x;
    if (meshletIdx >= payload.meshletCount) {
        return;
    }
    
    uint meshletIndex = payload.meshletIndices[meshletIdx];
    MeshletDescriptor meshlet = meshlets[meshletIndex];
    
    // Early out if this thread won't process anything
    uint tid = gl_LocalInvocationIndex;
    if (tid >= meshlet.vertexCount && tid >= meshlet.primitiveCount) {
        return;
    }
    
    // Process vertices
    if (tid < meshlet.vertexCount) {
        uint vertexIndex = meshlet.vertexOffset + tid;
        VertexData vertex = vertices[vertexIndex];
        
        // Transform to clip space
        vec4 worldPos = vertex.position;
        vec4 clipPos = camera.viewProjMatrix * worldPos;
        
        // Output vertex
        gl_MeshVerticesEXT[tid].gl_Position = clipPos;
        vertOut[tid].worldPos = worldPos.xyz;
        vertOut[tid].normal = vertex.normal.xyz;
        vertOut[tid].viewPos = (camera.viewMatrix * worldPos).xyz;
    }
    
    // Process primitives
    if (tid < meshlet.primitiveCount) {
        uint indexBase = meshlet.indexOffset + tid * 3;
        
        // Read triangle indices (relative to meshlet)
        uvec3 tri;
        tri.x = indices[indexBase + 0];
        tri.y = indices[indexBase + 1];
        tri.z = indices[indexBase + 2];
        
        // Output primitive
        gl_PrimitiveTriangleIndicesEXT[tid] = tri;
        primitiveID[tid] = meshletIndex * MAX_PRIMITIVES_PER_MESHLET + tid;
    }
    
    // Set output counts
    if (tid == 0) {
        SetMeshOutputsEXT(meshlet.vertexCount, meshlet.primitiveCount);
    }
}