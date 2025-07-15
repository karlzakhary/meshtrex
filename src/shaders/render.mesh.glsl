#version 450
#extension GL_NV_mesh_shader : require

#define MAX_VERTS 64u
#define MAX_PRIMS 126u

layout(local_size_x = 32) in;
layout(triangles, max_vertices = MAX_VERTS, max_primitives = MAX_PRIMS) out;

struct VertexData {
    vec4 position;
    vec4 normal;
};

layout(binding = 0) readonly buffer VertexBuffer {
    VertexData vertices[];
} vertexBuffer;

layout(binding = 1) readonly buffer IndexBuffer {
    uint indices[];
} indexBuffer;

struct MeshletDescriptor {
   uint vertexOffset;
   uint indexOffset;
   uint vertexCount;
   uint primitiveCount;
};

layout(binding = 2) readonly buffer MeshletDescriptorBuffer {
    MeshletDescriptor meshlets[];
} meshletDescriptorBuffer;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pc;

layout(location = 0) out PerVertexData {
    vec3 normal;
} outData[];


void main()
{
    uint meshletID = gl_WorkGroupID.x;
    MeshletDescriptor meshlet = meshletDescriptorBuffer.meshlets[meshletID];

    uint vertexCount = meshlet.vertexCount;
    uint triangleCount = meshlet.primitiveCount;

    // Reject invalid meshlets
    if (triangleCount == 0 || vertexCount > 64 || triangleCount > 126) {
        gl_PrimitiveCountNV = 0;
        return;
    }

    gl_PrimitiveCountNV = triangleCount;

    // Load and transform vertices
    for (uint i = gl_LocalInvocationID.x; i < vertexCount; i += 32)
    {
        uint globalVertexIndex = meshlet.vertexOffset + i;
        VertexData v = vertexBuffer.vertices[globalVertexIndex];
        
        gl_MeshVerticesNV[i].gl_Position = pc.viewProj * v.position;
        outData[i].normal = v.normal.xyz;
    }

    // Set indices for the meshlet
    uint indexCount = triangleCount * 3;
    for (uint i = gl_LocalInvocationID.x; i < indexCount; i += 32)
    {
        uint globalIndex = meshlet.indexOffset + i;
        // Indices in the buffer are global, they need to be made local to the meshlet's vertex set
        uint localIndex = indexBuffer.indices[globalIndex] - meshlet.vertexOffset;
        gl_PrimitiveIndicesNV[i] = localIndex;
    }
}