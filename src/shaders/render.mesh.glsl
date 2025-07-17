#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_debug_printf: require

#define MAX_VERTS 64
#define MAX_PRIMS 126

layout(local_size_x = 32) in;
layout(triangles, max_vertices = MAX_VERTS, max_primitives = MAX_PRIMS) out;

// --- Input Buffers ---
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

// --- Push Constants & Outputs ---
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pc;

layout(location = 0) out PerVertexData {
    vec3 normal;
} outData[];


// --- Task Shader Input (Payload) ---
#define MAX_VERTS_PER_SUB_MESHLET 64
#define MAX_SUB_MESHLETS 8

struct SubMeshlet {
    uint originalVertexIndices[MAX_VERTS_PER_SUB_MESHLET];
    uint firstTriangleIndex;
    uint vertexCount;
    uint primitiveCount;
};

// This struct must exactly match the one in the task shader.
struct TaskData {
    uint largeMeshletID;
    SubMeshlet subMeshlets[MAX_SUB_MESHLETS];
};

// CORRECTED DECLARATION: This must match the task shader's declaration exactly.
// The previous 'TaskPayloadBlock' wrapper was incorrect.
taskPayloadSharedEXT TaskData payload;

// --- Shared Memory for Fast Remapping ---
shared uint shOriginalVertexIndices[MAX_VERTS];

void main()
{
    // The mesh shader's gl_WorkGroupID.x IS the index of the sub-meshlet.
    uint subMeshletID = gl_WorkGroupID.x;
    
    // Get the data for this specific sub-meshlet.
    SubMeshlet subMeshlet = payload.subMeshlets[subMeshletID];

    uint vertexCount = subMeshlet.vertexCount;
    uint triangleCount = subMeshlet.primitiveCount;

    SetMeshOutputsEXT(vertexCount, triangleCount);

    // --- Step 1: Load vertices ---
    for (uint i = gl_LocalInvocationID.x; i < vertexCount; i += 32)
    {
        uint globalVertexIndex = subMeshlet.originalVertexIndices[i];
        VertexData v = vertexBuffer.vertices[globalVertexIndex];

        gl_MeshVerticesEXT[i].gl_Position = pc.viewProj * v.position;
        outData[i].normal = v.normal.xyz;
    }

    // --- Step 2: Load and remap indices ---
    for (uint i = gl_LocalInvocationID.x; i < vertexCount; i += 32) {
        shOriginalVertexIndices[i] = subMeshlet.originalVertexIndices[i];
    }
    barrier();

    // Get the original large meshlet descriptor using the ID from the payload.
    MeshletDescriptor largeMeshlet = meshletDescriptorBuffer.meshlets[payload.largeMeshletID];
    uint globalIndexBase = largeMeshlet.indexOffset + (subMeshlet.firstTriangleIndex * 3);

    // The output primitive indices are written per-triangle (uvec3)
    for (uint i = gl_LocalInvocationID.x; i < triangleCount; i += 32)
    {
        uint i0 = indexBuffer.indices[globalIndexBase + i * 3 + 0];
        uint i1 = indexBuffer.indices[globalIndexBase + i * 3 + 1];
        uint i2 = indexBuffer.indices[globalIndexBase + i * 3 + 2];

        uint local_i0 = 0xFFFFFFFF;
        uint local_i1 = 0xFFFFFFFF;
        uint local_i2 = 0xFFFFFFFF;

        // Find the new local indices by searching the shared memory list.
        for (uint k = 0; k < vertexCount; ++k) {
            if (shOriginalVertexIndices[k] == i0) local_i0 = k;
            if (shOriginalVertexIndices[k] == i1) local_i1 = k;
            if (shOriginalVertexIndices[k] == i2) local_i2 = k;
        }
        
        // Robustness check: If any index wasn't found, output a degenerate triangle.
        if (local_i0 == 0xFFFFFFFF || local_i1 == 0xFFFFFFFF || local_i2 == 0xFFFFFFFF) {
             gl_PrimitiveTriangleIndicesEXT[i] = uvec3(0,0,0);
        } else {
             gl_PrimitiveTriangleIndicesEXT[i] = uvec3(local_i0, local_i1, local_i2);
        }
    }
}
