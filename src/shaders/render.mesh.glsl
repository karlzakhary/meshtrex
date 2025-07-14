#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier: require

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 126) out;

taskPayloadSharedEXT struct TaskPayload {
  uint meshletID;
} payload;

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

layout(set=0, binding = 0, std140) uniform SceneUBO {
    mat4 viewProjectionMatrix;
    vec3 cameraPos;
    float padding;
    vec4 lightPos;
} scene;


layout(set = 0, binding = 1, std430) buffer MeshletDescriptorBuffer { MeshletDescriptor descriptors[]; } meshlets;


layout(set = 0, binding = 2, std430) buffer VertexBuffer { VertexData data[]; } vertices;

layout(set = 0, binding = 3, std430) buffer IndexBuffer { uint data[]; } indices;

layout(location = 0) out PerVertexData {
    vec3 normal;
    vec3 worldPos;
} outData[];

void main() {
    uint localID = gl_LocalInvocationID.x;
    uint meshletID = payload.meshletID;

    uint vertCount = meshlets.descriptors[meshletID].vertexCount;
    uint primCount = meshlets.descriptors[meshletID].primitiveCount;
    uint vertOffset = meshlets.descriptors[meshletID].vertexOffset;
    uint indexOffset = meshlets.descriptors[meshletID].indexOffset;

    SetMeshOutputsEXT(vertCount, primCount);

    if (localID < vertCount) {
        vec4 pos = vertices.data[vertOffset + localID].position;
        vec4 norm = vertices.data[vertOffset + localID].normal;

        gl_MeshVerticesEXT[localID].gl_Position = scene.viewProjectionMatrix * pos;
        
        outData[localID].worldPos = pos.xyz;
        outData[localID].normal = normalize(norm.xyz);
    }

    if (localID < primCount) {
        uint baseIndex = indexOffset + localID * 3;

        // **FIX:** Convert global indices from the buffer to local indices for the meshlet.
        // The indices in the buffer are offsets into the *global* vertex buffer.
        // We subtract the meshlet's vertexOffset to make them local to the vertices
        // loaded by this shader invocation (i.e., in the range [0, vertCount-1]).
        uint i0 = indices.data[baseIndex + 0] - vertOffset;
        uint i1 = indices.data[baseIndex + 1] - vertOffset;
        uint i2 = indices.data[baseIndex + 2] - vertOffset;

        gl_PrimitiveTriangleIndicesEXT[localID] = uvec3(i0, i1, i2);
    }
}