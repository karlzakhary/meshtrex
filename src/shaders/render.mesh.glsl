#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_debug_printf : enable

// Mesh shader for rendering pipeline
// Generates primitives from meshlet data

// Mesh workgroup size
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// Maximum output vertices and primitives per meshlet
const uint MAX_VERTICES = 64;
const uint MAX_PRIMITIVES = 126;  // Nvidia recommendation for optimal performance

layout(max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;
layout(triangles) out;

// Vertex data structure from extraction
struct VertexData {
    vec4 position;
    vec4 normal;
};

// Meshlet descriptor structure
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

// Task shader payload
struct TaskPayload {
    uint meshletIndices[32];
    uint meshletCount;
};

taskPayloadSharedEXT TaskPayload taskPayload;

// Binding 0: Scene UBO
layout(set = 0, binding = 0, std140) uniform SceneUBO {
    mat4 viewProjectionMatrix;
    mat4 modelMatrix;
    vec3 cameraPos_world;
    float pad;
} scene;

// Binding 1: Meshlet descriptors (not used in mesh shader, but kept for consistency)
// The task shader already selected which meshlet we process

// Binding 2: Vertex buffer from extraction
layout(set = 0, binding = 2, std430) readonly buffer VertexBuffer {
    VertexData data[];
} vertices;

// Binding 3: Index buffer from extraction
layout(set = 0, binding = 3, std430) readonly buffer IndexBuffer {
    uint data[];
} indices;

// Binding 1: Meshlet descriptor buffer (needed to get the specific meshlet info)
layout(set = 0, binding = 1, std430) readonly buffer MeshletDescriptors {
    MeshletDescriptor descriptors[];
} meshlets;

// Output to fragment shader
layout(location = 0) out vec3 outNormal_world[];
layout(location = 1) out vec3 outPosition_world[];

void main() {
    uint meshletIndex = taskPayload.meshletIndices[gl_WorkGroupID.x];
    uint tid = gl_LocalInvocationID.x;
    
    // Fetch the meshlet descriptor
    MeshletDescriptor meshlet = meshlets.descriptors[meshletIndex];
    
    // Early out if this thread has no work
    if (tid >= meshlet.vertexCount && tid >= meshlet.primitiveCount) {
        return;
    }
    
    // Process vertices (each thread handles one vertex if within range)
    if (tid < meshlet.vertexCount) {
        uint globalVertexIndex = meshlet.vertexOffset + tid;
        VertexData v = vertices.data[globalVertexIndex];
        
        // Transform vertex to clip space
        vec4 worldPos = scene.modelMatrix * v.position;
        gl_MeshVerticesEXT[tid].gl_Position = scene.viewProjectionMatrix * worldPos;
        
        // Transform normal to world space (using normal matrix)
        mat3 normalMatrix = transpose(inverse(mat3(scene.modelMatrix)));
        vec3 worldNormal = normalize(normalMatrix * v.normal.xyz);
        
        // Output per-vertex attributes
        outNormal_world[tid] = worldNormal;
        outPosition_world[tid] = worldPos.xyz;
    }
    
    // Process primitives (each thread handles one triangle if within range)
    if (tid < meshlet.primitiveCount) {
        uint indexOffset = meshlet.indexOffset + tid * 3;
        
        // Read triangle indices
        uint i0 = indices.data[indexOffset + 0];
        uint i1 = indices.data[indexOffset + 1];
        uint i2 = indices.data[indexOffset + 2];
        
        // Convert to mesh-local indices (0-based within this meshlet)
        uint idx0 = i0 - meshlet.vertexOffset;
        uint idx1 = i1 - meshlet.vertexOffset;
        uint idx2 = i2 - meshlet.vertexOffset;
        
        // Write triangle indices
        // In EXT_mesh_shader, we write entire triangles as uvec3
        gl_PrimitiveTriangleIndicesEXT[tid] = uvec3(idx0, idx1, idx2);
    }
    
    // Set output counts (only thread 0)
    if (tid == 0) {
        SetMeshOutputsEXT(meshlet.vertexCount, meshlet.primitiveCount);
        // Debug output
        // debugPrintfEXT("Mesh shader: Rendering meshlet %u with %u vertices and %u triangles\n",
        //                meshletIndex, meshlet.vertexCount, meshlet.primitiveCount);
    }
}