// marching_cubes_mesh.glsl
#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_debug_printf : require

layout(local_size_x = 32) in;
layout(triangles) out;
layout(max_vertices = 256, max_primitives = 128) out;

// Vertex structure
struct VertexData {
    vec4 position;
    // vec3 normal;
};

// Meshlet descriptor
struct MeshletDescriptor {
    uint vertexOffset;
    uint vertexCount;
    uint indexOffset;
    uint indexCount;
};

// Push constants
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} pc;

// Task payload shared between task and mesh shaders
struct TaskPayload {
    uint blockIds[32];
};
taskPayloadSharedEXT TaskPayload payload;

// Bindings
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable { 
    int triTable[256 * 16]; 
} mcTriTable;
layout(set = 0, binding = 14, std430) readonly buffer MarchingCubesEdgeTable { 
    uint edgeTable[256]; 
} mcEdgeTable;

layout(set = 0, binding = 6, std430) writeonly buffer VertexBuffer {
    VertexData vertices[];
} vertexBuffer;

layout(set = 0, binding = 8, std430) writeonly buffer IndexBuffer {
    uint indices[];
} indexBuffer;

layout(set = 0, binding = 10, std430) writeonly buffer MeshletDescriptors {
    MeshletDescriptor descriptors[];
} meshletDesc;

layout(set = 0, binding = 11, std430) buffer VertexCount { 
    uint vertexCount; 
} vertexCount;

layout(set = 0, binding = 12, std430) buffer MeshletDescCount {
    uint meshletCounter;
} meshletDescCount;

layout(set = 0, binding = 13, std430) buffer IndexCount { 
    uint indexCount; 
} indexCount;

// Mesh shader outputs (not used for rasterization, but required by spec)
layout(location = 0) out vec3 fragPos[];
// layout(location = 1) out vec3 fragNormal[];

// Shared memory for vertices and indices within this mesh shader
shared vec4 s_vertices[256];
// shared vec3 s_normals[256];
shared uint s_indices[384]; // 128 triangles * 3 vertices
shared uint s_vertexCount;
shared uint s_indexCount;
shared uint s_globalVertexOffset;
shared uint s_globalIndexOffset;
shared uint s_meshletIndex;

// Edge vertex positions relative to cube corners
const vec3 EDGE_VERTICES[12] = vec3[12](
    vec3(0.5, 0.0, 0.0), vec3(1.0, 0.5, 0.0), vec3(0.5, 1.0, 0.0), vec3(0.0, 0.5, 0.0),
    vec3(0.5, 0.0, 1.0), vec3(1.0, 0.5, 1.0), vec3(0.5, 1.0, 1.0), vec3(0.0, 0.5, 1.0),
    vec3(0.0, 0.0, 0.5), vec3(1.0, 0.0, 0.5), vec3(1.0, 1.0, 0.5), vec3(0.0, 1.0, 0.5)
);

// Corner offsets
const ivec3 CORNER_OFFSETS[8] = ivec3[8](
    ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(1, 1, 0), ivec3(0, 1, 0),
    ivec3(0, 0, 1), ivec3(1, 0, 1), ivec3(1, 1, 1), ivec3(0, 1, 1)
);

// Edge connections (which corners form each edge)
const uvec2 EDGE_CONNECTIONS[12] = uvec2[12](
    uvec2(0, 1), uvec2(1, 2), uvec2(2, 3), uvec2(3, 0),
    uvec2(4, 5), uvec2(5, 6), uvec2(6, 7), uvec2(7, 4),
    uvec2(0, 4), uvec2(1, 5), uvec2(2, 6), uvec2(3, 7)
);

// Helper function to sample volume
float sampleVolume(ivec3 coord) {
    if (coord.x >= 0 && coord.x < int(pc.volumeDim.x) &&
        coord.y >= 0 && coord.y < int(pc.volumeDim.y) &&
        coord.z >= 0 && coord.z < int(pc.volumeDim.z)) {
        return float(imageLoad(volumeImage, coord).x);
    }
    return 0.0;
}

// Calculate gradient for normal
vec3 calculateGradient(vec3 pos) {
    vec3 gradient;
    gradient.x = sampleVolume(ivec3(pos + vec3(1, 0, 0))) - sampleVolume(ivec3(pos - vec3(1, 0, 0)));
    gradient.y = sampleVolume(ivec3(pos + vec3(0, 1, 0))) - sampleVolume(ivec3(pos - vec3(0, 1, 0)));
    gradient.z = sampleVolume(ivec3(pos + vec3(0, 0, 1))) - sampleVolume(ivec3(pos - vec3(0, 0, 1)));
    return normalize(gradient);
}

// Vertex interpolation along edge
vec3 interpolateVertex(vec3 p1, vec3 p2, float v1, float v2) {
    float t = (pc.isovalue - v1) / (v2 - v1);
    return mix(p1, p2, t);
}

void main() {
    uint meshID = gl_WorkGroupID.x;
    uint threadID = gl_LocalInvocationID.x;
    
    // Initialize shared counters
    if (threadID == 0) {
        s_vertexCount = 0;
        s_indexCount = 0;
    }
    barrier();
    
    // Get the block ID for this mesh shader
    uint blockID1D = payload.blockIds[meshID];
    
    // Convert 1D block ID to 3D block coordinates
    uvec3 blockCoord;
    uint planeSize = pc.blockGridDim.x * pc.blockGridDim.y;
    blockCoord.z = blockID1D / planeSize;
    uint remainder = blockID1D % planeSize;
    blockCoord.y = remainder / pc.blockGridDim.x;
    blockCoord.x = remainder % pc.blockGridDim.x;
    
    // Calculate base position of this block in world coordinates
    ivec3 blockBase = ivec3(blockCoord * pc.blockDim.xyz);
    
    // Each thread processes multiple cells within the block
    uint cellsPerThread = (pc.blockDim.x * pc.blockDim.y * pc.blockDim.z) / 32;
    uint startCell = threadID * cellsPerThread;
    uint endCell = min(startCell + cellsPerThread, pc.blockDim.x * pc.blockDim.y * pc.blockDim.z);
    
    // Local vertex map to avoid duplicates (simplified - in practice you'd want a more sophisticated approach)
    uint localVertices[12];
    
    for (uint cellIdx = startCell; cellIdx < endCell; ++cellIdx) {
        // Convert cell index to 3D coordinates within block
        uvec3 cellCoord;
        cellCoord.z = cellIdx / (pc.blockDim.x * pc.blockDim.y);
        uint planeRemainder = cellIdx % (pc.blockDim.x * pc.blockDim.y);
        cellCoord.y = planeRemainder / pc.blockDim.x;
        cellCoord.x = planeRemainder % pc.blockDim.x;
        
        // Skip cells at block boundaries (they need neighbor data)
        if (cellCoord.x >= pc.blockDim.x - 1 || 
            cellCoord.y >= pc.blockDim.y - 1 || 
            cellCoord.z >= pc.blockDim.z - 1) {
            continue;
        }
        
        ivec3 cellBase = blockBase + ivec3(cellCoord);
        
        // Sample corner values
        float cornerValues[8];
        uint cubeIndex = 0;
        for (uint i = 0; i < 8; ++i) {
            cornerValues[i] = sampleVolume(cellBase + CORNER_OFFSETS[i]);
            if (cornerValues[i] < pc.isovalue) {
                cubeIndex |= (1u << i);
            }
        }
        
        // Skip if no triangles
        uint edgeMask = mcEdgeTable.edgeTable[cubeIndex];
        if (edgeMask == 0) continue;
        
        // Generate vertices for active edges
        vec3 edgeVertices[12];
        for (uint i = 0; i < 12; ++i) {
            localVertices[i] = 0xFFFFFFFF; // Invalid marker
        }
        
        // Edge table lookup to generate vertices
        for (uint edge = 0; edge < 12; ++edge) {
            if ((edgeMask & (1u << edge)) != 0) {
                // Get corner indices for this edge
                uint c1 = EDGE_CONNECTIONS[edge].x;
                uint c2 = EDGE_CONNECTIONS[edge].y;
                
                vec3 p1 = vec3(cellBase + CORNER_OFFSETS[c1]);
                vec3 p2 = vec3(cellBase + CORNER_OFFSETS[c2]);
                
                edgeVertices[edge] = interpolateVertex(p1, p2, cornerValues[c1], cornerValues[c2]);
                
                // Add vertex to shared memory
                uint vertexIdx = atomicAdd(s_vertexCount, 1);
                localVertices[edge] = vertexIdx;
                s_vertices[vertexIdx] = vec4(edgeVertices[edge], 1.0);
                // s_normals[vertexIdx] = calculateGradient(edgeVertices[edge]);
            }
        }
        
        // Generate triangles
        int triTableBase = int(cubeIndex) * 16;
        for (int i = 0; mcTriTable.triTable[triTableBase + i] != -1; i += 3) {
            uint idx = atomicAdd(s_indexCount, 3);
            s_indices[idx + 0] = localVertices[mcTriTable.triTable[triTableBase + i + 0]];
            s_indices[idx + 1] = localVertices[mcTriTable.triTable[triTableBase + i + 1]];
            s_indices[idx + 2] = localVertices[mcTriTable.triTable[triTableBase + i + 2]];
        }
    }
    
    barrier();
    
    // Allocate global memory for vertices and indices
    if (threadID == 0 && s_vertexCount > 0) {
        s_globalVertexOffset = atomicAdd(vertexCount.vertexCount, s_vertexCount);
        s_globalIndexOffset = atomicAdd(indexCount.indexCount, s_indexCount);
        s_meshletIndex = atomicAdd(meshletDescCount.meshletCounter, 1);
    }
    
    barrier();
    
    // Write vertices to global memory
    for (uint i = threadID; i < s_vertexCount; i += 32) {
        vertexBuffer.vertices[s_globalVertexOffset + i].position = s_vertices[i];
        // vertexBuffer.vertices[s_globalVertexOffset + i].normal = s_normals[i];
    }
    
    // Write indices to global memory (with offset adjustment)
    for (uint i = threadID; i < s_indexCount; i += 32) {
        indexBuffer.indices[s_globalIndexOffset + i] = s_indices[i] + s_globalVertexOffset;
    }
    
    // Write meshlet descriptor
    if (threadID == 0 && s_vertexCount > 0) {
        meshletDesc.descriptors[s_meshletIndex].vertexOffset = s_globalVertexOffset;
        meshletDesc.descriptors[s_meshletIndex].vertexCount = s_vertexCount;
        meshletDesc.descriptors[s_meshletIndex].indexOffset = s_globalIndexOffset;
        meshletDesc.descriptors[s_meshletIndex].indexCount = s_indexCount;
    }
    
    // Set mesh outputs (required by spec, even if not rasterizing)
    if (threadID == 0) {
        SetMeshOutputsEXT(s_vertexCount, s_indexCount / 3);
    }
    
    barrier();
    
    // Output vertices (required even if not rasterizing)
    for (uint i = threadID; i < s_vertexCount; i += 32) {
        gl_MeshVerticesEXT[i].gl_Position = vec4(s_vertices[i].xyz, 1.0);
        fragPos[i] = s_vertices[i].xyz;
        // fragNormal[i] = s_normals[i];
    }
    
    // Output primitive indices
    for (uint i = threadID; i < s_indexCount / 3; i += 32) {
        uint baseIdx = i * 3;
        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(
            s_indices[baseIdx + 0],
            s_indices[baseIdx + 1],
            s_indices[baseIdx + 2]
        );
    }
}