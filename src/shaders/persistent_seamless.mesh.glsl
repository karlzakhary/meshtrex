#version 460 core
#extension GL_EXT_mesh_shader : require

// Bindings - keeping your exact names and numbers
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockStride;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;

layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable {
    int triTable[256 * 16];
} mcTriTable;

struct VertexData {
    vec4 position;
};

layout(set = 0, binding = 6, std430) writeonly buffer VertexBuffer {
    VertexData vertices[];
} vertexBuffer;

layout(set = 0, binding = 8, std430) writeonly buffer IndexBuffer {
    uint indices[];
} indexBuffer;

struct MeshletDescriptor {
    uint vertexOffset;
    uint vertexCount;
    uint indexOffset;
    uint primitiveCount;
};

layout(set = 0, binding = 10, std430) writeonly buffer MeshletDescriptors {
    MeshletDescriptor descriptors[];
} meshletDesc;

layout(set = 0, binding = 12, std430) buffer MeshletDescCount {
    uint meshletCounter;
} meshletDescCount;

layout(set = 0, binding = 14, std430) readonly buffer MarchingCubesEdgeTable {
    uint edgeTable[256];
} mcEdgeTable;

// Must match task shader definitions
#define CORE_BLOCK_SIZE_X 8
#define CORE_BLOCK_SIZE_Y 8
#define CORE_BLOCK_SIZE_Z 8
#define EXTENDED_BLOCK_SIZE_X 10
#define EXTENDED_BLOCK_SIZE_Y 10
#define EXTENDED_BLOCK_SIZE_Z 10
#define MAX_MESHLET_VERTICES 128
#define MAX_MESHLET_PRIMITIVES 254
#define TOTAL_CORE_CELLS (CORE_BLOCK_SIZE_X * CORE_BLOCK_SIZE_Y * CORE_BLOCK_SIZE_Z)
#define TOTAL_EXTENDED_CELLS (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y * EXTENDED_BLOCK_SIZE_Z)
#define MAX_SUBBLOCKS 64

#define PMB_EDGE_X 0
#define PMB_EDGE_Y 3
#define PMB_EDGE_Z 8

// Include all structures from task shader
struct CellData {
    uint cubeCase;
    uint vertexMask;
    uint primitiveCount;
};

struct VertexOwnership {
    uint globalCellIndex;
    uint edgeType;
    uint localVertexIndex;
};

struct SubblockInfo {
    uint mortonStart;
    uint mortonCount;
    uvec3 minBounds;
    uvec3 maxBounds;
    uint estimatedVertices;
    uint estimatedPrimitives;
    uint globalVertexOffset;
    uint globalIndexOffset;
};

struct TaskPayload {
    uvec3 blockOrigin;
    uint blockId;
    uint numSubblocks;
    SubblockInfo subblocks[MAX_SUBBLOCKS];
    CellData cellData[EXTENDED_BLOCK_SIZE_X][EXTENDED_BLOCK_SIZE_Y][EXTENDED_BLOCK_SIZE_Z];
    uint occupiedMortonIndices[TOTAL_CORE_CELLS];
    uint occupiedCount;
    VertexOwnership vertexOwnership[TOTAL_EXTENDED_CELLS * 3];
    uint ownedVertexCount;
};

taskPayloadSharedEXT TaskPayload taskPayload;

// Outputs
layout(location = 0) out vec3 vertexPosition[];

layout(max_vertices = MAX_MESHLET_VERTICES, max_primitives = MAX_MESHLET_PRIMITIVES) out;
layout(triangles) out;

// Shared memory
shared vec3 shared_vertexPositions[MAX_MESHLET_VERTICES];
shared uint shared_vertexGlobalIndices[MAX_MESHLET_VERTICES];
shared uint shared_vertexCount;
shared uint shared_primitiveCount;
shared uint shared_indexData[MAX_MESHLET_PRIMITIVES * 3];

// Morton decoding
uint compactBits(uint v) {
    v &= 0x09249249;
    v = (v | (v >> 2)) & 0x030C30C3;
    v = (v | (v >> 4)) & 0x0300F00F;
    v = (v | (v >> 8)) & 0x030000FF;
    v = (v | (v >> 16)) & 0x000003FF;
    return v;
}

uvec3 mortonDecode3D(uint morton) {    
    return uvec3(
        compactBits(morton >> 0),
        compactBits(morton >> 1),
        compactBits(morton >> 2)
    );
}

// Edge vertex positions for MC edges
const ivec3 edgeVertices[12][2] = {
    {{0,0,0}, {1,0,0}}, {{1,0,0}, {1,1,0}}, {{0,1,0}, {1,1,0}}, {{0,0,0}, {0,1,0}},
    {{0,0,1}, {1,0,1}}, {{1,0,1}, {1,1,1}}, {{0,1,1}, {1,1,1}}, {{0,0,1}, {0,1,1}},
    {{0,0,0}, {0,0,1}}, {{1,0,0}, {1,0,1}}, {{1,1,0}, {1,1,1}}, {{0,1,0}, {0,1,1}}
};

// Neighbor mapping
const ivec4 neighborMap[12] = {
    ivec4(0, 0, 0, 0),  // Edge 0: X edge
    ivec4(1, 0, 0, 1),  // Edge 1: Y edge  
    ivec4(0, 1, 0, 0),  // Edge 2: X edge
    ivec4(0, 0, 0, 1),  // Edge 3: Y edge
    ivec4(0, 0, 1, 0),  // Edge 4: X edge
    ivec4(1, 0, 1, 1),  // Edge 5: Y edge
    ivec4(0, 1, 1, 0),  // Edge 6: X edge
    ivec4(0, 0, 1, 1),  // Edge 7: Y edge
    ivec4(0, 0, 0, 2),  // Edge 8: Z edge
    ivec4(1, 0, 0, 2),  // Edge 9: Z edge
    ivec4(1, 1, 0, 2),  // Edge 10: Z edge
    ivec4(0, 1, 0, 2)   // Edge 11: Z edge
};

float sampleVolume(ivec3 coord) {
    if (all(greaterThanEqual(coord, ivec3(0))) && all(lessThan(coord, ivec3(ubo.volumeDim.xyz)))) {
        return float(imageLoad(volumeImage, coord).r);
    }
    return 0.0;
}

vec3 interpolateVertex(vec3 p1, vec3 p2, float v1, float v2) {
    float t = clamp((ubo.isovalue - v1) / (v2 - v1), 0.0, 1.0);
    return mix(p1, p2, t);
}

// Find vertex index in ownership list
uint findVertexIndex(uint globalCellIndex, uint edgeType) {
    for (uint i = 0; i < taskPayload.ownedVertexCount; i++) {
        if (taskPayload.vertexOwnership[i].globalCellIndex == globalCellIndex &&
            taskPayload.vertexOwnership[i].edgeType == edgeType) {
            return taskPayload.vertexOwnership[i].localVertexIndex;
        }
    }
    return 0xFFFFFFFF;
}

uint getGlobalCellIndex(ivec3 cellPos) {
    return cellPos.x + cellPos.y * ubo.volumeDim.x + cellPos.z * ubo.volumeDim.x * ubo.volumeDim.y;
}

layout(local_size_x = 32) in;
void main() {
    uint threadId = gl_LocalInvocationIndex;
    uint subblockId = gl_WorkGroupID.x;
    
    if (subblockId >= taskPayload.numSubblocks) return;
    
    SubblockInfo subblock = taskPayload.subblocks[subblockId];
    
    // Initialize shared memory
    if (threadId == 0) {
        shared_vertexCount = 0;
        shared_primitiveCount = 0;
    }
    barrier();
    
    // Phase 1: Generate owned vertices
    uint verticesPerThread = (taskPayload.ownedVertexCount + 31) / 32;
    for (uint i = 0; i < verticesPerThread; i++) {
        uint vIdx = threadId + i * 32;
        if (vIdx >= taskPayload.ownedVertexCount) continue;
        
        VertexOwnership ownership = taskPayload.vertexOwnership[vIdx];
        
        // Decode global cell position
        uint cellIdx = ownership.globalCellIndex;
        ivec3 globalCellPos = ivec3(
            cellIdx % ubo.volumeDim.x,
            (cellIdx / ubo.volumeDim.x) % ubo.volumeDim.y,
            cellIdx / (ubo.volumeDim.x * ubo.volumeDim.y)
        );
        
        // Generate vertex
        ivec3 p0, p1;
        if (ownership.edgeType == 0) { // X edge
            p0 = globalCellPos;
            p1 = globalCellPos + ivec3(1, 0, 0);
        } else if (ownership.edgeType == 1) { // Y edge
            p0 = globalCellPos;
            p1 = globalCellPos + ivec3(0, 1, 0);
        } else { // Z edge
            p0 = globalCellPos;
            p1 = globalCellPos + ivec3(0, 0, 1);
        }
        
        float v0 = sampleVolume(p0);
        float v1 = sampleVolume(p1);
        vec3 pos = interpolateVertex(vec3(p0), vec3(p1), v0, v1);
        
        uint localIdx = atomicAdd(shared_vertexCount, 1);
        if (localIdx < MAX_MESHLET_VERTICES) {
            shared_vertexPositions[localIdx] = pos;
            shared_vertexGlobalIndices[localIdx] = subblock.globalVertexOffset + vIdx;
        }
    }
    
    barrier();
    
    // Phase 2: Generate triangles
    uint cellsPerThread = (subblock.mortonCount + 31) / 32;
    for (uint i = 0; i < cellsPerThread; i++) {
        uint cellIdx = threadId + i * 32;
        if (cellIdx >= subblock.mortonCount) continue;
        
        uint mortonIdx = taskPayload.occupiedMortonIndices[subblock.mortonStart + cellIdx];
        uvec3 localPos = mortonDecode3D(mortonIdx);
        
        CellData cell = taskPayload.cellData[localPos.x][localPos.y][localPos.z];
        if (cell.primitiveCount == 0) continue;
        
        ivec3 globalCellPos = ivec3(taskPayload.blockOrigin) + ivec3(localPos) - ivec3(1);
        
        // Generate triangles
        int baseIdx = int(cell.cubeCase) * 16;
        for (uint tri = 0; tri < cell.primitiveCount && tri < 5; tri++) {
            uvec3 indices = uvec3(0xFFFFFFFF);
            bool validTriangle = true;
            
            for (uint v = 0; v < 3; v++) {
                int edgeId = mcTriTable.triTable[baseIdx + tri * 3 + v];
                if (edgeId == -1) {
                    validTriangle = false;
                    break;
                }
                
                // Get neighbor cell
                ivec4 neighbor = neighborMap[edgeId];
                ivec3 neighborCellPos = globalCellPos + neighbor.xyz;
                uint edgeType = neighbor.w;
                
                // Get global cell index
                uint globalNeighborIdx = getGlobalCellIndex(neighborCellPos);
                
                // Find vertex in our ownership list
                uint vertexOwnershipIdx = findVertexIndex(globalNeighborIdx, edgeType);
                
                if (vertexOwnershipIdx != 0xFFFFFFFF) {
                    // We own this vertex - find its local index
                    for (uint j = 0; j < shared_vertexCount; j++) {
                        if (shared_vertexGlobalIndices[j] == subblock.globalVertexOffset + vertexOwnershipIdx) {
                            indices[v] = j;
                            break;
                        }
                    }
                } else {
                    // We don't own this vertex - need to reference it from another block
                    // For now, mark as invalid (in practice, would implement inter-block lookup)
                    validTriangle = false;
                    break;
                }
            }
            
            if (validTriangle && indices.x != 0xFFFFFFFF && indices.y != 0xFFFFFFFF && indices.z != 0xFFFFFFFF) {
                uint primIdx = atomicAdd(shared_primitiveCount, 1);
                if (primIdx < MAX_MESHLET_PRIMITIVES) {
                    shared_indexData[primIdx * 3 + 0] = indices.x;
                    shared_indexData[primIdx * 3 + 1] = indices.y;
                    shared_indexData[primIdx * 3 + 2] = indices.z;
                }
            }
        }
    }
    
    barrier();
    
    // Phase 3: Output
    if (threadId == 0) {
        uint actualVertexCount = min(shared_vertexCount, uint(MAX_MESHLET_VERTICES));
        uint actualPrimitiveCount = min(shared_primitiveCount, uint(MAX_MESHLET_PRIMITIVES));
        
        // Write meshlet descriptor
        uint meshletIdx = atomicAdd(meshletDescCount.meshletCounter, 1);
        meshletDesc.descriptors[meshletIdx] = MeshletDescriptor(
            subblock.globalVertexOffset,
            actualVertexCount,
            subblock.globalIndexOffset,
            actualPrimitiveCount
        );
        
        SetMeshOutputsEXT(actualVertexCount, actualPrimitiveCount);
    }
    
    barrier();
    
    // Write vertices
    uint verticesPerThread2 = (shared_vertexCount + 31) / 32;
    for (uint i = 0; i < verticesPerThread2; i++) {
        uint vIdx = threadId + i * 32;
        if (vIdx < shared_vertexCount && vIdx < MAX_MESHLET_VERTICES) {
            gl_MeshVerticesEXT[vIdx].gl_Position = vec4(shared_vertexPositions[vIdx], 1.0);
            vertexPosition[vIdx] = shared_vertexPositions[vIdx];
            
            vertexBuffer.vertices[shared_vertexGlobalIndices[vIdx]].position = vec4(shared_vertexPositions[vIdx], 1.0);
        }
    }
    
    // Write indices
    uint indicesPerThread = (shared_primitiveCount + 31) / 32;
    for (uint i = 0; i < indicesPerThread; i++) {
        uint primIdx = threadId + i * 32;
        if (primIdx < shared_primitiveCount && primIdx < MAX_MESHLET_PRIMITIVES) {
            uint idx0 = shared_indexData[primIdx * 3 + 0];
            uint idx1 = shared_indexData[primIdx * 3 + 1];
            uint idx2 = shared_indexData[primIdx * 3 + 2];
            
            gl_PrimitiveTriangleIndicesEXT[primIdx] = uvec3(idx0, idx1, idx2);
            
            uint globalIdx = subblock.globalIndexOffset + (primIdx * 3);
            indexBuffer.indices[globalIdx + 0] = shared_vertexGlobalIndices[idx0];
            indexBuffer.indices[globalIdx + 1] = shared_vertexGlobalIndices[idx1];
            indexBuffer.indices[globalIdx + 2] = shared_vertexGlobalIndices[idx2];
        }
    }
}