#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_debug_printf : enable

// Bindings
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
    vec4 position; // Store as vec4 for easier SSBO layout; w can be 1.0
    // vec4 normal;   // Store as vec4 for easier SSBO layout; w can be 0.0
};

layout(set = 0, binding = 6, std430) writeonly buffer VertexBuffer {
    VertexData vertices[];
} vertexBuffer;

layout(set = 0, binding = 8, std430) writeonly buffer IndexBuffer {
    uint indices[];
} indexBuffer;

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;      // Offset in vertices
    uint vertexCount;       // Number of vertices
    uint indexOffset;       // Offset in INDICES (not primitives!)
    uint primitiveCount;    // Number of triangles
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
#define MAX_SUBBLOCKS 64
#define EXTENDED_BLOCK_SIZE_X 10
#define EXTENDED_BLOCK_SIZE_Y 10
#define EXTENDED_BLOCK_SIZE_Z 10
#define MAX_MESHLET_VERTICES 256
#define MAX_MESHLET_PRIMITIVES 256
#define TOTAL_CORE_CELLS (CORE_BLOCK_SIZE_X * CORE_BLOCK_SIZE_Y * CORE_BLOCK_SIZE_Z)
#define TOTAL_EXTENDED_CELLS (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y * EXTENDED_BLOCK_SIZE_Z)

// PMB edge definitions
#define PMB_EDGE_X 0
#define PMB_EDGE_Y 3
#define PMB_EDGE_Z 8

// --- Data Structures ---
struct CellData {
    uint cubeCase;
    uint vertexMask;
    uint primitiveCount;
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
};

taskPayloadSharedEXT TaskPayload taskPayload;

// Vertex outputs
layout(location = 0) out vec3 vertexPosition[];

// Output specifications
layout(max_vertices = MAX_MESHLET_VERTICES, max_primitives = MAX_MESHLET_PRIMITIVES) out;
layout(triangles) out;

// Shared memory for vertex generation
shared vec3 shared_vertexPositions[MAX_MESHLET_VERTICES];
shared uint shared_vertexMap[EXTENDED_BLOCK_SIZE_X][EXTENDED_BLOCK_SIZE_Y][EXTENDED_BLOCK_SIZE_Z][3];
shared uint shared_vertexCount;
shared uint shared_primitiveCount;
shared uint shared_indexData[MAX_MESHLET_PRIMITIVES * 3];  // Store indices temporarily

// Morton decoding
uvec3 mortonDecode3D(uint morton) {
    uint x = morton & 0x09249249;
    uint y = (morton >> 1) & 0x09249249;
    uint z = (morton >> 2) & 0x09249249;
    
    x = (x | (x >> 2)) & 0x030C30C3;
    x = (x | (x >> 4)) & 0x0300F00F;
    x = (x | (x >> 8)) & 0x030000FF;
    x = (x | (x >> 16)) & 0x000003FF;
    
    y = (y | (y >> 2)) & 0x030C30C3;
    y = (y | (y >> 4)) & 0x0300F00F;
    y = (y | (y >> 8)) & 0x030000FF;
    y = (y | (y >> 16)) & 0x000003FF;
    
    z = (z | (z >> 2)) & 0x030C30C3;
    z = (z | (z >> 4)) & 0x0300F00F;
    z = (z | (z >> 8)) & 0x030000FF;
    z = (z | (z >> 16)) & 0x000003FF;
    
    return uvec3(x, y, z);
}

// Edge table for vertex interpolation
const ivec3 edgeVertices[12][2] = {
    {{0,0,0}, {1,0,0}}, {{1,0,0}, {1,1,0}}, {{0,1,0}, {1,1,0}}, {{0,0,0}, {0,1,0}},
    {{0,0,1}, {1,0,1}}, {{1,0,1}, {1,1,1}}, {{0,1,1}, {1,1,1}}, {{0,0,1}, {0,1,1}},
    {{0,0,0}, {0,0,1}}, {{1,0,0}, {1,0,1}}, {{1,1,0}, {1,1,1}}, {{0,1,0}, {0,1,1}}
};

// Neighbor mapping for PMB - which cell owns each edge
const ivec4 neighborMap[12] = {
    ivec4(0, 0, 0, 0),  // Edge 0: X edge at origin (PMB_EDGE_X)
    ivec4(1, 0, 0, 1),  // Edge 1: Y edge at (+1,0,0)
    ivec4(0, 1, 0, 0),  // Edge 2: X edge at (0,+1,0)
    ivec4(0, 0, 0, 1),  // Edge 3: Y edge at origin (PMB_EDGE_Y)
    ivec4(0, 0, 1, 0),  // Edge 4: X edge at (0,0,+1)
    ivec4(1, 0, 1, 1),  // Edge 5: Y edge at (+1,0,+1)
    ivec4(0, 1, 1, 0),  // Edge 6: X edge at (0,+1,+1)
    ivec4(0, 0, 1, 1),  // Edge 7: Y edge at (0,0,+1)
    ivec4(0, 0, 0, 2),  // Edge 8: Z edge at origin (PMB_EDGE_Z)
    ivec4(1, 0, 0, 2),  // Edge 9: Z edge at (+1,0,0)
    ivec4(1, 1, 0, 2),  // Edge 10: Z edge at (+1,+1,0)
    ivec4(0, 1, 0, 2)   // Edge 11: Z edge at (0,+1,0)
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

vec3 calculateNormal(vec3 vertexPos_global_coords) {
    // Central differencing for gradient calculation using global vertex position
    // Note: vertexPos_global_coords is the interpolated vertex position, not necessarily on a voxel corner.
    // For gradient, we sample around this position.
    // The PMB paper samples at +/-1 voxel from the cell's base voxel.
    // A common method is to sample around the vertex position itself.

    float delta = 0.5f; // Or a small voxel-relative offset like 1.0f for voxel units
    ivec3 volMax = ivec3(ubo.volumeDim.xyz) - 1;

    // Sample points for central differences around vertexPos_global_coords
    // Ensure sample points are within volume bounds.
    // The coordinates for imageLoad must be integer.
    float s_xp1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords + vec3(delta, 0,0)), ivec3(0), volMax)).r);
    float s_xm1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords - vec3(delta, 0,0)), ivec3(0), volMax)).r);
    float s_yp1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords + vec3(0, delta,0)), ivec3(0), volMax)).r);
    float s_ym1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords - vec3(0, delta,0)), ivec3(0), volMax)).r);
    float s_zp1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords + vec3(0,0, delta)), ivec3(0), volMax)).r);
    float s_zm1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords - vec3(0,0, delta)), ivec3(0), volMax)).r);

    vec3 grad = vec3(s_xp1 - s_xm1, s_yp1 - s_ym1, s_zp1 - s_zm1);

    if (length(grad) < 1e-5f) return vec3(0, 1, 0); // Default normal for zero gradient
    return -normalize(grad); // Normal points "out" if isovalue is a lower bound for "inside"
}

// Alternative: Use fixed-point arithmetic
vec3 interpolateVertexFixed(vec3 p1, vec3 p2, float v1, float v2) {
    const int FIXED_SCALE = 1000000; // 6 decimal places
    
    // Convert to fixed point
    ivec3 ip1 = ivec3(p1 * FIXED_SCALE);
    ivec3 ip2 = ivec3(p2 * FIXED_SCALE);
    int iv1 = int(v1 * FIXED_SCALE);
    int iv2 = int(v2 * FIXED_SCALE);
    int iiso = int(ubo.isovalue * FIXED_SCALE);
    
    // Compute t in fixed point
    int numerator = iiso - iv1;
    int denominator = iv2 - iv1;
    
    if (abs(denominator) < 1) {
        // Avoid division by zero
        return p1;
    }
    
    // Interpolate in fixed point
    ivec3 diff = ip2 - ip1;
    ivec3 result = ip1 + (diff * numerator) / denominator;
    
    // Convert back to float
    return vec3(result) / float(FIXED_SCALE);
}

// IMPORTANT: Also ensure consistent vertex ordering by using global cell coordinates
// When generating vertices, always use global coordinates:
vec3 generateVertexPosition(ivec3 globalCellPos, uint edgeType) {
    ivec3 p0, p1;
    
    // Use consistent edge definitions based on global coordinates
    if (edgeType == 0) { // X edge
        p0 = globalCellPos;
        p1 = globalCellPos + ivec3(1, 0, 0);
    } else if (edgeType == 1) { // Y edge
        p0 = globalCellPos;
        p1 = globalCellPos + ivec3(0, 1, 0);
    } else { // Z edge
        p0 = globalCellPos;
        p1 = globalCellPos + ivec3(0, 0, 1);
    }
    
    float v0 = sampleVolume(p0);
    float v1 = sampleVolume(p1);
    
    // Use the double precision version for bit-exact results
    return interpolateVertexFixed(vec3(p0), vec3(p1), v0, v1);
}

layout(local_size_x = 32) in;
void main() {
    uint threadId = gl_LocalInvocationIndex;
    uint subblockId = gl_WorkGroupID.x;
    
    if (subblockId >= taskPayload.numSubblocks) {
        return;
    }
    
    SubblockInfo subblock = taskPayload.subblocks[subblockId];
    
    // Initialize shared memory
    if (threadId == 0) {
        shared_vertexCount = 0;
        shared_primitiveCount = 0;
    }
    
    // Clear vertex map - CRITICAL: Initialize to invalid value
    uint cellsPerThread = (TOTAL_EXTENDED_CELLS + 31) / 32;
    for (uint i = 0; i < cellsPerThread; i++) {
        uint cellIdx = threadId + i * 32;
        if (cellIdx < TOTAL_EXTENDED_CELLS) {
            uint z = cellIdx / (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y);
            uint y = (cellIdx % (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y)) / EXTENDED_BLOCK_SIZE_X;
            uint x = cellIdx % EXTENDED_BLOCK_SIZE_X;
            shared_vertexMap[x][y][z][0] = 0xFFFFFFFF;
            shared_vertexMap[x][y][z][1] = 0xFFFFFFFF;
            shared_vertexMap[x][y][z][2] = 0xFFFFFFFF;
        }
    }
    
    barrier();
    
    // Phase 1: Generate vertices for ALL cells in the subblock bounds (including context)
    uvec3 regionSize = subblock.maxBounds - subblock.minBounds;
    uint totalCells = regionSize.x * regionSize.y * regionSize.z;
    cellsPerThread = (totalCells + 31) / 32;
    
    for (uint i = 0; i < cellsPerThread; i++) {
        uint localIdx = threadId + i * 32;
        if (localIdx >= totalCells) continue;
        
        // Convert linear index to 3D position
        uvec3 localPos;
        localPos.z = localIdx / (regionSize.x * regionSize.y);
        localPos.y = (localIdx % (regionSize.x * regionSize.y)) / regionSize.x;
        localPos.x = localIdx % regionSize.x;
        
        uvec3 cellPos = subblock.minBounds + localPos;
        
        // Bounds check
        if (any(greaterThanEqual(cellPos, uvec3(EXTENDED_BLOCK_SIZE_X, EXTENDED_BLOCK_SIZE_Y, EXTENDED_BLOCK_SIZE_Z)))) {
            continue;
        }
        
        CellData cell = taskPayload.cellData[cellPos.x][cellPos.y][cellPos.z];
        
        if (cell.vertexMask == 0) continue;
        
        ivec3 globalCellPos = ivec3(taskPayload.blockOrigin) + ivec3(cellPos) - ivec3(1);
        
        // Generate vertices for PMB edges
        for (uint edge = 0; edge < 3; edge++) {
            if ((cell.vertexMask & (1u << edge)) == 0) continue;
            
            uint edgeId = (edge == 0) ? PMB_EDGE_X : (edge == 1) ? PMB_EDGE_Y : PMB_EDGE_Z;
            
            ivec3 v0 = globalCellPos + edgeVertices[edgeId][0];
            ivec3 v1 = globalCellPos + edgeVertices[edgeId][1];
            
            float val0 = sampleVolume(v0);
            float val1 = sampleVolume(v1);
            
            // vec3 pos = interpolateVertex(vec3(v0), vec3(v1), val0, val1);
            vec3 pos = generateVertexPosition(globalCellPos, edge);
            
            uint vertexIdx = atomicAdd(shared_vertexCount, 1);
            if (vertexIdx < MAX_MESHLET_VERTICES) {
                shared_vertexPositions[vertexIdx] = pos;
                shared_vertexMap[cellPos.x][cellPos.y][cellPos.z][edge] = vertexIdx;
            }
        }
    }
    
    barrier();
    
    // Phase 2: Generate triangles ONLY from core cells
    uint occupiedPerThread = (subblock.mortonCount + 31) / 32;

    for (uint i = 0; i < occupiedPerThread; i++) {
        uint localIdx = threadId + i * 32;
        if (localIdx >= subblock.mortonCount) continue;
        
        uint mortonIdx = taskPayload.occupiedMortonIndices[subblock.mortonStart + localIdx];
        uvec3 cellPos = mortonDecode3D(mortonIdx);
        
        // REMOVED: The restrictive core cell check
        // We process ALL occupied cells in our list
        
        CellData cell = taskPayload.cellData[cellPos.x][cellPos.y][cellPos.z];
        if (cell.primitiveCount == 0) continue;
        
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
                
                // Get the cell that owns this edge
                ivec4 neighbor = neighborMap[edgeId];
                uvec3 neighborPos = cellPos + uvec3(neighbor.xyz);
                
                // CRITICAL FIX: Don't check if neighbor is within subblock bounds!
                // Just check if we have the vertex in our map
                // Context cells outside subblock bounds are VALID and NEEDED
                
                // Only check that the neighbor is within the extended block bounds
                if (!all(greaterThanEqual(neighborPos, uvec3(0))) || 
                    !all(lessThan(neighborPos, uvec3(EXTENDED_BLOCK_SIZE_X, 
                                                    EXTENDED_BLOCK_SIZE_Y, 
                                                    EXTENDED_BLOCK_SIZE_Z)))) {
                    validTriangle = false;
                    break;
                }
                
                // Get vertex index from the owning cell
                uint vertexIdx = shared_vertexMap[neighborPos.x][neighborPos.y][neighborPos.z][neighbor.w];
                
                if (vertexIdx == 0xFFFFFFFF || vertexIdx >= shared_vertexCount) {
                    validTriangle = false;
                    break;
                }
                
                indices[v] = vertexIdx;
            }
            
            if (validTriangle) {
                uint primIdx = atomicAdd(shared_primitiveCount, 1);
                if (primIdx < MAX_MESHLET_PRIMITIVES) {
                    // Store indices temporarily
                    shared_indexData[primIdx * 3 + 0] = indices.x;
                    shared_indexData[primIdx * 3 + 1] = indices.y;
                    shared_indexData[primIdx * 3 + 2] = indices.z;
                }
            }
        }
    }
    
    barrier();
    
    // Phase 3: Write output
    if (threadId == 0) {
        uint actualVertexCount = min(shared_vertexCount, uint(MAX_MESHLET_VERTICES));
        uint actualPrimitiveCount = min(shared_primitiveCount, uint(MAX_MESHLET_PRIMITIVES));
        
        // DON'T clamp to estimated values - this was the bug!
        // actualVertexCount = min(actualVertexCount, subblock.estimatedVertices);
        // actualPrimitiveCount = min(actualPrimitiveCount, subblock.estimatedPrimitives);
        
        // Instead, ensure we allocated enough space
        if (actualVertexCount > subblock.estimatedVertices) {
            // This shouldn't happen with proper estimation
            debugPrintfEXT("Warning: Meshlet exceeded vertex estimate: %d > %d", 
                        actualVertexCount, subblock.estimatedVertices);
            actualVertexCount = subblock.estimatedVertices; // Only clamp if we truly exceeded
        }
        
        // Write meshlet descriptor
        uint meshletIdx = atomicAdd(meshletDescCount.meshletCounter, 1);
        meshletDesc.descriptors[meshletIdx] = MeshletDescriptor(
            subblock.globalVertexOffset,
            actualVertexCount,
            subblock.globalIndexOffset,
            actualPrimitiveCount
        );
        
        // Set primitive count for mesh shader output
        SetMeshOutputsEXT(actualVertexCount, actualPrimitiveCount);
    }
    
    barrier();
    
    // Write vertices to global buffer
    uint verticesPerThread = (shared_vertexCount + 31) / 32;
    for (uint i = 0; i < verticesPerThread; i++) {
        uint vIdx = threadId + i * 32;
        if (vIdx < shared_vertexCount && vIdx < MAX_MESHLET_VERTICES) {
            // Write to mesh shader output
            gl_MeshVerticesEXT[vIdx].gl_Position = vec4(shared_vertexPositions[vIdx], 1.0);
            vertexPosition[vIdx] = shared_vertexPositions[vIdx];
            
            // Write to global buffer
            vertexBuffer.vertices[subblock.globalVertexOffset + vIdx].position = vec4(shared_vertexPositions[vIdx], 1.0);
            // vertexBuffer.vertices[subblock.globalVertexOffset + vIdx].normal = vec4(calculateNormal(shared_vertexPositions[vIdx]), 0.0);
        }
    }
    
    // Write indices to global buffer and primitive indices
    uint indicesPerThread = (shared_primitiveCount + 31) / 32;
    for (uint i = 0; i < indicesPerThread; i++) {
        uint primIdx = threadId + i * 32;
        if (primIdx < shared_primitiveCount && primIdx < MAX_MESHLET_PRIMITIVES) {
            // Read from temporary storage
            uint idx0 = shared_indexData[primIdx * 3 + 0];
            uint idx1 = shared_indexData[primIdx * 3 + 1];
            uint idx2 = shared_indexData[primIdx * 3 + 2];
            
            // Write to mesh shader primitive output
            gl_PrimitiveTriangleIndicesEXT[primIdx] = uvec3(idx0, idx1, idx2);
            
            // Write to global index buffer with proper offset
            uint globalIdx = subblock.globalIndexOffset + (primIdx * 3);
            indexBuffer.indices[globalIdx + 0] = subblock.globalVertexOffset + idx0;
            indexBuffer.indices[globalIdx + 1] = subblock.globalVertexOffset + idx1;
            indexBuffer.indices[globalIdx + 2] = subblock.globalVertexOffset + idx2;
        }
    }
}