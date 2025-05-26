#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 8
#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5
#define MAX_MESHLET_VERTS 256u
#define MAX_MESHLET_PRIMS 256u
#define MAX_TRI_INDICES 16
#define WORKGROUP_SIZE 32

// --- Structures ---
struct TaskPayload {
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletDescOffset;
    uvec3 blockOrigin;
    uvec3 subBlockDim;
    uint taskId;
    uint estimatedVertices;  // ADDED: Actual estimate for this sub-block
    uint estimatedPrimitives; // ADDED: Actual estimate for this sub-block
};

taskPayloadSharedEXT TaskPayload taskPayloadOut;

// --- Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount_SSBO { 
    uint count;
} activeBlockCountBuffer;
layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs_SSBO {
    uint compactedBlkArray[];
} blockIds;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO {
    int triTable[];
} mc;
layout(set = 0, binding = 11, std430) buffer VertexCount_SSBO { uint vertexCounter; } vCount;
layout(set = 0, binding = 13, std430) buffer IndexCount_SSBO { uint indexCounter; } iCount;
layout(set = 0, binding = 12, std430) buffer MeshletDescCount_SSBO { uint meshletCounter; } meshletCount;

// --- Helper Functions ---
uint sampleMcCase(uvec3 cell_global_origin) {
    uint cubeCase = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 corner_offset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
        ivec3 voxel_coord = ivec3(cell_global_origin) + corner_offset;
        uint val = 0;
        if (all(greaterThanEqual(voxel_coord, ivec3(0))) && all(lessThan(voxel_coord, ivec3(ubo.volumeDim.xyz)))) {
            val = imageLoad(volumeImage, voxel_coord).r;
        }
        if (float(val) >= ubo.isovalue) {
            cubeCase |= (1 << i);
        }
    }
    return cubeCase;
}

uint countCellVertices(uint cubeCase) {
    uint flags = 0u;
    uint vertCount = 0u;
    int base = int(cubeCase) * MAX_TRI_INDICES;
    for (int i = 0; i < MAX_TRI_INDICES; ++i) {
        int edgeID = mc.triTable[base + i];
        if (edgeID < 0) break;
        uint mask = 1u << uint(edgeID);
        if ((flags & mask) == 0u) {
            flags |= mask;
            vertCount++;
        }
    }
    return vertCount;
}

uint countCellPrimitives(uint cubeCase) {
    uint primCount = 0;
    int base = int(cubeCase) * MAX_TRI_INDICES;
    for (int i = 0; i < MAX_TRI_INDICES; i += 3) {
        if (mc.triTable[base + i] < 0) break;
        primCount++;
    }
    return primCount;
}

// Improved estimation with vertex sharing consideration
uint estimateVertsInSubBlockWithSharing(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) {
    uint totalVertices = 0;
    uint duplicateEstimate = 0;
    
    for (uint z = 0; z < subBlockDimensions.z; ++z) {
        for (uint y = 0; y < subBlockDimensions.y; ++y) {
            for (uint x = 0; x < subBlockDimensions.x; ++x) {
                uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                uint cubeCase = sampleMcCase(cellGlobalOrigin);
                uint cellVerts = countCellVertices(cubeCase);
                totalVertices += cellVerts;
                
                // Estimate vertex sharing within block
                // Interior cells share more vertices than boundary cells
                if (x > 0 && y > 0 && z > 0 && 
                    x < subBlockDimensions.x - 1 && 
                    y < subBlockDimensions.y - 1 && 
                    z < subBlockDimensions.z - 1) {
                    duplicateEstimate += cellVerts / 3; // Conservative sharing estimate
                } else {
                    duplicateEstimate += cellVerts / 6; // Less sharing at boundaries
                }
                
                if (totalVertices > MAX_MESHLET_VERTS * 2) {
                    return MAX_MESHLET_VERTS + 1; // Signal overflow early
                }
            }
        }
    }
    
    // Account for vertex sharing
    uint finalEstimate = max(totalVertices - duplicateEstimate, totalVertices / 2);
    return min(finalEstimate, MAX_MESHLET_VERTS);
}

uint estimatePrimsInSubBlock(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) {
    uint sum = 0;
    for (uint z = 0; z < subBlockDimensions.z; ++z) {
        for (uint y = 0; y < subBlockDimensions.y; ++y) {
            for (uint x = 0; x < subBlockDimensions.x; ++x) {
                uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                uint cubeCase = sampleMcCase(cellGlobalOrigin);
                sum += countCellPrimitives(cubeCase);
                if (sum > MAX_MESHLET_PRIMS) {
                    return MAX_MESHLET_PRIMS + 1; // Signal overflow
                }
            }
        }
    }
    return sum;
}

void emitSubBlock(uvec3 subBlockOrigin, uvec3 subBlockDim, uint compactedBlockID, uint estVerts, uint estPrims) {
    // More conservative reservation with larger safety margins
    uint reservedVertices = min(estVerts + 64u, MAX_MESHLET_VERTS); // Larger safety margin
    uint reservedIndices = min(estPrims * 3u + 64u, MAX_MESHLET_PRIMS * 3u);
    
    // Ensure minimum reservation for non-empty blocks
    if (estVerts > 0) {
        reservedVertices = max(reservedVertices, 32u);
    }
    if (estPrims > 0) {
        reservedIndices = max(reservedIndices, 96u); // 32 triangles minimum
    }
    
    uint vOff = atomicAdd(vCount.vertexCounter, reservedVertices);
    uint iOff = atomicAdd(iCount.indexCounter, reservedIndices);
    uint dOff = atomicAdd(meshletCount.meshletCounter, 1u);
    
    // debugPrintfEXT("TS EMIT: Block %u, Est V=%u P=%u, Reserved V=%u I=%u\n", 
    //               compactedBlockID, estVerts, estPrims, reservedVertices, reservedIndices);
    
    taskPayloadOut = TaskPayload(
        vOff, 
        iOff, 
        dOff, 
        subBlockOrigin, 
        subBlockDim, 
        compactedBlockID,
        estVerts,      // Pass actual estimate
        estPrims       // Pass actual estimate
    );
    
    EmitMeshTasksEXT(1, 1, 1);
}

// --- Task Shader Main ---
layout(local_size_x = WORKGROUP_SIZE) in;
void main() {
    if (gl_LocalInvocationIndex != 0u) {
        return;
    }

    uint compactedBlockID = gl_WorkGroupID.x;
    if (compactedBlockID >= activeBlockCountBuffer.count) {
        // debugPrintfEXT("TS WARN: compactedBlockID %u >= activeBlockCount %u. Skipping.\n", 
        //               compactedBlockID, activeBlockCountBuffer.count);
        return;
    }

    uint originalBlockIndex = blockIds.compactedBlkArray[compactedBlockID];

    // Calculate full block origin
    uvec3 fullBlockGlobalOrigin;
    fullBlockGlobalOrigin.z = originalBlockIndex / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = originalBlockIndex % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    fullBlockGlobalOrigin.y = sliceIndex / ubo.blockGridDim.x;
    fullBlockGlobalOrigin.x = sliceIndex % ubo.blockGridDim.x;
    fullBlockGlobalOrigin *= ubo.blockDim.xyz;

    uvec3 fullBlockDim = ubo.blockDim.xyz;

    // Try full block first with improved estimation
    uint estVertsFull = estimateVertsInSubBlockWithSharing(fullBlockGlobalOrigin, fullBlockDim);
    uint estPrimsFull = estimatePrimsInSubBlock(fullBlockGlobalOrigin, fullBlockDim);

    if (estVertsFull == 0 && estPrimsFull == 0) {
        // debugPrintfEXT("TS INFO: Full block ID %u is empty.\n", compactedBlockID);
        return;
    }

    if (estVertsFull <= MAX_MESHLET_VERTS && estPrimsFull <= MAX_MESHLET_PRIMS) {
        // debugPrintfEXT("TS EMIT: Full block %u fits. EstV=%u EstP=%u\n", 
        //               compactedBlockID, estVertsFull, estPrimsFull);
        emitSubBlock(fullBlockGlobalOrigin, fullBlockDim, compactedBlockID, estVertsFull, estPrimsFull);
        return;
    }

    // Subdivide to half blocks
    // debugPrintfEXT("TS SUBDIVIDE: Block %u too large (EstV=%u, EstP=%u). Subdividing...\n", 
    //               compactedBlockID, estVertsFull, estPrimsFull);
    
    uvec3 halfBlockDim = max(uvec3(1), fullBlockDim / 2u);
    
    for (uint oz = 0u; oz < 2u; ++oz) {
        for (uint oy = 0u; oy < 2u; ++oy) {
            for (uint ox = 0u; ox < 2u; ++ox) {
                // Calculate actual dimensions for this half block
                uvec3 currentHalfDim = uvec3(
                    (ox == 0) ? halfBlockDim.x : (fullBlockDim.x - halfBlockDim.x),
                    (oy == 0) ? halfBlockDim.y : (fullBlockDim.y - halfBlockDim.y),
                    (oz == 0) ? halfBlockDim.z : (fullBlockDim.z - halfBlockDim.z)
                );
                
                // Skip empty dimensions
                if (currentHalfDim.x == 0 || currentHalfDim.y == 0 || currentHalfDim.z == 0) {
                    continue;
                }
                
                uvec3 halfOrigin = fullBlockGlobalOrigin + 
                                  uvec3(ox * halfBlockDim.x, oy * halfBlockDim.y, oz * halfBlockDim.z);

                uint estVertsHalf = estimateVertsInSubBlockWithSharing(halfOrigin, currentHalfDim);
                uint estPrimsHalf = estimatePrimsInSubBlock(halfOrigin, currentHalfDim);

                if (estVertsHalf == 0 && estPrimsHalf == 0) continue;

                if (estVertsHalf <= MAX_MESHLET_VERTS && estPrimsHalf <= MAX_MESHLET_PRIMS) {
                    debugPrintfEXT("TS EMIT: Half-block");
                    emitSubBlock(halfOrigin, currentHalfDim, compactedBlockID, estVertsHalf, estPrimsHalf);
                } else {
                    // Further subdivide to quarter blocks
                    // debugPrintfEXT("TS SUBDIVIDE: Half-block %u_%u%u%u still too large. Quartering...\n", 
                    //               compactedBlockID, ox, oy, oz);
                    
                    uvec3 quarterBlockDim = max(uvec3(1), currentHalfDim / 2u);
                    bool emittedAny = false;
                    
                    for (uint oz2 = 0u; oz2 < 2u; ++oz2) {
                        for (uint oy2 = 0u; oy2 < 2u; ++oy2) {
                            for (uint ox2 = 0u; ox2 < 2u; ++ox2) {
                                uvec3 currentQuarterDim = uvec3(
                                    (ox2 == 0) ? quarterBlockDim.x : (currentHalfDim.x - quarterBlockDim.x),
                                    (oy2 == 0) ? quarterBlockDim.y : (currentHalfDim.y - quarterBlockDim.y),
                                    (oz2 == 0) ? quarterBlockDim.z : (currentHalfDim.z - quarterBlockDim.z)
                                );
                                
                                if (currentQuarterDim.x == 0 || currentQuarterDim.y == 0 || currentQuarterDim.z == 0) {
                                    continue;
                                }
                                
                                uvec3 quarterOrigin = halfOrigin + 
                                                     uvec3(ox2 * quarterBlockDim.x, 
                                                           oy2 * quarterBlockDim.y, 
                                                           oz2 * quarterBlockDim.z);

                                uint estVertsQuarter = estimateVertsInSubBlockWithSharing(quarterOrigin, currentQuarterDim);
                                uint estPrimsQuarter = estimatePrimsInSubBlock(quarterOrigin, currentQuarterDim);

                                if (estVertsQuarter == 0 && estPrimsQuarter == 0) continue;

                                // debugPrintfEXT("TS EMIT: Quarter-block %u_%u%u%u_%u%u%u. EstV=%u, EstP=%u.\n", 
                                //               compactedBlockID, ox, oy, oz, ox2, oy2, oz2, 
                                //               estVertsQuarter, estPrimsQuarter);
                                emitSubBlock(quarterOrigin, currentQuarterDim, compactedBlockID, 
                                           estVertsQuarter, estPrimsQuarter);
                                emittedAny = true;
                            }
                        }
                    }
                    
                    // Fallback: emit half-block if no quarters were emitted
                    if (!emittedAny) {
                        // debugPrintfEXT("TS FALLBACK: Emitting oversized half-block %u_%u%u%u.\n", 
                        //               compactedBlockID, ox, oy, oz);
                        emitSubBlock(halfOrigin, currentHalfDim, compactedBlockID, estVertsHalf, estPrimsHalf);
                    }
                }
            }
        }
    }
}