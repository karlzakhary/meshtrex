#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require // For subgroupBroadcastFirst
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 8
#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5
#define MAX_MESHLET_VERTS 256u
#define MAX_MESHLET_PRIMS 128u // Aligned with mesh shader
#define MAX_TRI_INDICES 16    // From Marching Cubes table structure
#define WORKGROUP_SIZE 32

// --- Structures ---
struct TaskPayload {
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletDescOffset;
    uvec3 blockOrigin;      // Absolute origin in volume coordinates OF THE SUB-BLOCK
    uvec3 subBlockDim;      // Dimensions of the sub-block for the mesh shader
    uint taskId;            // Original compactedBlockID
};

taskPayloadSharedEXT TaskPayload taskPayloadOut;

// --- Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;         // Default dimensions of blocks processed by task shader (e.g., 8,8,8)
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;

layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount_SSBO { // Assuming this exists for bounds check
                                                                             uint count;
} activeBlockCountBuffer;

layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs_SSBO {
    uint compactedBlkArray[];
} blockIds;

layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO {
    int triTable[];
} mc;

// Using countVertices (per-cell unique edges) instead of a separate numVertsTable SSBO
// layout(set=0, binding=5, std430) readonly buffer NumVertsTable { uint numVertsTable[]; } nvt;

layout(set = 0, binding = 7, std430) buffer VertexCount_SSBO { uint vertexCounter; } vCount;
layout(set = 0, binding = 9, std430) buffer IndexCount_SSBO { uint indexCounter; } iCount;
layout(set = 0, binding = 11, std430) buffer MeshletDescCount_SSBO { uint meshletCounter; } meshletCount;

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
    // Estimates vertices by unique edges for THIS cell
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
    // Counts triangles for THIS cell
    uint primCount = 0;
    int base = int(cubeCase) * MAX_TRI_INDICES;
    for (int i = 0; i < MAX_TRI_INDICES; i += 3) {
        if (mc.triTable[base + i] < 0) break;
        primCount++;
    }
    return primCount;
}

uint estimateVertsInSubBlock(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) {
    uint sum = 0;
    bool limitExceeded = false;
    if (gl_LocalInvocationIndex == 0) {
        // Estimation done by one thread
        for (uint z = 0; z < subBlockDimensions.z; ++z) {
            for (uint y = 0; y < subBlockDimensions.y; ++y) {
                for (uint x = 0; x < subBlockDimensions.x; ++x) {
                    uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                    uint cubeCase = sampleMcCase(cellGlobalOrigin);
                    sum += countCellVertices(cubeCase); // Using per-cell vertex estimate
                    if (sum > MAX_MESHLET_VERTS) {
                        limitExceeded = true;
                        break;
                    }
                }
                if (limitExceeded) break;
            }
            if (limitExceeded) break;
        }
    }
    return sum; // Only thread 0 uses this directly, no broadcast needed if logic is self-contained
}

uint estimatePrimsInSubBlock(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) {
    uint sum = 0;
    bool limitExceeded = false;
    if (gl_LocalInvocationIndex == 0) {
        // Estimation done by one thread
        for (uint z = 0; z < subBlockDimensions.z; ++z) {
            for (uint y = 0; y < subBlockDimensions.y; ++y) {
                for (uint x = 0; x < subBlockDimensions.x; ++x) {
                    uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                    uint cubeCase = sampleMcCase(cellGlobalOrigin);
                    sum += countCellPrimitives(cubeCase);
                    if (sum > MAX_MESHLET_PRIMS) {
                        limitExceeded = true;
                        break;
                    }
                }
                if (limitExceeded) break;
            }
            if (limitExceeded) break;
        }
    }
    return sum;
}

// --- Task Shader Main ---
layout(local_size_x = WORKGROUP_SIZE) in;
void main() {
    if (gl_LocalInvocationIndex != 0u) {
        return;
    }

    uint compactedBlockID = gl_WorkGroupID.x;
    if (compactedBlockID >= activeBlockCountBuffer.count) {
         debugPrintfEXT("TS WARN: compactedBlockID %u >= activeBlockCount %u. Skipping.\n", compactedBlockID, activeBlockCountBuffer.count);
        return;
    }

    uint originalBlockIndex = blockIds.compactedBlkArray[compactedBlockID];

    uvec3 fullBlockGlobalOrigin; // Origin of the initial 8x8x8 (or whatever ubo.blockDim is) block
    fullBlockGlobalOrigin.z = originalBlockIndex / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = originalBlockIndex % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    fullBlockGlobalOrigin.y = sliceIndex / ubo.blockGridDim.x;
    fullBlockGlobalOrigin.x = sliceIndex % ubo.blockGridDim.x;
    fullBlockGlobalOrigin *= ubo.blockDim.xyz;

    uvec3 fullBlockDim = ubo.blockDim.xyz;
    uvec3 halfBlockDim = max(uvec3(1), fullBlockDim / 2u);
    uvec3 quarterBlockDim = max(uvec3(1), fullBlockDim / 4u);

    uint estVertsFull = estimateVertsInSubBlock(fullBlockGlobalOrigin, fullBlockDim);
    uint estPrimsFull = estimatePrimsInSubBlock(fullBlockGlobalOrigin, fullBlockDim);

    if (estVertsFull == 0 && estPrimsFull == 0) {
         debugPrintfEXT("TS INFO: Full block ID %u (origin %u,%u,%u) is empty. EstV=%u, EstP=%u.\n", compactedBlockID, fullBlockGlobalOrigin.x, fullBlockGlobalOrigin.y, fullBlockGlobalOrigin.z, estVertsFull, estPrimsFull);
        return;
    }

    if (estVertsFull <= MAX_MESHLET_VERTS && estPrimsFull <= MAX_MESHLET_PRIMS) {
        // debugPrintfEXT("TS EMIT: Full block ID %u (origin %u,%u,%u dim %u,%u,%u). EstV=%u, EstP=%u.\n", compactedBlockID, fullBlockGlobalOrigin.x, fullBlockGlobalOrigin.y, fullBlockGlobalOrigin.z, fullBlockDim.x, fullBlockDim.y, fullBlockDim.z, estVertsFull, estPrimsFull);
        uint cells = fullBlockDim.x * fullBlockDim.y * fullBlockDim.z;
        uint mV = cells * MAX_VERTS_PER_CELL;
        uint mI = cells * MAX_PRIMS_PER_CELL * 3u;
        uint vOff = atomicAdd(vCount.vertexCounter, mV);
        uint iOff = atomicAdd(iCount.indexCounter, mI);
        uint dOff = atomicAdd(meshletCount.meshletCounter, 1u);
        taskPayloadOut = TaskPayload(vOff, iOff, dOff, fullBlockGlobalOrigin, fullBlockDim, compactedBlockID);
        EmitMeshTasksEXT(1, 1, 1);
        return;
    }

    // debugPrintfEXT("TS SUBDIVIDE: Full block ID %u (origin %u,%u,%u) too large (EstV=%u, EstP=%u). Subdividing to half...\n", compactedBlockID, fullBlockGlobalOrigin.x, fullBlockGlobalOrigin.y, fullBlockGlobalOrigin.z, estVertsFull, estPrimsFull);
    for (uint oz = 0u; oz < 2u; ++oz) {
        for (uint oy = 0u; oy < 2u; ++oy) {
            for (uint ox = 0u; ox < 2u; ++ox) {
                uvec3 currentHalfSubBlockDim = uvec3(
                (fullBlockDim.x == 1) ? 1 : (ox == 0 ? halfBlockDim.x : fullBlockDim.x - halfBlockDim.x),
                (fullBlockDim.y == 1) ? 1 : (oy == 0 ? halfBlockDim.y : fullBlockDim.y - halfBlockDim.y),
                (fullBlockDim.z == 1) ? 1 : (oz == 0 ? halfBlockDim.z : fullBlockDim.z - halfBlockDim.z)
                );
                if (ox > 0 && currentHalfSubBlockDim.x == halfBlockDim.x && fullBlockDim.x % 2u != 0 && fullBlockDim.x > 1)  {
                    currentHalfSubBlockDim.x = max(1u, currentHalfSubBlockDim.x -1); // crude fix for uneven
                }
                // Similar for y, z if strict non-overlapping is needed for uneven blocks. For 8x8x8, this simplifies.

                uvec3 halfSubBlockGlobalOrigin = fullBlockGlobalOrigin +
                    uvec3(ox * halfBlockDim.x, oy * halfBlockDim.y, oz * halfBlockDim.z);

                uint estVertsHalf = estimateVertsInSubBlock(halfSubBlockGlobalOrigin, currentHalfSubBlockDim);
                uint estPrimsHalf = estimatePrimsInSubBlock(halfSubBlockGlobalOrigin, currentHalfSubBlockDim);

                if (estVertsHalf == 0 && estPrimsHalf == 0) continue;

                if (estVertsHalf <= MAX_MESHLET_VERTS && estPrimsHalf <= MAX_MESHLET_PRIMS) {
                    // debugPrintfEXT("TS EMIT: Half-block for ID %u (offset %u%u%u from origin %u,%u,%u dim %u,%u,%u). EstV=%u, EstP=%u.\n", compactedBlockID, ox,oy,oz, halfSubBlockGlobalOrigin.x, halfSubBlockGlobalOrigin.y, halfSubBlockGlobalOrigin.z, currentHalfSubBlockDim.x, currentHalfSubBlockDim.y, currentHalfSubBlockDim.z, estVertsHalf, estPrimsHalf);
                    uint cells = currentHalfSubBlockDim.x * currentHalfSubBlockDim.y * currentHalfSubBlockDim.z;
                    uint mV = cells * MAX_VERTS_PER_CELL;
                    uint mI = cells * MAX_PRIMS_PER_CELL * 3u;
                    uint vOff = atomicAdd(vCount.vertexCounter, mV);
                    uint iOff = atomicAdd(iCount.indexCounter, mI);
                    uint dOff = atomicAdd(meshletCount.meshletCounter, 1u);
                    taskPayloadOut = TaskPayload(vOff, iOff, dOff, halfSubBlockGlobalOrigin, currentHalfSubBlockDim, compactedBlockID);
                    EmitMeshTasksEXT(1, 1, 1);
                } else {
                    // debugPrintfEXT("TS SUBDIVIDE: Half-block for ID %u (offset %u%u%u from origin %u,%u,%u) too large (EstV=%u, EstP=%u). Subdividing to quarter...\n", compactedBlockID, ox,oy,oz, halfSubBlockGlobalOrigin.x, halfSubBlockGlobalOrigin.y, halfSubBlockGlobalOrigin.z, estVertsHalf, estPrimsHalf);
                    bool canBeQuartered = (currentHalfSubBlockDim.x >= 2u && currentHalfSubBlockDim.y >= 2u && currentHalfSubBlockDim.z >= 2u);
                    if (!canBeQuartered) {
//                        debugPrintfEXT("TS EMIT WARN: Dense half-block for ID %u (offset %u%u%u from origin %u,%u,%u dim %u,%u,%u) cannot be quartered. Emitting as half. EstV=%u, EstP=%u.\n", compactedBlockID, ox,oy,oz, halfSubBlockGlobalOrigin.x,halfSubBlockGlobalOrigin.y,halfSubBlockGlobalOrigin.z, currentHalfSubBlockDim.x, currentHalfSubBlockDim.y, currentHalfSubBlockDim.z, estVertsHalf, estPrimsHalf);
                        debugPrintfEXT("TS EMIT WARN: Dense half-block, cannot be quartered 1");
                        uint cells = currentHalfSubBlockDim.x * currentHalfSubBlockDim.y * currentHalfSubBlockDim.z;
                        uint mV = cells * MAX_VERTS_PER_CELL;
                        uint mI = cells * MAX_PRIMS_PER_CELL * 3u;
                        uint vOff = atomicAdd(vCount.vertexCounter, mV);
                        uint iOff = atomicAdd(iCount.indexCounter, mI);
                        uint dOff = atomicAdd(meshletCount.meshletCounter, 1u);
                        taskPayloadOut = TaskPayload(vOff, iOff, dOff, halfSubBlockGlobalOrigin, currentHalfSubBlockDim, compactedBlockID);
                        EmitMeshTasksEXT(1, 1, 1);
                        continue;
                    }

                    bool emittedAnyQuarterTask = false;
                    for (uint oz2 = 0u; oz2 < 2u; ++oz2) {
                        for (uint oy2 = 0u; oy2 < 2u; ++oy2) {
                            for (uint ox2 = 0u; ox2 < 2u; ++ox2) {
                                uvec3 currentQuarterSubBlockDim = uvec3(
                                (currentHalfSubBlockDim.x == 1) ? 1 : (ox2 == 0 ? quarterBlockDim.x : currentHalfSubBlockDim.x - quarterBlockDim.x),
                                (currentHalfSubBlockDim.y == 1) ? 1 : (oy2 == 0 ? quarterBlockDim.y : currentHalfSubBlockDim.y - quarterBlockDim.y),
                                (currentHalfSubBlockDim.z == 1) ? 1 : (oz2 == 0 ? quarterBlockDim.z : currentHalfSubBlockDim.z - quarterBlockDim.z)
                                );

                                uvec3 quarterSubBlockGlobalOrigin = halfSubBlockGlobalOrigin + uvec3(ox2 * quarterBlockDim.x, oy2 * quarterBlockDim.y, oz2 * quarterBlockDim.z);

                                uint estVertsQuarter = estimateVertsInSubBlock(quarterSubBlockGlobalOrigin, currentQuarterSubBlockDim);
                                uint estPrimsQuarter = estimatePrimsInSubBlock(quarterSubBlockGlobalOrigin, currentQuarterSubBlockDim);

                                if (estVertsQuarter == 0 && estPrimsQuarter == 0) continue;

                                // Always emit non-empty quarter blocks. Mesh shader will clamp if estimates were off.
                                // debugPrintfEXT("TS EMIT: Quarter-block for ID %u (offset %u%u%u -> %u%u%u from origin %u,%u,%u dim %u,%u,%u). EstV=%u, EstP=%u.\n", compactedBlockID, ox,oy,oz, ox2,oy2,oz2, quarterSubBlockGlobalOrigin.x, quarterSubBlockGlobalOrigin.y, quarterSubBlockGlobalOrigin.z, currentQuarterSubBlockDim.x,currentQuarterSubBlockDim.y,currentQuarterSubBlockDim.z, estVertsQuarter, estPrimsQuarter);
                                uint cells = currentQuarterSubBlockDim.x * currentQuarterSubBlockDim.y * currentQuarterSubBlockDim.z;
                                uint mV = cells * MAX_VERTS_PER_CELL;
                                uint mI = cells * MAX_PRIMS_PER_CELL * 3u;
                                uint vOff = atomicAdd(vCount.vertexCounter, mV);
                                uint iOff = atomicAdd(iCount.indexCounter, mI);
                                uint dOff = atomicAdd(meshletCount.meshletCounter, 1u);
                                taskPayloadOut = TaskPayload(vOff, iOff, dOff, quarterSubBlockGlobalOrigin, currentQuarterSubBlockDim, compactedBlockID);
                                EmitMeshTasksEXT(1, 1, 1);
                                emittedAnyQuarterTask = true;
                            }
                        }
                    }
                    if (!emittedAnyQuarterTask && (estVertsHalf > 0 || estPrimsHalf > 0)) {
//                        debugPrintfEXT("TS WARN FALLBACK: Dense half-block for ID %u (offset %u %u %u from origin %u,%u,%u) had EstV=%u, EstP=%u BUT NO quarter tasks emitted. Emitting half-block.\n", compactedBlockID, ox,oy,oz, halfSubBlockGlobalOrigin.x, halfSubBlockGlobalOrigin.y, halfSubBlockGlobalOrigin.z, estVertsHalf, estPrimsHalf);
                        debugPrintfEXT("TS WARN FALLBACK: Dense half-block 2");
                        uint cells = currentHalfSubBlockDim.x * currentHalfSubBlockDim.y * currentHalfSubBlockDim.z;
                        uint mV = cells * MAX_VERTS_PER_CELL;
                        uint mI = cells * MAX_PRIMS_PER_CELL * 3u;
                        uint vOff = atomicAdd(vCount.vertexCounter, mV);
                        uint iOff = atomicAdd(iCount.indexCounter, mI);
                        uint dOff = atomicAdd(meshletCount.meshletCounter, 1u);
                        taskPayloadOut = TaskPayload(vOff, iOff, dOff, halfSubBlockGlobalOrigin, currentHalfSubBlockDim, compactedBlockID);
                        EmitMeshTasksEXT(1, 1, 1);
                    }
                }
            }
        }
    }
}