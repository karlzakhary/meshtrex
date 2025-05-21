#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_debug_printf : enable

#define CORE_BLOCK_DIM_X 8 // Assuming ubo.blockDim.xyz refers to core dimensions
#define CORE_BLOCK_DIM_Y 8
#define CORE_BLOCK_DIM_Z 8
#define CONTEXT_CELLS_PER_SIDE 1 // How many layers of context cells on each side

#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5
#define MAX_MESHLET_VERTS_ESTIMATE 64u // For subdivision decision (on core block)
#define MAX_MESHLET_PRIMS_ESTIMATE 126u // For subdivision decision (on core block)
#define MAX_TRI_INDICES 16
#define WORKGROUP_SIZE 32

struct TaskPayload {
    uint globalVertexOffset_reservation; // For meshlet's local SetMeshOutputsEXT (less critical now)
    uint globalIndexOffset_reservation;  // For meshlet's local SetMeshOutputsEXT

    uvec3 coreBlockOrigin_global;   // Global origin of the CORE cells
    uvec3 coreBlockDim;             // Dimensions of the CORE cells
    uvec3 processingBlockOrigin_global; // Global origin of the block including context cells (coreOrigin - CONTEXT_CELLS_PER_SIDE)
    uvec3 processingBlockDim_total; // Dimensions of the block including context (coreDim + 2 * CONTEXT_CELLS_PER_SIDE)

    uint taskId;                    // Original compactedBlockID
    // Removed globalMeshletDescOffset, as mesh shader will use a new global counter for filled descriptors
};

taskPayloadSharedEXT TaskPayload taskPayloadOut;

// Bindings (ensure these match your C++ setup)
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;        // Global volume dimensions
    uvec4 blockDim;         // THIS IS THE CORE BLOCK DIMENSION (e.g., 8,8,8)
    uvec4 blockGridDim;     // Grid of core blocks
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount_SSBO { uint count; } activeBlockCountBuffer;
layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs_SSBO { uint compactedBlkArray[]; } blockIds;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;

// These counters are for the *meshlet's local output* via SetMeshOutputsEXT,
// which is less important if the final mesh uses global IDs from SSBOs.
// For safety, we can still allocate some space for this local output.
layout(set = 0, binding = 7, std430) buffer LocalVertexCount_SSBO { uint counter; } localVCount;
layout(set = 0, binding = 9, std430) buffer LocalIndexCount_SSBO { uint counter; } localICount;

// Task shader no longer directly manages the final meshlet descriptor count or offsets.
// Mesh shader will use a global counter for filled descriptors.

// Helper functions (sampleMcCase, countCellVertices, countCellPrimitives,
// estimateVertsInSubBlock, estimatePrimsInSubBlock) - these should operate on CORE dimensions
// for making subdivision decisions. The actual vertex/index allocation for meshlet's
// temporary storage might need to consider the processingBlockDim_total if all those cells
// temporarily store vertices before global ID mapping.
// ... (Functions remain the same as previous version, ensure they use the coreDim for estimates) ...
uint sampleMcCase(uvec3 cell_global_origin) { /* ... same ... */
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
uint countCellVertices(uint cubeCase) { /* ... same ... */
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
uint countCellPrimitives(uint cubeCase) { /* ... same ... */
    uint primCount = 0;
    int base = int(cubeCase) * MAX_TRI_INDICES;
    for (int i = 0; i < MAX_TRI_INDICES; i += 3) {
        if (mc.triTable[base + i] < 0) break;
        primCount++;
    }
    return primCount;
}
uint estimateVertsInSubBlock(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) { /* ... same ... (operates on core) */
    uint sum = 0;
    bool limitExceeded = false;
    if (gl_LocalInvocationIndex == 0) {
        for (uint z = 0; z < subBlockDimensions.z; ++z) {
            for (uint y = 0; y < subBlockDimensions.y; ++y) {
                for (uint x = 0; x < subBlockDimensions.x; ++x) {
                    uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                    uint cubeCase = sampleMcCase(cellGlobalOrigin);
                    sum += countCellVertices(cubeCase);
                    if (sum > MAX_MESHLET_VERTS_ESTIMATE) { // Use estimate limit
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
uint estimatePrimsInSubBlock(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) { /* ... same ... (operates on core) */
    uint sum = 0;
    bool limitExceeded = false;
    if (gl_LocalInvocationIndex == 0) {
        for (uint z = 0; z < subBlockDimensions.z; ++z) {
            for (uint y = 0; y < subBlockDimensions.y; ++y) {
                for (uint x = 0; x < subBlockDimensions.x; ++x) {
                    uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                    uint cubeCase = sampleMcCase(cellGlobalOrigin);
                    sum += countCellPrimitives(cubeCase);
                    if (sum > MAX_MESHLET_PRIMS_ESTIMATE) { // Use estimate limit
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

void dispatchMesh(uvec3 currentCoreBlockOrigin, uvec3 currentCoreBlockDim, uint currentCompactedBlockID) {
    // Calculate context dimensions
    uvec3 processingDimTotal = currentCoreBlockDim + uvec3(CONTEXT_CELLS_PER_SIDE * 2);
    ivec3 processingOriginGlobal_signed = ivec3(currentCoreBlockOrigin) - ivec3(CONTEXT_CELLS_PER_SIDE);

    // Clamp processingOriginGlobal_signed and adjust processingDimTotal to stay within volume bounds
    // This ensures mesh shader doesn't read out of volume texture.
    uvec3 volDim = ubo.volumeDim.xyz;
    for(int i=0; i<3; ++i) {
        if (processingOriginGlobal_signed[i] < 0) {
            processingDimTotal[i] -= uint(-processingOriginGlobal_signed[i]);
            processingOriginGlobal_signed[i] = 0;
        }
        if (processingOriginGlobal_signed[i] + processingDimTotal[i] > volDim[i]) {
            processingDimTotal[i] = volDim[i] - processingOriginGlobal_signed[i];
        }
        processingDimTotal[i] = max(1u, processingDimTotal[i]); // Ensure not zero
    }
    uvec3 processingOriginGlobal = uvec3(processingOriginGlobal_signed);

    // For local SetMeshOutputsEXT (max 256 verts, 128 prims)
    // Allocation can be smaller if we rely on global vertex IDs for final mesh
    uint cells_for_local_alloc = currentCoreBlockDim.x * currentCoreBlockDim.y * currentCoreBlockDim.z; // Or processingDimTotal if temp storing all
    uint mV_local = min(MAX_VERTS_PER_CELL * cells_for_local_alloc, MAX_MESHLET_VERTS_ESTIMATE * 2); // Generous local buffer
    uint mI_local = min(MAX_PRIMS_PER_CELL * 3 * cells_for_local_alloc, MAX_MESHLET_PRIMS_ESTIMATE * 3 * 2);

    uint vOff_local = atomicAdd(localVCount.counter, mV_local);
    uint iOff_local = atomicAdd(localICount.counter, mI_local);

    taskPayloadOut = TaskPayload(
    vOff_local, iOff_local,
    currentCoreBlockOrigin,
    currentCoreBlockDim,
    processingOriginGlobal, // Pass the origin of the extended block
    processingDimTotal,     // Pass the dimensions of the extended block
    currentCompactedBlockID
    );
    EmitMeshTasksEXT(1, 1, 1);
}


void main() {
    if (gl_LocalInvocationIndex != 0u) return;

    uint compactedBlockID = gl_WorkGroupID.x;
    if (compactedBlockID >= activeBlockCountBuffer.count) return;

    uint originalBlockIndex = blockIds.compactedBlkArray[compactedBlockID];

    uvec3 initialCoreBlockGlobalOrigin;
    initialCoreBlockGlobalOrigin.z = originalBlockIndex / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = originalBlockIndex % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    initialCoreBlockGlobalOrigin.y = sliceIndex / ubo.blockGridDim.x;
    initialCoreBlockGlobalOrigin.x = sliceIndex % ubo.blockGridDim.x;
    initialCoreBlockGlobalOrigin *= ubo.blockDim.xyz;

    uvec3 coreFullBlockDim = ubo.blockDim.xyz; // This is the 'full' block from initial grid

    // Estimate based on CORE cells of the full block
    uint estVertsFull = estimateVertsInSubBlock(initialCoreBlockGlobalOrigin, coreFullBlockDim);
    uint estPrimsFull = estimatePrimsInSubBlock(initialCoreBlockGlobalOrigin, coreFullBlockDim);

    if (estVertsFull == 0 && estPrimsFull == 0) return;

    if (estVertsFull <= MAX_MESHLET_VERTS_ESTIMATE && estPrimsFull <= MAX_MESHLET_PRIMS_ESTIMATE) {
        dispatchMesh(initialCoreBlockGlobalOrigin, coreFullBlockDim, compactedBlockID);
        return;
    }

    // Subdivide (logic for subdivision remains, but calls dispatchMesh with core info)
    uvec3 coreHalfBlockDim = max(uvec3(1), coreFullBlockDim / 2u);
    uvec3 coreQuarterBlockDim = max(uvec3(1), coreFullBlockDim / 4u);

    for (uint oz = 0u; oz < 2u; ++oz) {
        for (uint oy = 0u; oy < 2u; ++oy) {
            for (uint ox = 0u; ox < 2u; ++ox) {
                uvec3 currentCoreHalfDim = uvec3(
                (coreFullBlockDim.x == 1u || coreHalfBlockDim.x == 0u) ? 1u : (ox == 0u ? coreHalfBlockDim.x : coreFullBlockDim.x - coreHalfBlockDim.x),
                (coreFullBlockDim.y == 1u || coreHalfBlockDim.y == 0u) ? 1u : (oy == 0u ? coreHalfBlockDim.y : coreFullBlockDim.y - coreHalfBlockDim.y),
                (coreFullBlockDim.z == 1u || coreHalfBlockDim.z == 0u) ? 1u : (oz == 0u ? coreHalfBlockDim.z : coreFullBlockDim.z - coreHalfBlockDim.z)
                );
                currentCoreHalfDim = max(uvec3(1u), currentCoreHalfDim);
                uvec3 currentCoreHalfOrigin = initialCoreBlockGlobalOrigin + uvec3(ox * coreHalfBlockDim.x, oy * coreHalfBlockDim.y, oz * coreHalfBlockDim.z);

                uint estVertsHalf = estimateVertsInSubBlock(currentCoreHalfOrigin, currentCoreHalfDim);
                uint estPrimsHalf = estimatePrimsInSubBlock(currentCoreHalfOrigin, currentCoreHalfDim);

                if (estVertsHalf == 0 && estPrimsHalf == 0) continue;

                if (estVertsHalf <= MAX_MESHLET_VERTS_ESTIMATE && estPrimsHalf <= MAX_MESHLET_PRIMS_ESTIMATE) {
                    dispatchMesh(currentCoreHalfOrigin, currentCoreHalfDim, compactedBlockID);
                } else {
                    bool canBeQuartered = (currentCoreHalfDim.x >= 2u && currentCoreHalfDim.y >= 2u && currentCoreHalfDim.z >= 2u);
                    if (!canBeQuartered) {
                        dispatchMesh(currentCoreHalfOrigin, currentCoreHalfDim, compactedBlockID);
                        continue;
                    }

                    bool emittedAnyQuarter = false;
                    for (uint oz2 = 0u; oz2 < 2u; ++oz2) {
                        for (uint oy2 = 0u; oy2 < 2u; ++oy2) {
                            for (uint ox2 = 0u; ox2 < 2u; ++ox2) {
                                uvec3 currentCoreQuarterDim = uvec3(
                                (currentCoreHalfDim.x == 1u || coreQuarterBlockDim.x == 0u) ? 1u : (ox2 == 0u ? coreQuarterBlockDim.x : currentCoreHalfDim.x - coreQuarterBlockDim.x),
                                (currentCoreHalfDim.y == 1u || coreQuarterBlockDim.y == 0u) ? 1u : (oy2 == 0u ? coreQuarterBlockDim.y : currentCoreHalfDim.y - coreQuarterBlockDim.y),
                                (currentCoreHalfDim.z == 1u || coreQuarterBlockDim.z == 0u) ? 1u : (oz2 == 0u ? coreQuarterBlockDim.z : currentCoreHalfDim.z - coreQuarterBlockDim.z)
                                );
                                currentCoreQuarterDim = max(uvec3(1u), currentCoreQuarterDim);
                                uvec3 currentCoreQuarterOrigin = currentCoreHalfOrigin + uvec3(ox2 * coreQuarterBlockDim.x, oy2 * coreQuarterBlockDim.y, oz2 * coreQuarterBlockDim.z);

                                uint estVQuarter = estimateVertsInSubBlock(currentCoreQuarterOrigin, currentCoreQuarterDim);
                                uint estPQaurter = estimatePrimsInSubBlock(currentCoreQuarterOrigin, currentCoreQuarterDim);

                                if (estVQuarter == 0 && estPQaurter == 0) continue;

                                dispatchMesh(currentCoreQuarterOrigin, currentCoreQuarterDim, compactedBlockID);
                                emittedAnyQuarter = true;
                            }
                        }
                    }
                    if (!emittedAnyQuarter && (estVertsHalf > 0 || estPrimsHalf > 0)) {
                        debugPrintfEXT("TS WARN FALLBACK: Task %u Dense half-block origin (%u,%u,%u) dim (%u,%u,%u) had EstV=%u, EstP=%u BUT NO quarter tasks emitted. Emitting half-block.\n", compactedBlockID, currentCoreHalfOrigin.x, currentCoreHalfOrigin.y, currentCoreHalfOrigin.z, currentCoreHalfDim.x,currentCoreHalfDim.y,currentCoreHalfDim.z, estVertsHalf, estPrimsHalf);
                        dispatchMesh(currentCoreHalfOrigin, currentCoreHalfDim, compactedBlockID);
                    }
                }
            }
        }
    }
}