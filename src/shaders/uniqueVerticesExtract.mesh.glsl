#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#define MAX_VERTICES 256u
#define MAX_PRIMITIVES 128u
#define WORKGROUP_SIZE 32

// Max dimensions of a CORE sub-block (from ubo.blockDim, e.g., 8x8x8)
// The actual processing dimension for vertices will be CORE_DIM + 1
#define MAX_CORE_SUB_BLOCK_X 8
#define MAX_CORE_SUB_BLOCK_Y 8
#define MAX_CORE_SUB_BLOCK_Z 8

// --- Structures ---
// ... (MeshletDescriptor, VertexData, TaskPayload same as before) ...
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

struct TaskPayload {
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletDescOffset; // Offset into the RESERVED descriptor slots
    uvec3 blockOrigin;      // Global origin of the CORE sub-block
    uvec3 subBlockDim;      // Dimensions of the CORE sub-block
    uint taskId;            // Original compactedBlockID
};

taskPayloadSharedEXT TaskPayload taskPayloadIn;


// --- Bindings ---
// ... (UBO, volumeImage, mc.triTable, SSBOs for vertices, indices same as before) ...
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;
layout(set = 0, binding = 6, std430) buffer Vertex_SSBO { VertexData vertex_data[]; } vertices_ssbo;
layout(set = 0, binding = 8, std430) buffer Index_SSBO { uint indices[]; } indices_ssbo;

// Output SSBO for Meshlet Descriptors (written by this shader)
layout(set = 0, binding = 10, std430) buffer MeshletDescOutput_SSBO { MeshletDescriptor meshletDescriptors[]; } meshletDescOutput;

// Counter for ACTUALLY WRITTEN (filled) meshlet descriptors
layout(set = 0, binding = 12, std430) buffer FilledMeshletDescCount_SSBO { uint filledMeshletCounter; } filledMeshletDescCount;


// --- Mesh Shader Output ---
layout(local_size_x = WORKGROUP_SIZE) in;
layout(max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;
layout(triangles) out;

layout(location = 0) out PerVertexData { vec3 normal; } meshVertexDataOut[];

// --- Shared Memory ---
shared VertexData sharedMeshletVertices[MAX_VERTICES];
shared uvec3 sharedMeshletIndices[MAX_PRIMITIVES];
shared uint sharedMeshletVertexCount_actual;
shared uint sharedMeshletPrimitiveCount_actual;

// For PMB-style distinct edge vertex deduplication
// Stores the local meshlet vertex index (0..MAX_VERTICES-1) for a vertex
// generated on a distinct edge of a cell.
// The dimensions here must accommodate cells up to coreSubBlockDim +1 for context.
// So, if max core is 8x8x8, we might need to cache for 9x9x9 cells.
#define EXT_DIM_X (MAX_CORE_SUB_BLOCK_X + 1)
#define EXT_DIM_Y (MAX_CORE_SUB_BLOCK_Y + 1)
#define EXT_DIM_Z (MAX_CORE_SUB_BLOCK_Z + 1)
// This array stores the LOCAL MESHLET INDEX of the vertex generated
// by cell (x,y,z within extended block) on its 'axis'-th distinct edge.
shared uint distinctEdgeOwner_VertexIdx[EXT_DIM_Z * EXT_DIM_Y * EXT_DIM_X * 3];


// --- Distinct Edge Helper Structures & Data (PMB Style) ---
struct EdgeMap {
    ivec3 ownerCellOffset_RelativeToCurrentMCcell; // Offset from current MC cell to the cell that "owns" this edge's vertex
    int distinctEdgeAxis_OfOwner;                 // 0 for X, 1 for Y, 2 for Z axis of that owner cell
};

const EdgeMap neighborMappingTable[12] = {
EdgeMap(ivec3(0,0,0), 0), // Edge 0: P0-P1, X-axis of P0=(0,0,0) within cell
EdgeMap(ivec3(1,0,0), 1), // Edge 1: P1-P2, Y-axis of P1=(1,0,0) relative to cell P0
EdgeMap(ivec3(0,1,0), 0), // Edge 2: P3-P2, X-axis of P3=(0,1,0) relative to cell P0
EdgeMap(ivec3(0,0,0), 1), // Edge 3: P0-P3, Y-axis of P0=(0,0,0)
EdgeMap(ivec3(0,0,1), 0), // Edge 4: P4-P5, X-axis of P4=(0,0,1)
EdgeMap(ivec3(1,0,1), 1), // Edge 5: P5-P6, Y-axis of P5=(1,0,1)
EdgeMap(ivec3(0,1,1), 0), // Edge 6: P7-P6, X-axis of P7=(0,1,1)
EdgeMap(ivec3(0,0,1), 1), // Edge 7: P4-P7, Y-axis of P4=(0,0,1)
EdgeMap(ivec3(0,0,0), 2), // Edge 8: P0-P4, Z-axis of P0=(0,0,0)
EdgeMap(ivec3(1,0,0), 2), // Edge 9: P1-P5, Z-axis of P1=(1,0,0)
EdgeMap(ivec3(1,1,0), 2), // Edge 10: P2-P6, Z-axis of P2=(1,1,0)
EdgeMap(ivec3(0,1,0), 2)  // Edge 11: P3-P7, Z-axis of P3=(0,1,0)
};

// ... (getEdgeInfo, interpolateVertex, calculateNormal functions - same as before) ...
void getEdgeInfo(int edgeIndex, out ivec3 p1_offset, out ivec3 p2_offset, out int corner1_idx, out int corner2_idx, out int axis) {
    switch (edgeIndex) { /* ... same as before ... */
        case 0:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(1,0,0); corner1_idx=0; corner2_idx=1; axis=0; return;
        case 1:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,1,0); corner1_idx=1; corner2_idx=2; axis=1; return;
        case 2:  p1_offset=ivec3(0,1,0); p2_offset=ivec3(1,1,0); corner1_idx=3; corner2_idx=2; axis=0; return;
        case 3:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,1,0); corner1_idx=0; corner2_idx=3; axis=1; return;
        case 4:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(1,0,1); corner1_idx=4; corner2_idx=5; axis=0; return;
        case 5:  p1_offset=ivec3(1,0,1); p2_offset=ivec3(1,1,1); corner1_idx=5; corner2_idx=6; axis=1; return;
        case 6:  p1_offset=ivec3(0,1,1); p2_offset=ivec3(1,1,1); corner1_idx=7; corner2_idx=6; axis=0; return;
        case 7:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(0,1,1); corner1_idx=4; corner2_idx=7; axis=1; return;
        case 8:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,0,1); corner1_idx=0; corner2_idx=4; axis=2; return;
        case 9:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,0,1); corner1_idx=1; corner2_idx=5; axis=2; return;
        case 10: p1_offset=ivec3(1,1,0); p2_offset=ivec3(1,1,1); corner1_idx=2; corner2_idx=6; axis=2; return;
        case 11: p1_offset=ivec3(0,1,0); p2_offset=ivec3(0,1,1); corner1_idx=3; corner2_idx=7; axis=2; return;
        default: p1_offset=ivec3(0); p2_offset=ivec3(0); corner1_idx=-1; corner2_idx=-1; axis=-1; return;
    }
}

vec3 interpolateVertex(ivec3 p1_global, ivec3 p2_global, float val1, float val2) {
    if (abs(val1 - val2) < 1e-6f) { return vec3(p1_global); }
    float t = clamp((ubo.isovalue - val1) / (val2 - val1), 0.0f, 1.0f);
    return mix(vec3(p1_global), vec3(p2_global), t);
}

vec3 calculateNormal(ivec3 globalCellCoord_not_used, vec3 vertexPos_global) {
    ivec3 Npos = ivec3(round(vertexPos_global));
    ivec3 volMax = ivec3(ubo.volumeDim.xyz) - 1;
    float s1_x = float(imageLoad(volumeImage, clamp(Npos + ivec3(1,0,0), ivec3(0), volMax)).r);
    float s2_x = float(imageLoad(volumeImage, clamp(Npos - ivec3(1,0,0), ivec3(0), volMax)).r);
    float s1_y = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,1,0), ivec3(0), volMax)).r);
    float s2_y = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,1,0), ivec3(0), volMax)).r);
    float s1_z = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,0,1), ivec3(0), volMax)).r);
    float s2_z = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,0,1), ivec3(0), volMax)).r);
    vec3 grad = vec3(s2_x - s1_x, s2_y - s1_y, s2_z - s1_z);
    if (length(grad) < 1e-5f) return vec3(0, 1, 0);
    return -normalize(grad);
}

void main() {
    if (gl_LocalInvocationIndex == 0) {
        sharedMeshletVertexCount_actual = 0;
        sharedMeshletPrimitiveCount_actual = 0;
    }
    // Initialize distinctEdgeOwner_VertexIdx. Size based on EXT_DIMs.
    uint totalExtendedCellsForCache = EXT_DIM_X * EXT_DIM_Y * EXT_DIM_Z;
    for (uint i = gl_LocalInvocationIndex; i < totalExtendedCellsForCache * 3; i += WORKGROUP_SIZE) {
        distinctEdgeOwner_VertexIdx[i] = 0xFFFFFFFFu; // Invalid index
    }
    barrier();

    uvec3 coreSubBlockDim = taskPayloadIn.subBlockDim;
    uvec3 subBlockBaseOrigin_global = taskPayloadIn.blockOrigin;

    // The mesh shader will generate vertices for an extended region: coreSubBlockDim + 1 on positive axes
    // This extendedDim is used for iterating cells whose distinct edges might be needed.
    uvec3 extendedProcessingDim = coreSubBlockDim + uvec3(1,1,1);
    // Clamp extendedProcessingDim to not exceed global volume limits relative to subBlockBaseOrigin_global
    // This is complex if subBlockBaseOrigin_global + extendedProcessingDim > ubo.volumeDim.xyz
    // For simplicity, assume task shader sends subBlocks that allow +1 context read safely.
    // A more robust way is to clamp reads inside loops.
    // PMB loads Bx+1, By+1, Bz+1 voxels for a BxByBz cell block. This means it reads up to
    // the corners of the cells at the (B-1) indices.

    uint cellsInExtendedBlock = extendedProcessingDim.x * extendedProcessingDim.y * extendedProcessingDim.z;


    // --- PHASE 1: Generate unique vertices for ALL distinct edges in the EXTENDED region ---
    // Each thread processes cells in the *extended* region to populate distinctEdgeOwner_VertexIdx
    for (uint linearCellIdx_ext = gl_LocalInvocationIndex; linearCellIdx_ext < cellsInExtendedBlock; linearCellIdx_ext += WORKGROUP_SIZE) {
        if (sharedMeshletVertexCount_actual >= MAX_VERTICES) break; // Early exit if vertex budget is blown

        uvec3 localCellInExtBlock; // Coords relative to the extended block, (0,0,0) to (extendedProcessingDim-1)
        localCellInExtBlock.x = linearCellIdx_ext % extendedProcessingDim.x;
        localCellInExtBlock.y = (linearCellIdx_ext / extendedProcessingDim.x) % extendedProcessingDim.y;
        localCellInExtBlock.z = linearCellIdx_ext / (extendedProcessingDim.x * extendedProcessingDim.y);

        ivec3 globalCellOrigin_for_val_sampling = ivec3(subBlockBaseOrigin_global) + ivec3(localCellInExtBlock);

        float cornerValuesF[8];
        uint cubeCase = 0;
        // Read corner values (8 voxels) for this cell
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellOrigin_for_val_sampling + cornerOffset;
            uint valU = 0;
            if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                valU = imageLoad(volumeImage, cornerVolCoord).r;
            }
            cornerValuesF[i] = float(valU);
            if (cornerValuesF[i] >= ubo.isovalue) cubeCase |= (1 << i);
        }

        if (cubeCase == 0 || cubeCase == 255) continue; // No geometry from this cell

        // This cell (localCellInExtBlock) generates its 3 distinct positive-going edge vertices
        int distinct_mc_edge_ids[3] = {0, 3, 8}; // MC edge IDs for X+, Y+, Z+ from cell (0,0,0) corner
        for (int axis_idx = 0; axis_idx < 3; ++axis_idx) { // 0:X, 1:Y, 2:Z
                                                           int mcEdgeID = distinct_mc_edge_ids[axis_idx];

                                                           bool edgeIsActive = false; // Check if this distinct edge is used by triangles of this cubeCase
                                                           int triTableBase = int(cubeCase * 16);
                                                           for (int k_tri_edge = 0; k_tri_edge < 15; ++k_tri_edge) {
                                                               if (mc.triTable[triTableBase + k_tri_edge] == -1) break;
                                                               if (mc.triTable[triTableBase + k_tri_edge] == mcEdgeID) {
                                                                   edgeIsActive = true;
                                                                   break;
                                                               }
                                                           }
                                                           if (!edgeIsActive) continue;

                                                           // This distinct edge needs a vertex. Add it to sharedMeshletVertices.
                                                           // Its local meshlet index will be stored in distinctEdgeOwner_VertexIdx.
                                                           if (sharedMeshletVertexCount_actual >= MAX_VERTICES) break; // Check before atomicAdd

                                                           uint new_vertex_local_idx = atomicAdd(sharedMeshletVertexCount_actual, 1u);
                                                           if (new_vertex_local_idx < MAX_VERTICES) {
                                                               ivec3 p1_offset_mc, p2_offset_mc; int c1_idx_mc, c2_idx_mc, interpolated_axis_mc;
                                                               getEdgeInfo(mcEdgeID, p1_offset_mc, p2_offset_mc, c1_idx_mc, c2_idx_mc, interpolated_axis_mc);

                                                               vec3 vertPosGlobal = interpolateVertex(globalCellOrigin_for_val_sampling + p1_offset_mc, globalCellOrigin_for_val_sampling + p2_offset_mc, cornerValuesF[c1_idx_mc], cornerValuesF[c2_idx_mc]);
                                                               vec3 vertNormGlobal = calculateNormal(globalCellOrigin_for_val_sampling, vertPosGlobal);

                                                               sharedMeshletVertices[new_vertex_local_idx].position = vec4(vertPosGlobal, 1.0);
                                                               sharedMeshletVertices[new_vertex_local_idx].normal = vec4(vertNormGlobal, 0.0);

                                                               // Store this new_vertex_local_idx in the shared cache for this distinct edge
                                                               uint distinct_edge_cache_flat_idx =
                                                               (localCellInExtBlock.z * EXT_DIM_Y * EXT_DIM_X +
                                                               localCellInExtBlock.y * EXT_DIM_X +
                                                               localCellInExtBlock.x) * 3 + axis_idx;
                                                               distinctEdgeOwner_VertexIdx[distinct_edge_cache_flat_idx] = new_vertex_local_idx;
                                                               // debugPrintfEXT("MS P1: Task %u, CellExt (%u,%u,%u) Axis %u -> VtxIdx %u\n", taskPayloadIn.taskId, localCellInExtBlock.x,localCellInExtBlock.y,localCellInExtBlock.z, axis_idx, new_vertex_local_idx);
                                                           }
                                                           // else: MAX_VERTICES hit, this vertex (and subsequent ones) won't be added.
                                                           // No need to decrement sharedMeshletVertexCount_actual.
        }
        if (sharedMeshletVertexCount_actual >= MAX_VERTICES) break;
    }
    barrier(); // Ensure all distinct vertices for the extended block are generated and indices stored.

    // --- PHASE 2: Assemble triangles for CORE cells ONLY ---
    // Iterate only over cells in the CORE sub-block.
    uint cellsInCoreSubBlock = coreSubBlockDim.x * coreSubBlockDim.y * coreSubBlockDim.z;
    bool threadShouldStopTriAssembly = false;

    for (uint linearCellIdx_core = gl_LocalInvocationIndex; linearCellIdx_core < cellsInCoreSubBlock; linearCellIdx_core += WORKGROUP_SIZE) {
        if (threadShouldStopTriAssembly) break;
        if (sharedMeshletPrimitiveCount_actual >= MAX_PRIMITIVES) {
            threadShouldStopTriAssembly = true;
            break;
        }

        uvec3 localCellCoord_core; // Coords relative to coreSubBlockDim, these are also coords into the extended block
        localCellCoord_core.x = linearCellIdx_core % coreSubBlockDim.x;
        localCellCoord_core.y = (linearCellIdx_core / coreSubBlockDim.x) % coreSubBlockDim.y;
        localCellCoord_core.z = linearCellIdx_core / (coreSubBlockDim.x * coreSubBlockDim.y);

        ivec3 globalCellOrigin_for_val_sampling = ivec3(subBlockBaseOrigin_global) + ivec3(localCellCoord_core);

        // Re-calculate cubeCase and cornerValues for the current CORE cell
        float cornerValuesF_assembly[8];
        uint cubeCase_assembly = 0;
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellOrigin_for_val_sampling + cornerOffset;
            uint valU = 0;
            if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                valU = imageLoad(volumeImage, cornerVolCoord).r;
            }
            cornerValuesF_assembly[i] = float(valU);
            if (cornerValuesF_assembly[i] >= ubo.isovalue) cubeCase_assembly |= (1 << i);
        }

        if (cubeCase_assembly == 0 || cubeCase_assembly == 255) continue;

        int baseTriTableIdx = int(cubeCase_assembly * 16);
        for (int tri_idx_in_cell = 0; tri_idx_in_cell < 5; ++tri_idx_in_cell) {
            if (sharedMeshletPrimitiveCount_actual >= MAX_PRIMITIVES) {
                threadShouldStopTriAssembly = true;
                break;
            }

            int mcEdge0 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 0];
            if (mcEdge0 == -1) break;

            int mcEdge1 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 1];
            int mcEdge2 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 2];
            int currentTriangleMCEdges[3] = {mcEdge0, mcEdge1, mcEdge2};

            uvec3 currentTriangleLocalIndices;
            bool triangleIsValid = true;

            for (int v_idx = 0; v_idx < 3; ++v_idx) {
                int mcEdgeID_for_tri_vtx = currentTriangleMCEdges[v_idx];
                EdgeMap mapInfo = neighborMappingTable[mcEdgeID_for_tri_vtx];

                // ownerCellLocalCoord_in_ExtBlock is relative to the start of the extended block (0,0,0)
                // localCellCoord_core is also relative to the start of the extended block because
                // task_payload.blockOrigin is the origin of the core block.
                ivec3 ownerCellLocalCoord_in_ExtBlock = ivec3(localCellCoord_core) + mapInfo.ownerCellOffset_RelativeToCurrentMCcell;

                // The owner cell's coordinates must be within the bounds of cells for which Phase 1 generated vertices.
                // This means ownerCellLocalCoord_in_ExtBlock must be [0, extendedProcessingDim-1]
                if (any(lessThan(ownerCellLocalCoord_in_ExtBlock, ivec3(0))) ||
                any(greaterThanEqual(ownerCellLocalCoord_in_ExtBlock, ivec3(extendedProcessingDim)))) {
                    debugPrintfEXT("MS P2 ERR: Task %u, CoreCell (%u,%u,%u) MCedge %d -> OwnerExtCell (%d,%d,%d) OUTSIDE ExtDim (%u,%u,%u). Skipping tri.\n",
                                   taskPayloadIn.taskId, localCellCoord_core.x, localCellCoord_core.y, localCellCoord_core.z, mcEdgeID_for_tri_vtx,
                                   ownerCellLocalCoord_in_ExtBlock.x, ownerCellLocalCoord_in_ExtBlock.y, ownerCellLocalCoord_in_ExtBlock.z,
                                   extendedProcessingDim.x, extendedProcessingDim.y, extendedProcessingDim.z);
                    triangleIsValid = false;
                    break;
                }

                uint distinct_edge_cache_flat_idx =
                (ownerCellLocalCoord_in_ExtBlock.z * EXT_DIM_Y * EXT_DIM_X + // Use EXT_DIM for strides
                ownerCellLocalCoord_in_ExtBlock.y * EXT_DIM_X +
                ownerCellLocalCoord_in_ExtBlock.x) * 3 + mapInfo.distinctEdgeAxis_OfOwner;

                uint vertexLocalIdx = distinctEdgeOwner_VertexIdx[distinct_edge_cache_flat_idx];

                if (vertexLocalIdx == 0xFFFFFFFFu || vertexLocalIdx >= MAX_VERTICES) { // Vertex wasn't generated or index out of bounds
                                                                                       debugPrintfEXT("MS P2 ERR: Task %u, CoreCell (%u,%u,%u) MCedge %d -> OwnerExtCell (%d,%d,%d) Axis %d -> VtxIdx %u (MAX %u) INVALID. CacheIdx %u. Skipping tri.\n",
                                                                                                      taskPayloadIn.taskId, localCellCoord_core.x, localCellCoord_core.y, localCellCoord_core.z, mcEdgeID_for_tri_vtx,
                                                                                                      ownerCellLocalCoord_in_ExtBlock.x, ownerCellLocalCoord_in_ExtBlock.y, ownerCellLocalCoord_in_ExtBlock.z, mapInfo.distinctEdgeAxis_OfOwner,
                                                                                                      vertexLocalIdx, MAX_VERTICES, distinct_edge_cache_flat_idx);
                                                                                       triangleIsValid = false;
                                                                                       break;
                }
                currentTriangleLocalIndices[v_idx] = vertexLocalIdx;
            } // End v_idx loop

            if (triangleIsValid) {
                if (currentTriangleLocalIndices.x == currentTriangleLocalIndices.y ||
                currentTriangleLocalIndices.x == currentTriangleLocalIndices.z ||
                currentTriangleLocalIndices.y == currentTriangleLocalIndices.z) {
                    // debugPrintfEXT("MS P2 Degen: Task %u, CoreCell (%u,%u,%u)\n", taskPayloadIn.taskId, localCellCoord_core.x, localCellCoord_core.y, localCellCoord_core.z);
                    continue;
                }
                uint primIdx = atomicAdd(sharedMeshletPrimitiveCount_actual, 1u);
                if (primIdx < MAX_PRIMITIVES) {
                    sharedMeshletIndices[primIdx] = currentTriangleLocalIndices;
                }
                // else: primitive limit hit, this primIdx is invalid for write. No decrement.
                // Outer loop checks will cause thread to stop.
            }
            if (sharedMeshletPrimitiveCount_actual >= MAX_PRIMITIVES) {threadShouldStopTriAssembly = true; break;}
        } // End triangle loop for cell
        if (threadShouldStopTriAssembly) break;
    } // End core cell loop for this thread

    // --- Final Output Stage ---
    barrier();
    uint finalVertexCount = min(sharedMeshletVertexCount_actual, MAX_VERTICES);
    uint finalPrimitiveCount = min(sharedMeshletPrimitiveCount_actual, MAX_PRIMITIVES);

    SetMeshOutputsEXT(finalVertexCount, finalPrimitiveCount);

    // ... (Output to gl_MeshVerticesEXT, vertexDataOut, SSBOs, and meshletDescOutput using filledMeshletDescCount - same as before) ...
    // ... Remember to use actual buffer capacities for SSBO bounds checks ...
    uint globalVtxBase = taskPayloadIn.globalVertexOffset;
    for (uint i = gl_LocalInvocationIndex; i < finalVertexCount; i += WORKGROUP_SIZE) {
        VertexData vData = sharedMeshletVertices[i];
        gl_MeshVerticesEXT[i].gl_Position = vData.position;
        meshVertexDataOut[i].normal = vData.normal.xyz;
        uint effectiveVertexCapacity = 200000000; // Placeholder
        if (globalVtxBase + i < effectiveVertexCapacity) {
            vertices_ssbo.vertex_data[globalVtxBase + i] = vData;
        } else {
            debugPrintfEXT("MS VTX WRITE OOB: Task %u, globalVtxBase %u, i %u. Total %u Cap %u. Skipping write.\n", taskPayloadIn.taskId, globalVtxBase, i, globalVtxBase + i, effectiveVertexCapacity);
        }
    }

    uint globalIdxBase = taskPayloadIn.globalIndexOffset;
    for (uint i = gl_LocalInvocationIndex; i < finalPrimitiveCount; i += WORKGROUP_SIZE) {
        uvec3 localIndicesForPrim = sharedMeshletIndices[i];
        if (localIndicesForPrim.x >= finalVertexCount ||
        localIndicesForPrim.y >= finalVertexCount ||
        localIndicesForPrim.z >= finalVertexCount) {
            debugPrintfEXT("MS IDX LOGIC ERR: Task %u, local indices (%u,%u,%u) out of finalVertexCount %u. Skipping prim.\n", taskPayloadIn.taskId, localIndicesForPrim.x, localIndicesForPrim.y, localIndicesForPrim.z, finalVertexCount);
            continue;
        }

        gl_PrimitiveTriangleIndicesEXT[i] = localIndicesForPrim;

        uint effectiveIndexCapacity = 600000000; // Placeholder
        if (globalIdxBase + i * 3 + 2 < effectiveIndexCapacity) {
            indices_ssbo.indices[globalIdxBase + i * 3 + 0] = localIndicesForPrim.x + globalVtxBase;
            indices_ssbo.indices[globalIdxBase + i * 3 + 1] = localIndicesForPrim.y + globalVtxBase;
            indices_ssbo.indices[globalIdxBase + i * 3 + 2] = localIndicesForPrim.z + globalVtxBase;
        } else {
            debugPrintfEXT("MS IDX WRITE OOB: Task %u, globalIdxBase %u, i*3+2 %u. Total %u Cap %u Skipping write.\n", taskPayloadIn.taskId, globalIdxBase, i*3+2, globalIdxBase + i * 3 + 2, effectiveIndexCapacity);
        }
    }

    barrier();
    if (gl_LocalInvocationIndex == 0) {
        if (finalVertexCount > 0 && finalPrimitiveCount > 0) {
            uint actualDescWriteIdx = atomicAdd(filledMeshletDescCount.filledMeshletCounter, 1u);
            uint effectiveDescCapacity = 2000000; // Placeholder
            if (actualDescWriteIdx < effectiveDescCapacity) {
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].vertexOffset = globalVtxBase;
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].indexOffset = globalIdxBase;
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].vertexCount = finalVertexCount;
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].primitiveCount = finalPrimitiveCount;
            } else {
                debugPrintfEXT("MS DESC WRITE OOB (FILLED): Task %u, actualDescWriteIdx %u. Verts %u Prims %u Skipping.\n", taskPayloadIn.taskId, actualDescWriteIdx, finalVertexCount, finalPrimitiveCount);
            }
        } else if (taskPayloadIn.subBlockDim.x > 0) {
            debugPrintfEXT("MS EMPTY MESHLET FINAL: Task %u, SBDim %u %u %u, SBOrigin %u %u %u. SharedVC %u SharedPC %u. FinalV %u FinalP %u\n",
                           taskPayloadIn.taskId, taskPayloadIn.subBlockDim.x, taskPayloadIn.subBlockDim.y, taskPayloadIn.subBlockDim.z,
                           taskPayloadIn.blockOrigin.x, taskPayloadIn.blockOrigin.y, taskPayloadIn.blockOrigin.z,
                           sharedMeshletVertexCount_actual, sharedMeshletPrimitiveCount_actual,
                           finalVertexCount, finalPrimitiveCount);
        }
    }
}