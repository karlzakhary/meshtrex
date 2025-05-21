#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#define MAX_VERTICES_LOCAL_OUTPUT 64u // For SetMeshOutputsEXT (can be small if not rasterizing)
#define MAX_PRIMITIVES_LOCAL_OUTPUT 126u // For SetMeshOutputsEXT

#define WORKGROUP_SIZE 32
#define CONTEXT_CELLS_PER_SIDE 1 // Should match task shader
#define MAX_CACHE_PROBES 8

// Max dimensions of the PROCESSING block (core + context)
// If core is 8x8x8 and context is 1, then processing is 10x10x10
// This is used for sizing the shared cache for edge IDs.
#define MAX_PROC_BLOCK_X (8 + 2*CONTEXT_CELLS_PER_SIDE)
#define MAX_PROC_BLOCK_Y (8 + 2*CONTEXT_CELLS_PER_SIDE)
#define MAX_PROC_BLOCK_Z (8 + 2*CONTEXT_CELLS_PER_SIDE)


struct MeshletDescriptor {
    uint vertexOffset; // Will be 0 if using global IDs directly for final mesh
    uint indexOffset;
    uint vertexCount;  // Count of vertices *output by this meshlet to SetMeshOutputsEXT*
    uint primitiveCount;
};

struct VertexData {
    vec4 position;
    vec4 normal;
};

struct TaskPayload {
    uint globalVertexOffset_reservation; // For this meshlet's local output to SetMeshOutputsEXT
    uint globalIndexOffset_reservation;  // For this meshlet's local output to SetMeshOutputsEXT

    uvec3 coreBlockOrigin_global;   // Global origin of the CORE cells this meshlet is responsible for
    uvec3 coreBlockDim;             // Dimensions of the CORE cells
    uvec3 processingBlockOrigin_global; // Global origin of the larger block to process (core + context)
    uvec3 processingBlockDim_total; // Dimensions of this larger block (core + context)

    uint taskId;                    // Original compactedBlockID
};

taskPayloadSharedEXT TaskPayload taskPayloadIn;

// Bindings
layout(set = 0, binding = 0, std140) uniform PushConstants { /* ... same ... */
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;

// GLOBAL Output Buffers - vertices written using global IDs
layout(set = 0, binding = 6, std430) buffer GlobalVertexBuffer_SSBO { VertexData vertex_data[]; } globalVerticesSSBO;
layout(set = 0, binding = 8, std430) buffer GlobalIndexBuffer_SSBO { uint indices[]; } globalIndicesSSBO;

// Output for non-empty meshlet descriptors (written by this shader using its own counter)
layout(set = 0, binding = 10, std430) buffer MeshletDescOutput_SSBO { MeshletDescriptor meshletDescriptors[]; } meshletDescOutput;
// Counter for Global Vertices (used to assign unique IDs)
layout(set = 0, binding = 11, std430) buffer GlobalVertexIDCounter_SSBO { uint counter; } globalVertexIDCounter;
// Counter for ACTUALLY WRITTEN (filled) meshlet descriptors
layout(set = 0, binding = 12, std430) buffer FilledMeshletDescCount_SSBO { uint filledMeshletCounter; } filledMeshletDescCount;
// Counter for Global Indices (written by this shader)
layout(set = 0, binding = 13, std430) buffer GlobalIndexOutputCount_SSBO { uint counter; } globalIndexOutputCount;


layout(local_size_x = WORKGROUP_SIZE) in;
// These are for the direct rasterizer path, less critical if we write to SSBOs for persistent mesh
layout(max_vertices = MAX_VERTICES_LOCAL_OUTPUT, max_primitives = MAX_PRIMITIVES_LOCAL_OUTPUT) out;
layout(triangles) out;

layout(location = 0) out PerVertexData { vec3 normal; } meshVertexDataOut_local[]; // For local rasterizer path

// Shared Memory
// For the *local output* via SetMeshOutputsEXT (can be minimal if not used)
shared VertexData sharedLocalVertices[MAX_VERTICES_LOCAL_OUTPUT];
shared uvec3 sharedLocalIndices[MAX_PRIMITIVES_LOCAL_OUTPUT];
shared uint sharedLocalVertexCount;
shared uint sharedLocalPrimitiveCount;
shared uint tempGlobalTriangleIndices[MAX_PRIMITIVES_LOCAL_OUTPUT * 3]; // Max output for rasterizer
shared uint tempLocalPrimitiveCount;

// Cache to map a global edge ID (from any cell in processingBlockDim_total)
// to its already assigned global unique vertex ID.
// Key: hash of (global_cell_coord_of_edge_origin, edge_axis_in_cell (0-2 for distinct))
// Value: The global unique vertex ID.
#define EDGE_ID_CACHE_SLOTS 1024 // Power of 2, adjust based on typical # of unique edges in processingBlockDim_total
#define EDGE_CACHE_EMPTY_SLOT 0xFFFFFFFFu
#define EDGE_POS_EPSILON 1e-5f

// Key for this cache will be a globally unique identifier for an edge on the grid
// e.g., hash(global_voxel_coord_of_edge_start, edge_axis (0..11 relative to that voxel as cell origin))
shared uvec4 edgeCache_Keys[EDGE_ID_CACHE_SLOTS]; // Key: x,y,z of edge start voxel, w=MC edge ID (0-11)
shared uint  edgeCache_GlobalVertexIDs[EDGE_ID_CACHE_SLOTS];  // Value: The Global Vertex ID

// Helper functions (getEdgeInfo, interpolateVertex, calculateNormal)
// ... (These functions remain the same as in the previous correct version) ...
void getEdgeInfo(int edgeIndex, out ivec3 p1_offset, out ivec3 p2_offset, out int corner1_idx, out int corner2_idx, out int axis) {
    switch (edgeIndex) {
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

// Generates a key for an edge based on its globally lowest indexed voxel and its axis relative to that voxel
uvec2 getGlobalEdgeKey(ivec3 globalCellOrigin_of_MC_cell, int mcEdgeID_0_11) {
    ivec3 p1_offset, p2_offset;
    int c1_idx, c2_idx, mc_axis;
    getEdgeInfo(mcEdgeID_0_11, p1_offset, p2_offset, c1_idx, c2_idx, mc_axis);

    // Determine the globally "lower" voxel of the edge and the edge's primary axis
    ivec3 v1_global = globalCellOrigin_of_MC_cell + p1_offset;
    ivec3 v2_global = globalCellOrigin_of_MC_cell + p2_offset;

    ivec3 edgeOriginVoxel_global = min(v1_global, v2_global);
    int edgeAxisInVoxel; // 0 for X, 1 for Y, 2 for Z edge from edgeOriginVoxel_global
    if (v1_global.x != v2_global.x) edgeAxisInVoxel = 0;
    else if (v1_global.y != v2_global.y) edgeAxisInVoxel = 1;
    else edgeAxisInVoxel = 2;

    // Hash the 3D voxel coordinate and axis
    uint hash_coord = uint(edgeOriginVoxel_global.x * 73856093 ^
    edgeOriginVoxel_global.y * 19349663 ^
    edgeOriginVoxel_global.z * 83492791);
    uint final_hash = (hash_coord ^ edgeAxisInVoxel) & (EDGE_ID_CACHE_SLOTS - 1);

    // Return a more complete key for precise matching in cache, not just the hash slot
    // For precise matching, store and compare edgeOriginVoxel_global and edgeAxisInVoxel
    // Pack into uvec2: key.x = hash_coord, key.y = edgeAxisInVoxel for precise comparison
    return uvec2(hash_coord, uint(edgeAxisInVoxel));
}


void main() {

    if (gl_LocalInvocationIndex == 0) {
        sharedLocalVertexCount = 0; // For SetMeshOutputsEXT path
        sharedLocalPrimitiveCount = 0; // For SetMeshOutputsEXT path
        for (int i = 0; i < EDGE_ID_CACHE_SLOTS; ++i) {
            edgeCache_GlobalVertexIDs[i] = EDGE_CACHE_EMPTY_SLOT;
            edgeCache_Keys[i] = uvec4(0xFFFFFFFFu); // Mark key as invalid
        }
    }
    barrier();

    uvec3 coreBlockOrigin         = taskPayloadIn.coreBlockOrigin_global;
    uvec3 coreDim                 = taskPayloadIn.coreBlockDim;
    uvec3 processingBlockOrigin   = taskPayloadIn.processingBlockOrigin_global;
    uvec3 processingDim           = taskPayloadIn.processingBlockDim_total;
    uint taskId                   = taskPayloadIn.taskId;

    uint cellsInProcessingBlock = processingDim.x * processingDim.y * processingDim.z;

    // Temporary storage for this meshlet's generated global vertex IDs for its triangles
    // Max 5 triangles * 3 vertices = 15 global IDs per cell.
    // Processed by all cells in coreDim.
    uint max_tris_from_core = coreDim.x * coreDim.y * coreDim.z * 5;
    if (max_tris_from_core == 0) max_tris_from_core = 1; // Avoid zero size array if coreDim is 0,0,0

    if(gl_LocalInvocationIndex == 0) tempLocalPrimitiveCount = 0;
    barrier();


    // Each thread iterates ALL cells in the processingBlockDim (core + context)
    // to generate vertices and populate the edge cache.
    for (uint linearCellIdx_proc = gl_LocalInvocationIndex;
    linearCellIdx_proc < cellsInProcessingBlock;
    linearCellIdx_proc += WORKGROUP_SIZE) {

        uvec3 localCellInProcBlock; // Coords relative to processingBlockOrigin, 0 to processingDim-1
        localCellInProcBlock.x = linearCellIdx_proc % processingDim.x;
        localCellInProcBlock.y = (linearCellIdx_proc / processingDim.x) % processingDim.y;
        localCellInProcBlock.z = linearCellIdx_proc / (processingDim.x * processingDim.y);

        ivec3 globalCellOrigin = ivec3(processingBlockOrigin) + ivec3(localCellInProcBlock);

        float cornerValuesF[8];
        uint cubeCase = 0;
        // ... (Calculate cubeCase and cornerValuesF for globalCellOrigin - same as before) ...
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellOrigin + cornerOffset;
            uint valU = 0;
            if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                valU = imageLoad(volumeImage, cornerVolCoord).r;
            }
            cornerValuesF[i] = float(valU);
            if (cornerValuesF[i] >= ubo.isovalue) cubeCase |= (1 << i);
        }
        if (cubeCase == 0 || cubeCase == 255) continue;


        // For each of the up to 15 edges that can form vertices for this cell's triangles
        int baseTriTableIdx = int(cubeCase * 16);
        for (int k_edge = 0; k_edge < 15; ++k_edge) {
            int mcEdgeID = mc.triTable[baseTriTableIdx + k_edge];
            if (mcEdgeID == -1) break;
            // Check if this vertex is already processed (for this meshlet)
            // Key for cache: global origin of the cell this mcEdgeID belongs to, and the mcEdgeID itself,
            // or more canonically, the global coords of the "lower" voxel of the edge and its axis.
            uvec2 globalEdgeKeyData = getGlobalEdgeKey(globalCellOrigin, mcEdgeID);
            uint slot = globalEdgeKeyData.x & (EDGE_ID_CACHE_SLOTS - 1); // Use only hash part for slot index

            bool foundInCache = false;
            for (int probe = 0; probe < MAX_CACHE_PROBES; ++probe) {
                uint currentSlot = (slot + probe) & (EDGE_ID_CACHE_SLOTS - 1);
                if (edgeCache_GlobalVertexIDs[currentSlot] != EDGE_CACHE_EMPTY_SLOT) {
                    if (edgeCache_Keys[currentSlot].x == globalEdgeKeyData.x &&
                    edgeCache_Keys[currentSlot].y == globalEdgeKeyData.y) { // Precise key match
                                                                            foundInCache = true;
                                                                            break;
                    }
                } else { // Empty slot found during probe
                         // Try to claim it for this globalEdgeKeyData
                         uint newGlobalVertexID = atomicAdd(globalVertexIDCounter.counter, 1u);

                         // This write to vertex_data needs to be globally unique for this newGlobalVertexID
                         // If multiple meshlets generate same newGlobalVertexID for *different* edges due to counter races,
                         // this is an issue. The globalVertexIDCounter *should* provide unique IDs.
                         // The critical part is that only ONE thread across ALL meshlets writes to vertex_data[newGlobalVertexID].
                         // This is often solved by a "first one wins" flag per global ID, or by design.
                         // For now, assume current thread is responsible if it's the one allocating.

                         ivec3 p1_offset, p2_offset; int c1_idx, c2_idx, mc_axis;
                         getEdgeInfo(mcEdgeID, p1_offset, p2_offset, c1_idx, c2_idx, mc_axis);
                         vec3 vertPos = interpolateVertex(globalCellOrigin + p1_offset, globalCellOrigin + p2_offset, cornerValuesF[c1_idx], cornerValuesF[c2_idx]);
                         vec3 vertNorm = calculateNormal(globalCellOrigin, vertPos);

                         // This write needs to be 100% guaranteed unique by newGlobalVertexID.
                         // Placeholder for bounds check:
                         if (newGlobalVertexID < 1000000000) { // Max capacity of globalVerticesSSBO
                                                               globalVerticesSSBO.vertex_data[newGlobalVertexID].position = vec4(vertPos, 1.0);
                                                               globalVerticesSSBO.vertex_data[newGlobalVertexID].normal = vec4(vertNorm, 0.0);
                         }


                         // Now try to put newGlobalVertexID into cache
                         uint prev_cache_val = atomicCompSwap(edgeCache_GlobalVertexIDs[currentSlot], EDGE_CACHE_EMPTY_SLOT, newGlobalVertexID);
                         if (prev_cache_val == EDGE_CACHE_EMPTY_SLOT) { // Successfully cached
                                                                        edgeCache_Keys[currentSlot] = uvec4(globalEdgeKeyData.x, globalEdgeKeyData.y, 0, 0); // Store precise key
                         }
                         // else: another thread cached its vertex here. Our vertex is still in global SSBO.
                         foundInCache = true; // It's processed.
                         break;
                }
                if (foundInCache) break;
            }
            if (!foundInCache) { // All probes failed to find/cache
                                 // This means cache is full of OTHER global edges. Vertex still needs to be generated and put in SSBO.
                                 uint newGlobalVertexID_uncached = atomicAdd(globalVertexIDCounter.counter, 1u);
                                 ivec3 p1_offset, p2_offset; int c1_idx, c2_idx, mc_axis;
                                 getEdgeInfo(mcEdgeID, p1_offset, p2_offset, c1_idx, c2_idx, mc_axis);
                                 vec3 vertPos = interpolateVertex(globalCellOrigin + p1_offset, globalCellOrigin + p2_offset, cornerValuesF[c1_idx], cornerValuesF[c2_idx]);
                                 vec3 vertNorm = calculateNormal(globalCellOrigin, vertPos);
                                 if (newGlobalVertexID_uncached < 1000000000) {
                                     globalVerticesSSBO.vertex_data[newGlobalVertexID_uncached].position = vec4(vertPos, 1.0);
                                     globalVerticesSSBO.vertex_data[newGlobalVertexID_uncached].normal = vec4(vertNorm, 0.0);
                                 }
                                 // debugPrintfEXT("MS VtxGenUncached: Task %u, GlobalID %u for edge %d in cell (%d,%d,%d)\n", taskId, newGlobalVertexID_uncached, mcEdgeID, globalCellOrigin.x,globalCellOrigin.y,globalCellOrigin.z);
            }
        } // End edge loop for vertex generation
    } // End cell loop for vertex generation/caching phase
    barrier(); // Ensure all cache writes are visible

    // --- PHASE 2: Assemble Triangles for CORE cells ONLY ---
    // Each thread processes cells belonging to the CORE block
    uint cellsInCoreBlock = coreDim.x * coreDim.y * coreDim.z;
    bool threadShouldStopTriAssembly = false;

    for (uint linearCellIdx_core = gl_LocalInvocationIndex;
    linearCellIdx_core < cellsInCoreBlock;
    linearCellIdx_core += WORKGROUP_SIZE) {

        if (threadShouldStopTriAssembly) break;
        if (sharedLocalPrimitiveCount >= MAX_PRIMITIVES_LOCAL_OUTPUT) { // Using local output limits here
                                                                        threadShouldStopTriAssembly = true; break;
        }

        uvec3 localCellInCoreBlock; // Relative to coreBlockOrigin
        localCellInCoreBlock.x = linearCellIdx_core % coreDim.x;
        localCellInCoreBlock.y = (linearCellIdx_core / coreDim.x) % coreDim.y;
        localCellInCoreBlock.z = linearCellIdx_core / (coreDim.x * coreDim.y);

        ivec3 globalCellOrigin_core = ivec3(coreBlockOrigin) + ivec3(localCellInCoreBlock);

        // ... (Re-calculate cubeCase and cornerValuesF for this globalCellOrigin_core - same as before) ...
        float cornerValuesF_assembly[8];
        uint cubeCase_assembly = 0;
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellOrigin_core + cornerOffset;
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
            if (sharedLocalPrimitiveCount >= MAX_PRIMITIVES_LOCAL_OUTPUT) {
                threadShouldStopTriAssembly = true; break;
            }
            int mcEdge0 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 0];
            if (mcEdge0 == -1) break;
            // ... (get mcEdge1, mcEdge2) ...
            int mcEdge1 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 1];
            int mcEdge2 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 2];
            int currentTriangleMCEdges[3] = {mcEdge0, mcEdge1, mcEdge2};

            uvec3 currentTriangleGlobalVertexIDs;
            bool triangleIsValid = true;

            for (int v_idx = 0; v_idx < 3; ++v_idx) {
                int mcEdgeID_for_tri_vtx = currentTriangleMCEdges[v_idx];

                uvec2 globalEdgeKeyData = getGlobalEdgeKey(globalCellOrigin_core, mcEdgeID_for_tri_vtx);
                uint slot = globalEdgeKeyData.x & (EDGE_ID_CACHE_SLOTS - 1);
                uint foundGlobalVertexID = EDGE_CACHE_EMPTY_SLOT;

                for (int probe = 0; probe < MAX_CACHE_PROBES; ++probe) {
                    uint currentSlot = (slot + probe) & (EDGE_ID_CACHE_SLOTS - 1);
                    if (edgeCache_GlobalVertexIDs[currentSlot] != EDGE_CACHE_EMPTY_SLOT &&
                    edgeCache_Keys[currentSlot].x == globalEdgeKeyData.x &&
                    edgeCache_Keys[currentSlot].y == globalEdgeKeyData.y) {
                        foundGlobalVertexID = edgeCache_GlobalVertexIDs[currentSlot];
                        break;
                    }
                    if (edgeCache_GlobalVertexIDs[currentSlot] == EDGE_CACHE_EMPTY_SLOT && probe > 0) {
                        // If we hit an empty slot after the first probe, means it wasn't further along (if it was a collision)
                        break;
                    }
                }

                if (foundGlobalVertexID == EDGE_CACHE_EMPTY_SLOT) {
                    // This should ideally not happen if Phase 1 correctly populated cache for all needed edges.
                    // Could happen if MAX_PROBES too small or cache too small.
                    debugPrintfEXT("MS P2 VTX LOOKUP FAIL: Task %u, CoreCell (%u,%u,%u) MCedge %d. GlobalKey (%u,%u). No GlobalID in cache.\n",
                                   taskId, localCellInCoreBlock.x, localCellInCoreBlock.y, localCellInCoreBlock.z, mcEdgeID_for_tri_vtx,
                                   globalEdgeKeyData.x, globalEdgeKeyData.y);
                    triangleIsValid = false;
                    break;
                }
                currentTriangleGlobalVertexIDs[v_idx] = foundGlobalVertexID;
            } // End v_idx loop

            if (triangleIsValid) {
                if (currentTriangleGlobalVertexIDs.x == currentTriangleGlobalVertexIDs.y ||
                currentTriangleGlobalVertexIDs.x == currentTriangleGlobalVertexIDs.z ||
                currentTriangleGlobalVertexIDs.y == currentTriangleGlobalVertexIDs.z) {
                    continue;
                }
                // Add to global index buffer
                uint global_idx_buffer_offset = atomicAdd(globalIndexOutputCount.counter, 3u);
                // Placeholder for bounds check on globalIndexOutputCount.counter
                if (global_idx_buffer_offset + 2 < 1000000000) { // Max capacity of globalIndicesSSBO
                                                                 globalIndicesSSBO.indices[global_idx_buffer_offset + 0] = currentTriangleGlobalVertexIDs.x;
                                                                 globalIndicesSSBO.indices[global_idx_buffer_offset + 1] = currentTriangleGlobalVertexIDs.y;
                                                                 globalIndicesSSBO.indices[global_idx_buffer_offset + 2] = currentTriangleGlobalVertexIDs.z;

                                                                 // Also add to local indices for SetMeshOutputsEXT if needed for rasterizer path
                                                                 // This requires mapping global IDs back to a local list, or just outputting some dummy data
                                                                 // for SetMeshOutputsEXT. For persistent mesh, SSBO is key.
                                                                 // For now, let's also try to populate sharedLocalIndices for SetMeshOutputsEXT as a proxy.
                                                                 // This part is tricky as sharedLocalVertices isn't populated with global IDs.
                                                                 // To make SetMeshOutputsEXT work, we'd need to fill sharedLocalVertices and map global IDs.
                                                                 // Simplification: only write to SSBO for now. Increment local prim count for descriptor.
                                                                 atomicAdd(sharedLocalPrimitiveCount, 1u); // Just to make descriptor non-empty if needed

                } else {
                    // Ran out of global index buffer space
                    threadShouldStopTriAssembly = true;
                }
            }
            if (threadShouldStopTriAssembly) break;
        } // End triangle loop
        if (threadShouldStopTriAssembly) break;
    } // End core cell loop

    barrier();
    // Final counts for SetMeshOutputsEXT (can be minimal, e.g., 0,0 if not rasterizing)
    // For the descriptor, we need vertex/primitive counts *for this meshlet's contribution to global buffers*.
    // This is harder to get accurately without more counters.
    // Let's use sharedLocalPrimitiveCount for the descriptor. Vertex count is harder if using global IDs.
    // The actual "vertex count" for a meshlet using global IDs is effectively the number of *unique* global IDs it referenced.

    uint finalLocalPrimCount = min(sharedLocalPrimitiveCount, MAX_PRIMITIVES_LOCAL_OUTPUT);
    // For finalVertexCount for descriptor: we could count unique global IDs used by this meshlet's tris,
    // or just put a placeholder if the descriptor isn't strictly used for rendering.
    // For simplicity, if prims > 0, say verts > 0.
    uint finalLocalVertCount = (finalLocalPrimCount > 0) ? min(globalVertexIDCounter.counter, MAX_VERTICES_LOCAL_OUTPUT) : 0; // This is not right.
    // The descriptor should reflect what this meshlet "outputs" or "references".
    // Let's use a small placeholder for local output for now.
    if(finalLocalPrimCount > 0 && finalLocalVertCount == 0) finalLocalVertCount = 3; // Min 3 verts for a tri

    SetMeshOutputsEXT(finalLocalVertCount, finalLocalPrimCount); // For rasterizer path (if any)

    // Write meshlet descriptor
    if (gl_LocalInvocationIndex == 0) {
        // This descriptor should reflect the *actual contribution to the global buffers*
        // This needs careful thought. If indices are global, vertexOffset might be 0.
        // The count could be the total in globalVertexIDCounter.counter *after this meshlet*.
        // For now, let's assume the C++ side will rebuild from the global vertex/index buffers.
        // The descriptor could point to the segment of *indices* this meshlet wrote.
        // However, the task shader reserved globalVertexOffset_reservation and globalIndexOffset_reservation
        // which were for the old model of meshlet-local vertex/index lists.

        // New Descriptor Logic:
        // Get the number of primitives this specific meshlet instance added to the global index buffer.
        // This is tricky without another atomic counter per meshlet for its own outputted primitives.
        // Let's assume `tempLocalPrimitiveCount` (if made truly atomic per meshlet) could be used.
        // For now, if any prims were generated for local output, write a descriptor using global offsets for that.

        uint currentFilledPrims = sharedLocalPrimitiveCount; // Prims added to sharedLocalIndices for raster path

        if (currentFilledPrims > 0) {
            uint actualDescWriteIdx = atomicAdd(filledMeshletDescCount.filledMeshletCounter, 1u);
            uint effectiveDescCapacity = 2000000; // Placeholder
            if (actualDescWriteIdx < effectiveDescCapacity) {
                // The vertex/index offsets here would ideally point to the *global* buffers
                // and counts would be how many this meshlet *contributed* or *references*.
                // This is where the model of independent meshlets with global IDs gets complex for descriptors.
                // Simplest for OBJ export: descriptor just notes it was non-empty.
                // The actual vertex/index ranges for this meshlet's triangles are spread in global buffers.
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].vertexOffset = 0; // Using global IDs
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].indexOffset = 0; // Using global IDs
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].vertexCount = globalVertexIDCounter.counter; // Total global vertices so far
                meshletDescOutput.meshletDescriptors[actualDescWriteIdx].primitiveCount = currentFilledPrims; // Prims this meshlet tried to output locally
            }
        }
    }
}