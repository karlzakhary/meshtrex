#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
#define MAX_VERTICES 256u // Max unique vertices per meshlet
#define MAX_PRIMITIVES 256u // Max primitives per meshlet
#define WORKGROUP_SIZE 32

// IMPORTANT: These should be set to match the maximum dimensions of ubo.blockDim
// (or the largest possible subBlockDim the task shader can send)
// Example: if ubo.blockDim is 8x8x8, these should be 8.
// If ubo.blockDim can be up to 16x16x16, these should be 16.
// For now, using a common default. ADJUST TO YOUR ubo.blockDim!
#define MAX_UBO_BLOCK_DIM_X 8
#define MAX_UBO_BLOCK_DIM_Y 8
#define MAX_UBO_BLOCK_DIM_Z 8

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

struct VertexData {
    vec4 position; // Store as vec4 for easier SSBO layout; w can be 1.0
    vec4 normal;   // Store as vec4 for easier SSBO layout; w can be 0.0
};

struct TaskPayload {
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletDescOffset;
    uvec3 blockOrigin;      // Global origin of the CORE sub-block
    uvec3 subBlockDim;      // Dimensions of the CORE sub-block
    uint taskId;            // Original compactedBlockID (for debugging)
};

taskPayloadSharedEXT TaskPayload taskPayloadIn;

// --- Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;         // Max dimensions of a base block (e.g., 8,8,8)
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;

// Classic Marching Cubes triTable (defines triangles using edge indices)
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO {
    int triTable[]; // 256 cases * 16 ints (max 5 tris * 3 verts + terminator)
} mc;

// NEW: Marching Cubes edgeTable (defines which edges are intersected for a case)
layout(set = 0, binding = 14, std430) readonly buffer EdgeTable_SSBO {
    uint edgeTable[]; // 256 entries, each a 12-bit mask
} edgeTable_ssbo; // Make sure binding 5 is free and correctly set up

// Output SSBOs
layout(set = 0, binding = 6, std430) buffer Vertex_SSBO { VertexData vertex_data[]; } vertices_ssbo;
layout(set = 0, binding = 8, std430) buffer Index_SSBO { uint indices[]; } indices_ssbo;
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
shared uvec3 sharedMeshletIndices[MAX_PRIMITIVES]; // Stores local (to meshlet) vertex indices for triangles
shared uint sharedMeshletVertexCount_actual;
shared uint sharedMeshletPrimitiveCount_actual;

// For PMB-style distinct edge vertex deduplication
// Dimensions accommodate cells up to MAX_UBO_BLOCK_DIM + 1 for context
#define EXT_DIM_X (MAX_UBO_BLOCK_DIM_X + 1)
#define EXT_DIM_Y (MAX_UBO_BLOCK_DIM_Y + 1)
#define EXT_DIM_Z (MAX_UBO_BLOCK_DIM_Z + 1)
// Stores local meshlet vertex index (0..MAX_VERTICES-1) for a vertex generated
// by cell (x,y,z within extended block) on its 'axis'-th distinct edge.
shared uint distinctEdgeOwner_VertexIdx[EXT_DIM_Z * EXT_DIM_Y * EXT_DIM_X * 3];

// --- Distinct Edge Helper Structures & Data (PMB Style) ---
struct EdgeMap {
    ivec3 ownerCellOffset_RelativeToCurrentMCcell; // Offset from current MC cell to the cell that "owns" this edge's vertex
    int distinctEdgeAxis_OfOwner;                 // 0 for X+, 1 for Y+, 2 for Z+ axis of that owner cell
};

// Maps one of the 12 standard MC edges of a cell to the cell that "owns"
// the vertex on that edge (relative to current cell) and which of the owner's
// 3 distinct positive-going edges it is.
const EdgeMap neighborMappingTable[12] = {
    EdgeMap(ivec3(0,0,0), 0), // Edge 0 (P0-P1) owned by current cell, its X+ distinct edge
    EdgeMap(ivec3(1,0,0), 1), // Edge 1 (P1-P2) owned by cell (X+1,Y,Z), its Y+ distinct edge
    EdgeMap(ivec3(0,1,0), 0), // Edge 2 (P3-P2) owned by cell (X,Y+1,Z), its X+ distinct edge
    EdgeMap(ivec3(0,0,0), 1), // Edge 3 (P0-P3) owned by current cell, its Y+ distinct edge
    EdgeMap(ivec3(0,0,1), 0), // Edge 4 (P4-P5) owned by cell (X,Y,Z+1), its X+ distinct edge
    EdgeMap(ivec3(1,0,1), 1), // Edge 5 (P5-P6) owned by cell (X+1,Y,Z+1), its Y+ distinct edge
    EdgeMap(ivec3(0,1,1), 0), // Edge 6 (P7-P6) owned by cell (X,Y+1,Z+1), its X+ distinct edge
    EdgeMap(ivec3(0,0,1), 1), // Edge 7 (P4-P7) owned by cell (X,Y,Z+1), its Y+ distinct edge
    EdgeMap(ivec3(0,0,0), 2), // Edge 8 (P0-P4) owned by current cell, its Z+ distinct edge
    EdgeMap(ivec3(1,0,0), 2), // Edge 9 (P1-P5) owned by cell (X+1,Y,Z), its Z+ distinct edge
    EdgeMap(ivec3(1,1,0), 2), // Edge 10 (P2-P6) owned by cell (X+1,Y+1,Z), its Z+ distinct edge
    EdgeMap(ivec3(0,1,0), 2)  // Edge 11 (P3-P7) owned by cell (X,Y+1,Z), its Z+ distinct edge
};

// --- Helper Functions ---
void getEdgeInfo(int edgeIndex, out ivec3 p1_offset, out ivec3 p2_offset, out int corner1_idx, out int corner2_idx, out int interpolated_axis) {
    // Standard MC edge definitions relative to cell origin (P0)
    // p1_offset, p2_offset are local to the cell; corner1_idx, corner2_idx are indices into the cell's 8 corners
    switch (edgeIndex) {
        case 0:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(1,0,0); corner1_idx=0; corner2_idx=1; interpolated_axis=0; return;
        case 1:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,1,0); corner1_idx=1; corner2_idx=2; interpolated_axis=1; return;
        case 2:  p1_offset=ivec3(0,1,0); p2_offset=ivec3(1,1,0); corner1_idx=3; corner2_idx=2; interpolated_axis=0; return; // Note: P2 is (1,1,0), P3 is (0,1,0)
        case 3:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,1,0); corner1_idx=0; corner2_idx=3; interpolated_axis=1; return;
        case 4:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(1,0,1); corner1_idx=4; corner2_idx=5; interpolated_axis=0; return;
        case 5:  p1_offset=ivec3(1,0,1); p2_offset=ivec3(1,1,1); corner1_idx=5; corner2_idx=6; interpolated_axis=1; return;
        case 6:  p1_offset=ivec3(0,1,1); p2_offset=ivec3(1,1,1); corner1_idx=7; corner2_idx=6; interpolated_axis=0; return;
        case 7:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(0,1,1); corner1_idx=4; corner2_idx=7; interpolated_axis=1; return;
        case 8:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,0,1); corner1_idx=0; corner2_idx=4; interpolated_axis=2; return;
        case 9:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,0,1); corner1_idx=1; corner2_idx=5; interpolated_axis=2; return;
        case 10: p1_offset=ivec3(1,1,0); p2_offset=ivec3(1,1,1); corner1_idx=2; corner2_idx=6; interpolated_axis=2; return;
        case 11: p1_offset=ivec3(0,1,0); p2_offset=ivec3(0,1,1); corner1_idx=3; corner2_idx=7; interpolated_axis=2; return;
        default: p1_offset=ivec3(0); p2_offset=ivec3(0); corner1_idx=-1; corner2_idx=-1; interpolated_axis=-1; return;
    }
}

vec3 interpolateVertex(ivec3 p1_global_voxel_coord, ivec3 p2_global_voxel_coord, float val1, float val2) {
    if (abs(ubo.isovalue - val1) < 1e-6f) return vec3(p1_global_voxel_coord); // Vertex is at p1
    if (abs(ubo.isovalue - val2) < 1e-6f) return vec3(p2_global_voxel_coord); // Vertex is at p2
    if (abs(val1 - val2) < 1e-6f) return vec3(p1_global_voxel_coord);      // Avoid division by zero if vals are same but not isovalue

    float t = (ubo.isovalue - val1) / (val2 - val1);
    // t should ideally be between 0 and 1 if edge is intersected, but clamp for robustness
    // t = clamp(t, 0.0f, 1.0f); // Not strictly needed if edgeTable guarantees intersection
    return mix(vec3(p1_global_voxel_coord), vec3(p2_global_voxel_coord), t);
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


void main() {
    // Initialization
    if (gl_LocalInvocationIndex == 0) {
        sharedMeshletVertexCount_actual = 0;
        sharedMeshletPrimitiveCount_actual = 0;
    }
    // Initialize distinctEdgeOwner_VertexIdx. Size based on EXT_DIMs.
    uint totalExtendedCellsForCache = EXT_DIM_X * EXT_DIM_Y * EXT_DIM_Z;
    // Parallel initialization of the shared cache
    for (uint i = gl_LocalInvocationIndex; i < totalExtendedCellsForCache * 3; i += WORKGROUP_SIZE) {
        distinctEdgeOwner_VertexIdx[i] = 0xFFFFFFFFu; // Invalid index marker
    }
    barrier(); // Ensure initialization is complete

    // --- Payload and Dimensions ---
    uvec3 coreSubBlockDim = taskPayloadIn.subBlockDim;
    uvec3 subBlockBaseOrigin_global = taskPayloadIn.blockOrigin; // Global voxel coords of cell (0,0,0) of this core sub-block

    // Mesh shader generates vertices for an extended region: coreSubBlockDim + 1 on positive axes
    // This is to ensure all cells within coreSubBlockDim can find their "owner" vertices for all 12 edges.
    uvec3 extendedProcessingDim = coreSubBlockDim + uvec3(1,1,1);
    // Clamp extendedProcessingDim to not exceed actual physical limits of distinctEdgeOwner_VertexIdx
    // This check is crucial if taskPayloadIn.subBlockDim can exceed MAX_UBO_BLOCK_DIM_X/Y/Z
    if (extendedProcessingDim.x > EXT_DIM_X || extendedProcessingDim.y > EXT_DIM_Y || extendedProcessingDim.z > EXT_DIM_Z) {
        // This sub-block is too large for the pre-defined shared memory cache.
        // This should ideally be prevented by the task shader by further subdivision
        // or by ensuring MAX_UBO_BLOCK_DIM_X/Y/Z correctly reflect ubo.blockDim.
        debugPrintfEXT("MS FATAL: Task %u, SubBlockDim (%u,%u,%u) results in ExtendedDim (%u,%u,%u) too large for cache (%u,%u,%u)!\n",
            taskPayloadIn.taskId, coreSubBlockDim.x, coreSubBlockDim.y, coreSubBlockDim.z,
            extendedProcessingDim.x, extendedProcessingDim.y, extendedProcessingDim.z,
            EXT_DIM_X, EXT_DIM_Y, EXT_DIM_Z);
        // Cannot proceed safely.
        SetMeshOutputsEXT(0,0); // Output no geometry
        return;
    }
    uint cellsInExtendedBlock = extendedProcessingDim.x * extendedProcessingDim.y * extendedProcessingDim.z;


    // --- PHASE 1: Generate unique vertices for active DISTINCT edges in the EXTENDED region ---
    // Each thread in the workgroup processes cells in the *extended* region.
    // It generates vertices ONLY for its 3 positive-going distinct edges (X+, Y+, Z+).
    for (uint linearCellIdx_ext = gl_LocalInvocationIndex; linearCellIdx_ext < cellsInExtendedBlock; linearCellIdx_ext += WORKGROUP_SIZE) {
        if (sharedMeshletVertexCount_actual >= MAX_VERTICES) {
             // Optional Debug: Log if workgroup stops early due to vertex capacity
             if(gl_LocalInvocationIndex == 0) { // Log once per workgroup
                debugPrintfEXT("MS P1 INFO: Task %u reached MAX_VERTICES (%u) at linearCellIdx_ext %u/%u. Ending Phase 1 early.\n",
                               taskPayloadIn.taskId, MAX_VERTICES, linearCellIdx_ext, cellsInExtendedBlock);
             }
            break; // Stop processing more cells if vertex budget for this meshlet is full
        }

        uvec3 localCellInExtBlock; // Coordinates relative to the start of the extended block, (0,0,0) to (extendedProcessingDim-1)
        localCellInExtBlock.x = linearCellIdx_ext % extendedProcessingDim.x;
        localCellInExtBlock.y = (linearCellIdx_ext / extendedProcessingDim.x) % extendedProcessingDim.y;
        localCellInExtBlock.z = linearCellIdx_ext / (extendedProcessingDim.x * extendedProcessingDim.y);

        // Global voxel coordinates for the origin (P0) of this cell in the extended block
        ivec3 globalCellOrigin_for_val_sampling = ivec3(subBlockBaseOrigin_global) + ivec3(localCellInExtBlock);

        // Sample 8 corner values for this cell to determine its cubeCase
        float cornerValuesF[8];
        uint cubeCase = 0;
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellOrigin_for_val_sampling + cornerOffset;
            uint valU = 0;
            // Safe image load with boundary checks
            if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                valU = imageLoad(volumeImage, cornerVolCoord).r;
            }
            cornerValuesF[i] = float(valU);
            if (cornerValuesF[i] >= ubo.isovalue) { // Check against isovalue
                cubeCase |= (1 << i);
            }
        }

        if (cubeCase == 0 || cubeCase == 255) continue; // Cell is entirely inside or outside

        // Get the 12-bit edge intersection mask for this cell's cubeCase
        uint cellEdgeIntersectionMask = edgeTable_ssbo.edgeTable[cubeCase];

        // This cell (localCellInExtBlock) is responsible for generating vertices on its 3 distinct positive-going edges.
        // These are MC edges 0 (X+ from P0), 3 (Y+ from P0), and 8 (Z+ from P0).
        // Their corresponding bit positions in the edgeTable mask are also 0, 3, and 8.
        int distinct_mc_edge_ids[3] = {0, 3, 8};    // MC edge ID for the distinct edge
        // int distinct_edge_bit_pos[3] = {0, 3, 8}; // Bit position in cellEdgeIntersectionMask

        for (int axis_idx = 0; axis_idx < 3; ++axis_idx) { // 0 for X-axis, 1 for Y-axis, 2 for Z-axis distinct edge
            int mcEdgeID_of_distinct_edge = distinct_mc_edge_ids[axis_idx];
            // int bitPos_of_distinct_edge = distinct_edge_bit_pos[axis_idx]; // Same as mcEdgeID for 0,3,8

            // Check if THIS cell's specific distinct edge is intersected
            bool isDistinctEdgeActive = (cellEdgeIntersectionMask & (1u << mcEdgeID_of_distinct_edge)) != 0u;

            if (!isDistinctEdgeActive) {
                if(gl_SubgroupInvocationID == 0 && gl_LocalInvocationIndex < WORKGROUP_SIZE && (taskPayloadIn.taskId % 50 == 0)) { // Reduce log spam
                   debugPrintfEXT("MS P1 WARN_MAX_VERTS_HIT: Task %u, CellExt (%u,%u,%u) Axis %d (MC Edge %d) IS ACTIVE (mask %x), but MAX_VERTICES (%u) already met. Vertex not stored.\n",
                                  taskPayloadIn.taskId, localCellInExtBlock.x, localCellInExtBlock.y, localCellInExtBlock.z,
                                  axis_idx, mcEdgeID_of_distinct_edge, cellEdgeIntersectionMask, MAX_VERTICES);
                }
                // Optional Debug: Log if a distinct edge is skipped
                // debugPrintfEXT("MS P1 SKIP: Task %u, CellExt (%u,%u,%u) Axis %d (MC Edge %d) not active from edgeTable mask %x. Vtx not generated.\n",
                //                taskPayloadIn.taskId, localCellInExtBlock.x,localCellInExtBlock.y,localCellInExtBlock.z,
                //                axis_idx, mcEdgeID_of_distinct_edge, cellEdgeIntersectionMask);
                continue; // If not intersected, no vertex is generated for this distinct edge by this cell
            }

            // This distinct edge IS active and needs a vertex.
            // Check vertex capacity BEFORE atomicAdd to avoid over-incrementing if already full.
            if (sharedMeshletVertexCount_actual >= MAX_VERTICES) {
                 // debugPrintfEXT("MS P1 WARN: Task %u, CellExt (%u,%u,%u) Axis %d active but MAX_VERTICES (%u) hit before atomicAdd. Vertex not stored.\n",
                 //                taskPayloadIn.taskId, localCellInExtBlock.x, localCellInExtBlock.y, localCellInExtBlock.z,
                 //                axis_idx, MAX_VERTICES);
                break; // Break from axis_idx loop for this cell if full, try next cell
            }

            uint new_vertex_local_idx = atomicAdd(sharedMeshletVertexCount_actual, 1u);
            if (new_vertex_local_idx < MAX_VERTICES) {
                ivec3 p1_offset_mc, p2_offset_mc;
                int c1_idx_mc, c2_idx_mc, interpolated_axis_mc; // interpolated_axis_mc indicates the edge's primary axis
                getEdgeInfo(mcEdgeID_of_distinct_edge, p1_offset_mc, p2_offset_mc, c1_idx_mc, c2_idx_mc, interpolated_axis_mc);

                // Global voxel coordinates of the two endpoints of the distinct edge
                ivec3 p1_global_vox = globalCellOrigin_for_val_sampling + p1_offset_mc;
                ivec3 p2_global_vox = globalCellOrigin_for_val_sampling + p2_offset_mc;

                // Interpolate vertex position along this distinct edge
                vec3 vertPosGlobal = interpolateVertex(p1_global_vox, p2_global_vox, cornerValuesF[c1_idx_mc], cornerValuesF[c2_idx_mc]);
                vec3 vertNormGlobal = calculateNormal(vertPosGlobal); // Pass interpolated global position

                // Store vertex data in shared memory for this meshlet
                sharedMeshletVertices[new_vertex_local_idx].position = vec4(vertPosGlobal, 1.0);
                sharedMeshletVertices[new_vertex_local_idx].normal   = vec4(vertNormGlobal, 0.0);

                // Store this new_vertex_local_idx in the shared cache for this cell's distinct edge
                // The cache index is based on the cell's local coordinates within the extended block and the axis.
                uint distinct_edge_cache_flat_idx =
                    (localCellInExtBlock.z * EXT_DIM_Y * EXT_DIM_X +  // Stride by Z
                     localCellInExtBlock.y * EXT_DIM_X +              // Stride by Y
                     localCellInExtBlock.x) * 3 + axis_idx;           // Select X, Y, or Z axis cache slot

                distinctEdgeOwner_VertexIdx[distinct_edge_cache_flat_idx] = new_vertex_local_idx;

                // Optional Debug: Log successful vertex generation and caching
                // debugPrintfEXT("MS P1 OK: Task %u, CellExt (%u,%u,%u) Axis %d (MC Edge %d) -> VtxIdx %u. CacheIdx %u. Mask %x\n",
                //                taskPayloadIn.taskId, localCellInExtBlock.x,localCellInExtBlock.y,localCellInExtBlock.z,
                //                axis_idx, mcEdgeID_of_distinct_edge, new_vertex_local_idx, distinct_edge_cache_flat_idx, cellEdgeIntersectionMask);

            } else {
                if(gl_SubgroupInvocationID == 0 && gl_LocalInvocationIndex < WORKGROUP_SIZE && (taskPayloadIn.taskId % 50 == 0)) {
                    debugPrintfEXT("MS P1 WARN_MAX_VERTS_OVERFLOW: Task %u, CellExt (%u,%u,%u) Axis %d. new_idx %u >= MAX_VERTICES after atomicAdd. Vertex lost.\n",
                                   taskPayloadIn.taskId, localCellInExtBlock.x,localCellInExtBlock.y,localCellInExtBlock.z,
                                   axis_idx, new_vertex_local_idx);
                 }
                // Vertex limit was hit exactly by this atomicAdd or others.
                // Decrement if we overshot, though the check before add is better.
                // atomicAdd(sharedMeshletVertexCount_actual, -1u); // This can be problematic with concurrent atomics.
                // Better to rely on the check before atomicAdd and the min() in the final output stage.
                // If new_vertex_local_idx >= MAX_VERTICES, this vertex won't be used anyway.
            }
        } // End loop over 3 distinct axes for the cell
         if (sharedMeshletVertexCount_actual >= MAX_VERTICES) break; // Check again to break from outer cell loop
    } // End loop over cells in extended block (linearCellIdx_ext)
    barrier(); // Ensure all distinct vertices for the extended block are generated and indices stored.

    // --- PHASE 2: Assemble triangles for CORE cells ONLY ---
    // Iterate only over cells in the CORE sub-block (dimensions from taskPayloadIn.subBlockDim).
    // These cells will look up vertices from the distinctEdgeOwner_VertexIdx cache,
    // which was populated by cells in the EXTENDED region.
    uint cellsInCoreSubBlock = coreSubBlockDim.x * coreSubBlockDim.y * coreSubBlockDim.z;
    bool threadShouldStopTriAssembly = false; // Per-thread flag to stop its contribution if limits hit

    for (uint linearCellIdx_core = gl_LocalInvocationIndex; linearCellIdx_core < cellsInCoreSubBlock; linearCellIdx_core += WORKGROUP_SIZE) {
        if (threadShouldStopTriAssembly) break; // This thread already hit a limit

        // Check shared primitive count limit before processing each cell
        if (sharedMeshletPrimitiveCount_actual >= MAX_PRIMITIVES) {
            threadShouldStopTriAssembly = true;
            break; // Stop all threads in this workgroup if primitive budget full.
                   // Note: This break is for the 'linearCellIdx_core' loop.
                   // Other threads might still be adding to sharedMeshletPrimitiveCount_actual.
                   // A barrier might be needed if strict "stop all" is required, but atomic check is often sufficient.
        }

        uvec3 localCellCoord_core; // Coords relative to coreSubBlockDim (0,0,0) to (coreSubBlockDim-1)
        localCellCoord_core.x = linearCellIdx_core % coreSubBlockDim.x;
        localCellCoord_core.y = (linearCellIdx_core / coreSubBlockDim.x) % coreSubBlockDim.y;
        localCellCoord_core.z = linearCellIdx_core / (coreSubBlockDim.x * coreSubBlockDim.y);

        // Global voxel coordinates for the origin (P0) of this CORE cell
        ivec3 globalCellOrigin_for_val_sampling_assembly = ivec3(subBlockBaseOrigin_global) + ivec3(localCellCoord_core);

        // Re-calculate cubeCase and cornerValues for the current CORE cell
        float cornerValuesF_assembly[8];
        uint cubeCase_assembly = 0;
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellOrigin_for_val_sampling_assembly + cornerOffset;
            uint valU = 0;
            if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                valU = imageLoad(volumeImage, cornerVolCoord).r;
            }
            cornerValuesF_assembly[i] = float(valU);
            if (cornerValuesF_assembly[i] >= ubo.isovalue) cubeCase_assembly |= (1 << i);
        }

        if (cubeCase_assembly == 0 || cubeCase_assembly == 255) continue; // No geometry from this core cell

        // Use the classic Marching Cubes triTable to get triangle definitions
        int baseTriTableIdx = int(cubeCase_assembly * 16); // Assumes triTable is padded to 16 ints per case

        for (int tri_idx_in_cell = 0; tri_idx_in_cell < 5; ++tri_idx_in_cell) { // Max 5 triangles per cell
            if (sharedMeshletPrimitiveCount_actual >= MAX_PRIMITIVES) {
                threadShouldStopTriAssembly = true;
                break;
            }

            int mcEdge0 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 0];
            if (mcEdge0 == -1) break; // End of triangle list for this cell case

            // Assuming your triTable is correct now, mcEdge0/1/2 should be in [0,11]
            int mcEdge1 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 1];
            int mcEdge2 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 2];
            
            // Add validation for mcEdge values if there's still doubt about triTable content
            // (already suggested this in the thought process for previous user response)
            if (mcEdge0 < 0 || mcEdge0 > 11 || mcEdge1 < 0 || mcEdge1 > 11 || mcEdge2 < 0 || mcEdge2 > 11) {
               debugPrintfEXT("MS P2 TABLE CORRUPT");
               continue; // Skip this malformed triangle from table
            }


            int currentTriangleMCEdges[3] = {mcEdge0, mcEdge1, mcEdge2};
            uvec3 currentTriangleLocalIndices; // Will store local meshlet vertex indices
            bool triangleIsValid = true;

            for (int v_idx = 0; v_idx < 3; ++v_idx) { // For each of the 3 vertices of the current triangle
                int mcEdgeID_for_tri_vtx = currentTriangleMCEdges[v_idx];

                // Use neighborMappingTable to find which cell "owns" the vertex on this MC edge
                // and which of its 3 distinct axes it corresponds to.
                EdgeMap mapInfo = neighborMappingTable[mcEdgeID_for_tri_vtx];

                // Calculate the local coordinates (within the extended block) of the cell that owns this vertex
                ivec3 ownerCellLocalCoord_in_ExtBlock = ivec3(localCellCoord_core) + mapInfo.ownerCellOffset_RelativeToCurrentMCcell;

                // Sanity check: owner cell must be within the bounds of the extended region for which Phase 1 generated vertices
                if (any(lessThan(ownerCellLocalCoord_in_ExtBlock, ivec3(0))) ||
                    any(greaterThanEqual(ownerCellLocalCoord_in_ExtBlock, ivec3(extendedProcessingDim)))) {
                    // This error means the neighborMappingTable pointed to an owner cell
                    // that is outside the +1 context layer of the current coreSubBlock.
                    // This would be a logic error in neighborMappingTable or sub-block definition.
                    debugPrintfEXT("MS P2 ERR OWNER_BOUNDS: Task %u, CoreCell (%u,%u,%u) MCedge %d -> OwnerExtCell (%d,%d,%d) is OUTSIDE current ExtendedDim (%u,%u,%u). Skipping tri.\n",
                                   taskPayloadIn.taskId, localCellCoord_core.x, localCellCoord_core.y, localCellCoord_core.z, mcEdgeID_for_tri_vtx,
                                   ownerCellLocalCoord_in_ExtBlock.x, ownerCellLocalCoord_in_ExtBlock.y, ownerCellLocalCoord_in_ExtBlock.z,
                                   extendedProcessingDim.x, extendedProcessingDim.y, extendedProcessingDim.z);
                    triangleIsValid = false;
                    break;
                }

                // Calculate the flat index into distinctEdgeOwner_VertexIdx for the owner cell's distinct edge
                uint distinct_edge_cache_flat_idx =
                    (uint(ownerCellLocalCoord_in_ExtBlock.z) * EXT_DIM_Y * EXT_DIM_X +
                     uint(ownerCellLocalCoord_in_ExtBlock.y) * EXT_DIM_X +
                     uint(ownerCellLocalCoord_in_ExtBlock.x)) * 3 + uint(mapInfo.distinctEdgeAxis_OfOwner);

                uint vertexLocalIdx = distinctEdgeOwner_VertexIdx[distinct_edge_cache_flat_idx];

                if (vertexLocalIdx == 0xFFFFFFFFu) { // Vertex wasn't generated or index is invalid
                    // This is where the "VtxIdx INVALID" error occurs.
                    // Now, Phase 1 only generates vertices for active distinct edges.
                    // If this distinct edge on the owner cell was NOT active (per edgeTable),
                    // then vertexLocalIdx == 0xFFFFFFFFu is the EXPECTED AND CORRECT outcome.
                    // The triangle simply cannot be formed if a required vertex doesn't exist.
                    // debugPrintfEXT("MS P2 INFO VTX_NOT_FOUND: Task %u, CoreCell (%u,%u,%u) MCedge %d -> OwnerExtCell (%d,%d,%d) Axis %d. VtxIdx was 0xFFFFFFFF (CacheIdx %u). Intended? Skipping tri.\n",
                    //                taskPayloadIn.taskId, localCellCoord_core.x, localCellCoord_core.y, localCellCoord_core.z, mcEdgeID_for_tri_vtx,
                    //                ownerCellLocalCoord_in_ExtBlock.x, ownerCellLocalCoord_in_ExtBlock.y, ownerCellLocalCoord_in_ExtBlock.z, mapInfo.distinctEdgeAxis_OfOwner,
                    //                distinct_edge_cache_flat_idx);
                    triangleIsValid = false;
                    break;
                }
                if (vertexLocalIdx >= MAX_VERTICES) { // Index out of bounds for sharedMeshletVertices
                    //  debugPrintfEXT("MS P2 ERR VTX_OOB: Task %u, CoreCell (%u,%u,%u) MCedge %d -> OwnerExtCell (%d,%d,%d) Axis %d. VtxIdx %u >= MAX_VERTICES (%u) (CacheIdx %u). Corrupted? Skipping tri.\n",
                    //                taskPayloadIn.taskId, localCellCoord_core.x, localCellCoord_core.y, localCellCoord_core.z, mcEdgeID_for_tri_vtx,
                    //                ownerCellLocalCoord_in_ExtBlock.x, ownerCellLocalCoord_in_ExtBlock.y, ownerCellLocalCoord_in_ExtBlock.z, mapInfo.distinctEdgeAxis_OfOwner,
                    //                vertexLocalIdx, MAX_VERTICES, distinct_edge_cache_flat_idx);
                    triangleIsValid = false;
                    break;
                }
                currentTriangleLocalIndices[v_idx] = vertexLocalIdx;
            } // End v_idx loop (vertices of a triangle)

            if (triangleIsValid) {
                // Check for degenerate triangles (where two or more local indices are the same)
                if (currentTriangleLocalIndices.x == currentTriangleLocalIndices.y ||
                    currentTriangleLocalIndices.x == currentTriangleLocalIndices.z ||
                    currentTriangleLocalIndices.y == currentTriangleLocalIndices.z) {
                    // debugPrintfEXT("MS P2 DEGEN: Task %u, CoreCell (%u,%u,%u) has degenerate tri (%u,%u,%u).\n",
                    //                taskPayloadIn.taskId, localCellCoord_core.x,localCellCoord_core.y,localCellCoord_core.z,
                    //                currentTriangleLocalIndices.x, currentTriangleLocalIndices.y, currentTriangleLocalIndices.z);
                    continue; // Skip emitting this degenerate triangle
                }

                uint primIdx = atomicAdd(sharedMeshletPrimitiveCount_actual, 1u);
                if (primIdx < MAX_PRIMITIVES) {
                    sharedMeshletIndices[primIdx] = currentTriangleLocalIndices;
                } else {
                    // Primitive limit hit by this thread's atomicAdd or another's.
                    // No need to decrement, but this thread should stop trying to add more.
                    // atomicAdd(sharedMeshletPrimitiveCount_actual, -1u); // Avoid this due to concurrency.
                    threadShouldStopTriAssembly = true; // Signal this thread to stop.
                    // Outer loop checks sharedMeshletPrimitiveCount_actual to stop the whole workgroup.
                }
            } // End if triangleIsValid
            if (sharedMeshletPrimitiveCount_actual >= MAX_PRIMITIVES) {threadShouldStopTriAssembly = true; break;} // Check again before next triangle
        } // End triangle loop for cell (tri_idx_in_cell)
        if (threadShouldStopTriAssembly) break; // Break from cell loop if this thread should stop
    } // End core cell loop (linearCellIdx_core)

    // --- Final Output Stage ---
    barrier(); // Ensure all threads have finished triangle assembly and updated shared counts
    uint finalVertexCount = min(sharedMeshletVertexCount_actual, MAX_VERTICES);
    uint finalPrimitiveCount = min(sharedMeshletPrimitiveCount_actual, MAX_PRIMITIVES);

    SetMeshOutputsEXT(finalVertexCount, finalPrimitiveCount);

    // --- Write to global buffers and mesh output arrays ---
    uint globalVtxBase = taskPayloadIn.globalVertexOffset;
    uint globalIdxBase = taskPayloadIn.globalIndexOffset;
    // Use .length() to get actual buffer capacity at runtime if buffers are dynamically sized.
    // If using fixed-size buffers, ensure these reflect the actual allocated sizes.
    uint effectiveVertexCapacity = (vertices_ssbo.vertex_data.length() == 0 && finalVertexCount > 0) ? globalVtxBase + finalVertexCount + WORKGROUP_SIZE : vertices_ssbo.vertex_data.length();
    uint effectiveIndexCapacity = (indices_ssbo.indices.length() == 0 && finalPrimitiveCount > 0) ? globalIdxBase + finalPrimitiveCount*3 + WORKGROUP_SIZE : indices_ssbo.indices.length();
    uint effectiveDescCapacity = (meshletDescOutput.meshletDescriptors.length() == 0 && finalPrimitiveCount > 0) ? taskPayloadIn.globalMeshletDescOffset + 1 + WORKGROUP_SIZE : meshletDescOutput.meshletDescriptors.length();


    for (uint i = gl_LocalInvocationIndex; i < finalVertexCount; i += WORKGROUP_SIZE) {
        VertexData vData = sharedMeshletVertices[i];
        // Output to built-in gl_MeshVerticesEXT for rasterization
        gl_MeshVerticesEXT[i].gl_Position = vData.position; // Position should be in world space. MVP applied later.
        // Output custom vertex attributes (e.g., normal)
        meshVertexDataOut[i].normal = vData.normal.xyz;

        // Write to global vertex buffer SSBO
        if (globalVtxBase + i < effectiveVertexCapacity) {
            vertices_ssbo.vertex_data[globalVtxBase + i] = vData;
        } else {
            // debugPrintfEXT("MS VTX WRITE OOB: Task %u, Gvo %u, i %u. Total %u >= Cap %u. Skipping write.\n", taskPayloadIn.taskId, globalVtxBase, i, globalVtxBase + i, effectiveVertexCapacity);
        }
    }

    for (uint i = gl_LocalInvocationIndex; i < finalPrimitiveCount; i += WORKGROUP_SIZE) {
        uvec3 localIndicesForPrim = sharedMeshletIndices[i];

        // Sanity check: local indices must be within the meshlet's actual vertex count
        if (localIndicesForPrim.x >= finalVertexCount ||
            localIndicesForPrim.y >= finalVertexCount ||
            localIndicesForPrim.z >= finalVertexCount) {
            // debugPrintfEXT("MS P2 IDX_LOGIC_ERR: Task %u, local indices (%u,%u,%u) out of finalVertexCount %u. Skipping prim.\n", taskPayloadIn.taskId, localIndicesForPrim.x, localIndicesForPrim.y, localIndicesForPrim.z, finalVertexCount);
            continue; // Skip this primitive if its indices are invalid
        }

        // Output to built-in gl_PrimitiveTriangleIndicesEXT for rasterization
        gl_PrimitiveTriangleIndicesEXT[i] = localIndicesForPrim;

        // Write to global index buffer SSBO (indices are global, relative to start of vertex_ssbo)
        if (globalIdxBase + i * 3 + 2 < effectiveIndexCapacity) {
            indices_ssbo.indices[globalIdxBase + i * 3 + 0] = localIndicesForPrim.x + globalVtxBase;
            indices_ssbo.indices[globalIdxBase + i * 3 + 1] = localIndicesForPrim.y + globalVtxBase;
            indices_ssbo.indices[globalIdxBase + i * 3 + 2] = localIndicesForPrim.z + globalVtxBase;
        } else {
            // debugPrintfEXT("MS IDX WRITE OOB: Task %u, Gio %u, i*3+2 %u. Total %u >= Cap %u. Skipping write.\n", taskPayloadIn.taskId, globalIdxBase, i*3+2, globalIdxBase + i * 3 + 2, effectiveIndexCapacity);
        }
    }

    barrier(); // Ensure all SSBO writes are done before the single thread writes the descriptor
    if (gl_LocalInvocationIndex == 0) {
        if (finalVertexCount > 0 && finalPrimitiveCount > 0) {
            // Use the pre-allocated descriptor slot passed from the task shader
            uint descriptorSlotIndex = taskPayloadIn.globalMeshletDescOffset;

            // The task shader already incremented meshletCounter for us.
            // We now use filledMeshletDescCount for the *actual draw indirect* count.
            uint actualDescWriteIdx = atomicAdd(filledMeshletDescCount.filledMeshletCounter, 1u);

            // For safety, ensure we don't write out of bounds if using filledMeshletDescCount directly
            // for indexing into a potentially smaller descriptor buffer than what meshletCounter might imply.
            // However, PMB typically has task shader allocate the slot, and mesh shader fills it.
            // So, descriptorSlotIndex should be the one to use if it's from meshletCounter.
            // If filledMeshletDescCount is for a *compacted* list of descriptors, then actualDescWriteIdx is right.
            // Assuming descriptorSlotIndex IS the correct pre-allocated slot:
            if (descriptorSlotIndex < effectiveDescCapacity) {
                 meshletDescOutput.meshletDescriptors[descriptorSlotIndex].vertexOffset = globalVtxBase;
                 meshletDescOutput.meshletDescriptors[descriptorSlotIndex].indexOffset = globalIdxBase;
                 meshletDescOutput.meshletDescriptors[descriptorSlotIndex].vertexCount = finalVertexCount;
                 meshletDescOutput.meshletDescriptors[descriptorSlotIndex].primitiveCount = finalPrimitiveCount;
            } else {
                // debugPrintfEXT("MS DESC WRITE OOB (PRE-ALLOC): Task %u, descriptorSlotIndex %u. Verts %u Prims %u. Skipping.\n", taskPayloadIn.taskId, descriptorSlotIndex, finalVertexCount, finalPrimitiveCount);
            }
        } else if (taskPayloadIn.subBlockDim.x > 0) { // Only log if it was a real block
            // debugPrintfEXT("MS EMPTY MESHLET SKIPPED: Task %u, SBDim %u %u %u. FinalV %u FinalP %u (ActualShared V:%u P:%u)\n",
            //                taskPayloadIn.taskId, taskPayloadIn.subBlockDim.x, taskPayloadIn.subBlockDim.y, taskPayloadIn.subBlockDim.z,
            //                finalVertexCount, finalPrimitiveCount, sharedMeshletVertexCount_actual, sharedMeshletPrimitiveCount_actual);
        }
    }
}