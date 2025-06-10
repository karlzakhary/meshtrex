#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
#define WORKGROUP_SIZE 32
/* core-cell grid ---------------------------------------------------- */
#define BX 4u
#define BY 4u
#define BZ 4u
#define CELLS_PER_BLOCK 64u      /* 4×4×4 */

/* voxel region you must read (core + 1-voxel halo) ----------------- */
#define STRIDE 4u           /* overlap = 1 voxel */

#define MAX_VERTS_PER_MESHLET 256u
#define MAX_PRIMS_PER_MESHLET 256u
#define MAX_MESHLETS_PER_BLOCK 8u        /* 64 cells / 5-tris ~= 13 → 8 is safe for 4³ */
#define MAX_OCC_CELLS_PER_THREAD (CELLS_PER_BLOCK + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE

// --- PMB Vertex Ownership Edges ---
// These are the indices of the 3 edges a cell is responsible for generating vertices on.
const uint PMB_EDGE_X = 0u; // Edge from corner 0 to 1
const uint PMB_EDGE_Y = 3u; // Edge from corner 0 to 3
const uint PMB_EDGE_Z = 8u; // Edge from corner 0 to 4

taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
    uint meshletCount;
    uint firstCell[MAX_MESHLETS_PER_BLOCK];
    uint cellCount[MAX_MESHLETS_PER_BLOCK];
    // Storing more info: (cellID << 16) | (prims << 8) | owner_verts
    uint packedCellData[CELLS_PER_BLOCK];
} TP;


/* ---------------- shared scratch ------------------------------------------ */
shared uint sh_temp_occ_list[WORKGROUP_SIZE * MAX_OCC_CELLS_PER_THREAD];
shared uint sh_subgroup_sums[32];
shared uint total_occ_count;


// --- Descriptor Set Bindings ---
// (Bindings are unchanged)
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount { uint count; } activeBlockCount;
layout(set = 0, binding = 3, std430) readonly buffer ActiveBlockIDs { uint ids[]; } activeBlockIDs;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;
layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesEdgeTable { int edgeTable[]; } mcEdgeTable;

// --- Helper Functions (Unchanged) ---
uint getPrimitiveCount(uint configuration) {
    uint primitiveCount = 0;
    for (int i = 0; i < 5; i++) {
        if (mcTriangleTable.triTable[configuration * 16 + i * 3] == -1) break;
        primitiveCount++;
    }
    return primitiveCount;
}

uvec3 unpack_block_id(uint id) {
    uint grid_width = ubo.blockGridDim.x;
    uint grid_slice = ubo.blockGridDim.x * ubo.blockGridDim.y;
    return uvec3( id % grid_width, (id / grid_width) % ubo.blockGridDim.y, id / grid_slice );
}

const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1)
);

uint calculate_configuration(ivec3 cell_coord_global) {
    uint configuration = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 neighbor_coord = cell_coord_global + cornerOffset[i];
        float value = float(imageLoad(volumeImage, neighbor_coord).r);
        if (value <= ubo.isovalue) {
            configuration |= (1 << i);
        }
    }
    return configuration;
}


layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const uint lane = gl_LocalInvocationIndex;
    if (gl_WorkGroupID.x >= activeBlockCount.count)
    {
        if (lane == 0) EmitMeshTasksEXT(0, 1, 1);
        return;
    }


    // =======================================================================
    // PASS 1: Find occupied cells and store their data temporarily
    // =======================================================================
    uint local_occ_count = 0;
    uint blockID = activeBlockIDs.ids[gl_WorkGroupID.x];
    ivec3 base = ivec3(unpack_block_id(blockID)) * int(STRIDE);

    for (uint cell = lane; cell < CELLS_PER_BLOCK; cell += WORKGROUP_SIZE)
    {
        uvec3 cLoc = uvec3(cell % BX, (cell / BX) % BY, cell / (BX * BY));
        if (any(greaterThanEqual(base + ivec3(cLoc), ivec3(ubo.volumeDim.xyz) - 1))) continue;
        
        uint cfg = calculate_configuration(base + ivec3(cLoc));
        uint prims = getPrimitiveCount(cfg);
        if (prims == 0u) continue;

        uint eMask = mcEdgeTable.edgeTable[cfg];
        if (eMask == 0u) {
            continue;                     // nothing else to do for this cell
        }
        uint owner_verts = 0;
        if ((eMask & (1u << PMB_EDGE_X)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Y)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Z)) != 0) owner_verts++;

        // Instead of atomicAdd, write to a private temporary slot
        if (local_occ_count < MAX_OCC_CELLS_PER_THREAD) {
            uint packed_data = (cell & 0xFFFFu) << 16 | (prims & 0xFFu) << 8 | (owner_verts & 0xFFu);
            sh_temp_occ_list[lane * MAX_OCC_CELLS_PER_THREAD + local_occ_count] = packed_data;
            local_occ_count++;
        }
    }
    barrier();

    // =======================================================================
    // PASS 2: Perform parallel scan to find compacted write offsets
    // =======================================================================
    uint subgroup_offset = subgroupExclusiveAdd(local_occ_count);
    uint subgroup_total = subgroupAdd(local_occ_count);
    if (subgroupElect()) {
        sh_subgroup_sums[gl_SubgroupID] = subgroup_total;
    }
    barrier();
    if (gl_SubgroupID == 0) {
        uint subgroup_sum_val = (gl_SubgroupInvocationID < gl_NumSubgroups) ? sh_subgroup_sums[gl_SubgroupInvocationID] : 0;
        uint subgroup_base_offset = subgroupExclusiveAdd(subgroup_sum_val);
        if (gl_SubgroupInvocationID < gl_NumSubgroups) {
            sh_subgroup_sums[gl_SubgroupInvocationID] = subgroup_base_offset;
        }
    }
    barrier();
    uint final_occ_offset = sh_subgroup_sums[gl_SubgroupID] + subgroup_offset;
    if (lane == WORKGROUP_SIZE - 1) {
        total_occ_count = final_occ_offset + local_occ_count;
    }
    barrier();

    // =======================================================================
    // PASS 3: Write the compacted list to the final TaskPayload
    // =======================================================================
    for (uint i = 0; i < local_occ_count; i++) {
        uint write_idx = final_occ_offset + i;
        if (write_idx < CELLS_PER_BLOCK) {
            TP.packedCellData[write_idx] = sh_temp_occ_list[lane * MAX_OCC_CELLS_PER_THREAD + i];
        }
    }
    barrier();

    // =======================================================================
    // Final Partitioning (only thread 0) - This logic is now fed by the scan result
    // =======================================================================
    if (lane == 0)
    {
        uint occ = total_occ_count; // Use the precise count from the scan
        uint m = 0;
        if (occ > 0) {
            uint runV = 0, runP = 0, first = 0;
            for (uint i = 0; i < occ; ++i) {
                // Read from the now-compacted list in the TaskPayload
                uint packed_data = TP.packedCellData[i];
                uint prims = (packed_data >> 8) & 0xFFu;
                uint owner_verts = packed_data & 0xFFu;

                if (i > first && (runV + owner_verts > MAX_VERTS_PER_MESHLET || runP + prims > MAX_PRIMS_PER_MESHLET)) {
                    TP.firstCell[m] = first;
                    TP.cellCount[m] = i - first;
                    m++;
                    first = i;
                    runV = 0;
                    runP = 0;
                }
                runV += owner_verts;
                runP += prims;
            }

            if (m < MAX_MESHLETS_PER_BLOCK) {
                TP.firstCell[m] = first;
                TP.cellCount[m] = occ - first;
                m++;
            }
        }
        TP.meshletCount = m;
        TP.blockID = blockID;
        EmitMeshTasksEXT(m, 1, 1);
    }
}