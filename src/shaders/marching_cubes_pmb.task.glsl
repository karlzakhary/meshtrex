#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
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
shared uint  occCount;
shared uint  occList[CELLS_PER_BLOCK];          /* store packed cell data */


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

#define WORKGROUP_SIZE 32
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const uint lane = gl_LocalInvocationIndex;
    if (gl_WorkGroupID.x >= activeBlockCount.count)
    {
        if (lane == 0) EmitMeshTasksEXT(0, 1, 1);
        return;
    }

    if (lane == 0) occCount = 0u;
    barrier();

    // 1. Each thread inspects a subset of the 64 core cells
    uint  blockID  = activeBlockIDs.ids[gl_WorkGroupID.x];
    uvec3 blkCoord = unpack_block_id(blockID);
    ivec3 base     = ivec3(blkCoord) * int(STRIDE);

    for (uint cell = lane; cell < CELLS_PER_BLOCK; cell += gl_WorkGroupSize.x)
    {
        uvec3 cLoc  = uvec3(cell % BX, (cell / BX) % BY, cell / (BX*BY));
        ivec3 cGlob = base + ivec3(cLoc);

        if (any(greaterThanEqual(cGlob, ivec3(ubo.volumeDim.xyz) - 1)))
            continue;

        uint cfg   = calculate_configuration(cGlob);
        uint prims = getPrimitiveCount(cfg);

        if (prims == 0u) continue;

        // *** FIX: Calculate vertices this cell OWNS, not total vertices. ***
        // This is the core of PMB's efficiency.
        uint eMask = mcEdgeTable.edgeTable[cfg];
        uint owner_verts = 0;
        if ((eMask & (1u << PMB_EDGE_X)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Y)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Z)) != 0) owner_verts++;

        uint idx = atomicAdd(occCount, 1u);
        if (idx < CELLS_PER_BLOCK) {
            // Pack all necessary data for the mesh shader
            occList[idx] = (cell & 0xFFFFu) << 16 | (prims & 0xFFu) << 8 | (owner_verts & 0xFFu);
        }
    }
    barrier();

    // 2. Partition occupied cells into meshlets (single thread)
    if (lane == 0)
    {
        uint occ = min(occCount, CELLS_PER_BLOCK);

        for (uint i = 0; i < occ; ++i) TP.packedCellData[i] = occList[i];

        // Greedy partitioning based on corrected vertex and primitive counts
        uint m = 0u, runV = 0u, runP = 0u, first = 0u;
        for (uint i = 0; i < occ; ++i)
        {
            uint packed_data = occList[i];
            uint prims       = (packed_data >> 8) & 0xFFu;
            uint owner_verts = (packed_data)      & 0xFFu;

            bool over =
              (runV + owner_verts > MAX_VERTS_PER_MESHLET) ||
              (runP + prims > MAX_PRIMS_PER_MESHLET);

            if (over && runP != 0)
            {
                TP.firstCell[m] = first;
                TP.cellCount[m] = i - first;
                m++;
                first = i;
                runV = 0u;
                runP = 0u;
            }
            runV += owner_verts;
            runP += prims;
        }

        if (runP != 0)
        {
            TP.firstCell[m] = first;
            TP.cellCount[m] = occ - first;
            m++;
        }

        TP.meshletCount = m;
        TP.blockID      = blockID;

        EmitMeshTasksEXT(m, 1, 1);
    }
}