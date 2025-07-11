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
    uvec3 pageCoord;
    uint mipLevel;
} TP;

/* ---------------- shared scratch ------------------------------------------ */
shared uint sh_temp_occ_list[WORKGROUP_SIZE * MAX_OCC_CELLS_PER_THREAD];
shared uint sh_subgroup_sums[32];
shared uint total_occ_count;

// --- Push Constants ---
layout(push_constant) uniform PushConstants {
    uvec3 pageCoord;      // Page coordinate for this pass
    uint mipLevel;        // Mip level
    float isoValue;       // Isovalue for filtering
    uint blockSize;       // Block size (typically 4)
    uint pageSizeX;       // Page size X (64)
    uint pageSizeY;       // Page size Y (32)
    uint pageSizeZ;       // Page size Z (32)
    // Global buffer offsets for persistent geometry extraction
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletOffset;
    uint volumeSizeX;     // Full volume dimension X
    uint volumeSizeY;     // Full volume dimension Y
    uint volumeSizeZ;     // Full volume dimension Z
    uint padding[2];
} pc;

// --- Descriptor Set Bindings ---
layout(set = 0, binding = 0, std430) readonly buffer PageTable {
    uvec2 pageEntries[];
} pageTable;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeAtlas;

layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount { 
    uint count; 
} activeBlockCount;

layout(set = 0, binding = 3, std430) readonly buffer ActiveBlockIDs { 
    uint ids[]; 
} activeBlockIDs;

layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { 
    int triTable[]; 
} mcTriangleTable;

layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesEdgeTable { 
    int edgeTable[]; 
} mcEdgeTable;

// --- Helper Functions ---

uint getPageIndex(uvec3 pageCoord, uint mipLevel) {
    // Calculate based on actual volume dimensions from push constants
    // This needs to match the CPU-side calculation in streamingSystem.cpp
    uint pagesX = (pc.volumeSizeX + pc.pageSizeX - 1) / pc.pageSizeX;
    uint pagesY = (pc.volumeSizeY + pc.pageSizeY - 1) / pc.pageSizeY;
    return pageCoord.z * pagesX * pagesY + pageCoord.y * pagesX + pageCoord.x;
}

bool isPageResident(uvec3 pageCoord, uint mipLevel) {
    uint pageIndex = getPageIndex(pageCoord, mipLevel);
    if (pageIndex >= pageTable.pageEntries.length()) return false;
    return pageTable.pageEntries[pageIndex].y != 0;
}

uvec3 getAtlasCoord(uvec3 pageCoord, uint mipLevel) {
    uint pageIndex = getPageIndex(pageCoord, mipLevel);
    uint atlasCoord = pageTable.pageEntries[pageIndex].x;
    
    return uvec3(
        atlasCoord & 0x3FF,
        (atlasCoord >> 10) & 0x3FF,
        (atlasCoord >> 20) & 0x3FF
    );
}

uint sampleVolumeAtlas(uvec3 worldCoord) {
    uvec3 pageCoord = uvec3(worldCoord.x / pc.pageSizeX,
                            worldCoord.y / pc.pageSizeY,
                            worldCoord.z / pc.pageSizeZ);
    
    if (!isPageResident(pageCoord, pc.mipLevel)) {
        return 0;
    }
    
    uvec3 atlasCoord = getAtlasCoord(pageCoord, pc.mipLevel);
    uvec3 localCoord = uvec3(worldCoord.x % pc.pageSizeX,
                             worldCoord.y % pc.pageSizeY,
                             worldCoord.z % pc.pageSizeZ);
    
    // atlasCoord from page table is in granularity units (divided by granularity during packing)
    // We need to multiply by granularity to get back to texel coordinates
    ivec3 atlasTexel = ivec3(atlasCoord.x * 64 + localCoord.x,
                             atlasCoord.y * 32 + localCoord.y,
                             atlasCoord.z * 32 + localCoord.z);
    
    // Use imageLoad for integer textures
    uint value = imageLoad(volumeAtlas, atlasTexel).r;
    
    return value;
}

uint getPrimitiveCount(uint configuration) {
    uint primitiveCount = 0;
    for (int i = 0; i < 5; i++) {
        if (mcTriangleTable.triTable[configuration * 16 + i * 3] == -1) break;
        primitiveCount++;
    }
    return primitiveCount;
}

uvec3 unpack_block_id(uint id) {
    // For streaming, calculate blocks within the current page
    uint blocksPerPageX = pc.pageSizeX / pc.blockSize;
    uint blocksPerPageY = pc.pageSizeY / pc.blockSize;
    uint blocksPerPageZ = pc.pageSizeZ / pc.blockSize;
    uint grid_slice = blocksPerPageX * blocksPerPageY;
    return uvec3(
        id % blocksPerPageX, 
        (id / blocksPerPageX) % blocksPerPageY, 
        id / grid_slice
    );
}

const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1)
);

uint calculate_configuration_streaming(ivec3 cell_coord_global) {
    uint configuration = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 neighbor_coord = cell_coord_global + cornerOffset[i];
        
        // Sample from volume atlas with page residency check
        uint value = sampleVolumeAtlas(uvec3(max(neighbor_coord, ivec3(0))));
        
        // Debug first few samples
        if (gl_WorkGroupID.x == 0 && cell_coord_global.x < 4 && cell_coord_global.y < 4 && cell_coord_global.z < 4 && i == 0) {
            // debugPrintfEXT("Config calc: coord(%d,%d,%d) value=%d, isoValue=%f, test=%d",
            //               neighbor_coord.x, neighbor_coord.y, neighbor_coord.z, 
            //               value, pc.isoValue, (float(value) <= pc.isoValue) ? 1 : 0);
        }
        
        if (float(value) <= pc.isoValue) {
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
    
    // Convert block ID to 3D coordinates within the page
    uvec3 blockCoord = unpack_block_id(blockID);
    
    // Debug block and page info
    if (gl_WorkGroupID.x == 0 && lane == 0) {
        // debugPrintfEXT("Task shader: blockID=%d, blockCoord=(%d,%d,%d), pageCoord=(%d,%d,%d), pageSize=(%d,%d,%d)",
        //               blockID, blockCoord.x, blockCoord.y, blockCoord.z,
        //               pc.pageCoord.x, pc.pageCoord.y, pc.pageCoord.z,
        //               pc.pageSizeX, pc.pageSizeY, pc.pageSizeZ);
    }
    
    // Calculate world coordinates for this block
    ivec3 pageStart = ivec3(pc.pageCoord.x * pc.pageSizeX,
                            pc.pageCoord.y * pc.pageSizeY,
                            pc.pageCoord.z * pc.pageSizeZ);
    ivec3 base = pageStart + ivec3(blockCoord * pc.blockSize);

    for (uint cell = lane; cell < CELLS_PER_BLOCK; cell += WORKGROUP_SIZE)
    {
        uvec3 cLoc = uvec3(cell % BX, (cell / BX) % BY, cell / (BX * BY));
        ivec3 cellWorldCoord = base + ivec3(cLoc);
        
        // Check if we're within volume bounds - need one voxel margin for marching cubes
        if (any(lessThan(cellWorldCoord, ivec3(0))) || 
            any(greaterThanEqual(cellWorldCoord, ivec3(pc.volumeSizeX, pc.volumeSizeY, pc.volumeSizeZ) - 1))) {
            continue;
        }
        
        // Check page residency for this cell
        uvec3 cellPageCoord = uvec3(cellWorldCoord.x / int(pc.pageSizeX),
                                    cellWorldCoord.y / int(pc.pageSizeY),
                                    cellWorldCoord.z / int(pc.pageSizeZ));
        bool isResident = isPageResident(cellPageCoord, pc.mipLevel);
        
        // Debug residency check
        if (gl_WorkGroupID.x == 0 && cell < 4) {
            // debugPrintfEXT("Task shader: Cell %d page residency check - pageCoord(%d,%d,%d), resident=%d", 
            //               cell, cellPageCoord.x, cellPageCoord.y, cellPageCoord.z, isResident ? 1 : 0);
        }
        
        if (!isResident) continue;
        
        uint cfg = calculate_configuration_streaming(cellWorldCoord);
        uint prims = getPrimitiveCount(cfg);
        
        // Debug output for first few cells
        if (gl_WorkGroupID.x == 0 && cell < 4) {
            // debugPrintfEXT("Task shader: Cell %d, worldCoord (%d,%d,%d), cfg=%d, prims=%d", 
            //               cell, cellWorldCoord.x, cellWorldCoord.y, cellWorldCoord.z, cfg, prims);
        }
        
        if (prims == 0u) continue;

        uint eMask = mcEdgeTable.edgeTable[cfg];
        if (eMask == 0u) {
            continue;
        }
        
        uint owner_verts = 0;
        if ((eMask & (1u << PMB_EDGE_X)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Y)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Z)) != 0) owner_verts++;

        // Store to temporary list
        if (local_occ_count < MAX_OCC_CELLS_PER_THREAD) {
            uint packed_data = (cell & 0xFFFFu) << 16 | (prims & 0xFFu) << 8 | (owner_verts & 0xFFu);
            sh_temp_occ_list[lane * MAX_OCC_CELLS_PER_THREAD + local_occ_count] = packed_data;
            
            // Debug cell packing
            if (gl_WorkGroupID.x == 0 && local_occ_count == 0 && lane < 4) {
                // debugPrintfEXT("Task packing: lane=%d, cell=%d, cLoc=(%d,%d,%d), world=(%d,%d,%d), prims=%d",
                //               lane, cell, cLoc.x, cLoc.y, cLoc.z, 
                //               cellWorldCoord.x, cellWorldCoord.y, cellWorldCoord.z, prims);
            }
            
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
    // Final Partitioning (only thread 0)
    // =======================================================================
    if (lane == 0)
    {
        uint occ = total_occ_count;
        uint m = 0;
        
        if (occ > 0) {
            uint runV = 0, runP = 0, first = 0;
            for (uint i = 0; i < occ; ++i) {
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
        TP.pageCoord = pc.pageCoord;
        TP.mipLevel = pc.mipLevel;
        
        if (gl_WorkGroupID.x < 2) {
            // debugPrintfEXT("Task: WG=%d emitting %d meshlets, %d occupied cells", 
            //               gl_WorkGroupID.x, m, occ);
        }
        
        EmitMeshTasksEXT(m, 1, 1);
    }
}