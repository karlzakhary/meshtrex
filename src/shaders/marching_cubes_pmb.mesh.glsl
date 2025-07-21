#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_ballot: require

// --- Specialization Constants for Dynamic Block Dimensions ---
layout(constant_id = 0) const uint BX = 3u;
layout(constant_id = 1) const uint BY = 3u;
layout(constant_id = 2) const uint BZ = 3u;

// --- Configurable Parameters ---
#define WORKGROUP_SIZE 128u

/* voxel region you must read (core + 1-voxel halo) ----------------- */
const uint BLOCK_DIM_X = BX + 1u;
const uint BLOCK_DIM_Y = BY + 1u;
const uint BLOCK_DIM_Z = BZ + 1u;
const uint STRIDE = BX;  /* overlap = 1 voxel */
#define MAX_PRIMS_PER_CELL 5u
// MAX_CELLS_IN_BLOCK must be a compile-time constant for array sizing
// Use the maximum expected block size (e.g., 4x4x4 = 64)
#define MAX_CELLS_IN_BLOCK 64u
const uint ACTUAL_CELLS_IN_BLOCK = BX * BY * BZ;
const uint MAX_CELLS_PER_THREAD = (MAX_CELLS_IN_BLOCK + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
const uint MAX_PRIMS_PER_THREAD = MAX_CELLS_PER_THREAD * MAX_PRIMS_PER_CELL;
#define MAX_VERTS_PER_MESHLET 64u
#define MAX_PRIMS_PER_MESHLET 126u

/* ------------------  PMB edge ownership -------------------------- */
const uint PMB_EDGE_X = 0u;
const uint PMB_EDGE_Y = 3u;
const uint PMB_EDGE_Z = 8u;

/* neighbour mapping: <dx,dy,dz,edgeType(0=x 1=y 2=z)>               */
/* from Table 1 of the PMB pseudo-code */
const uvec4 edgeOwner[12] = uvec4[12](
  uvec4(0,0,0,0), uvec4(1,0,0,1), uvec4(0,1,0,0), uvec4(0,0,0,1),
  uvec4(0,0,1,0), uvec4(1,0,1,1), uvec4(0,1,1,0), uvec4(0,0,1,1),
  uvec4(0,0,0,2), uvec4(1,0,0,2), uvec4(1,1,0,2), uvec4(0,1,0,2));

// --- Structures ---
struct VertexData {
    vec4 position;
    vec4 normal;
};

struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

#define MAX_MESHLETS_PER_BLOCK 8u
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
    uint meshletCount;
    uint firstCell[MAX_MESHLETS_PER_BLOCK];
    uint cellCount[MAX_MESHLETS_PER_BLOCK];
    uint packedCellData[MAX_CELLS_IN_BLOCK]; // Use max size, but only access ACTUAL_CELLS_IN_BLOCK
} taskPayloadIn;

// --- Descriptor Set Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;
layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesEdgeTable { int edgeTable[]; } mcEdgeTable;
layout(set = 0, binding = 6, std430) buffer VertexBuffer { VertexData data[]; } vertices;
layout(set = 0, binding = 7, std430) buffer VertexCount { uint vertexCounter; } vCount;
layout(set = 0, binding = 8, std430) buffer IndexBuffer { uint data[]; } indices;
layout(set = 0, binding = 9, std430) buffer IndexCount { uint indexCounter; } iCount;
layout(set = 0, binding = 10, std430) buffer MeshletDescriptorBuffer { MeshletDescriptor descriptors[]; } meshlets;
layout(set = 0, binding = 11, std430) buffer MeshletDescriptorCount { uint meshletCounter; } meshletCount;

const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0,0,0),  // 0
    ivec3(1,0,0),  // 1
    ivec3(1,1,0),  // 2
    ivec3(0,1,0),  // 3
    ivec3(0,0,1),  // 4
    ivec3(1,0,1),  // 5
    ivec3(1,1,1),  // 6
    ivec3(0,1,1)   // 7
);
// --- Helper Functions ---
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
    return uvec3(id % grid_width, (id / grid_width) % ubo.blockGridDim.y, id / grid_slice);
}

// This version clamps coordinates to prevent reading outside the volume texture.
vec3 calculate_normal(ivec3 p) {
    ivec3 dims = ivec3(ubo.volumeDim.xyz - 1);
    float s1 = float(imageLoad(volumeImage, clamp(p + ivec3(-1, 0, 0), ivec3(0), dims)).r);
    float s2 = float(imageLoad(volumeImage, clamp(p + ivec3( 1, 0, 0), ivec3(0), dims)).r);
    float s3 = float(imageLoad(volumeImage, clamp(p + ivec3( 0,-1, 0), ivec3(0), dims)).r);
    float s4 = float(imageLoad(volumeImage, clamp(p + ivec3( 0, 1, 0), ivec3(0), dims)).r);
    float s5 = float(imageLoad(volumeImage, clamp(p + ivec3( 0, 0,-1), ivec3(0), dims)).r);
    float s6 = float(imageLoad(volumeImage, clamp(p + ivec3( 0, 0, 1), ivec3(0), dims)).r);
    return normalize(vec3(s1 - s2, s3 - s4, s5 - s6));
}

VertexData interpolate_vertex(float isolevel, ivec3 p1_coord, ivec3 p2_coord) {
    float v1_val = float(imageLoad(volumeImage, p1_coord).r);
    float v2_val = float(imageLoad(volumeImage, p2_coord).r);

    vec3 n1 = calculate_normal(p1_coord);
    vec3 n2 = calculate_normal(p2_coord);

    float mu = 0.5;
    float denominator = v2_val - v1_val;
    if (abs(denominator) > 0.00001) {
        mu = (isolevel - v1_val) / denominator;
    }
    mu = clamp(mu, 0.0, 1.0);
    
    vec3 pos = mix(vec3(p1_coord), vec3(p2_coord), mu);
    vec3 norm = normalize(mix(n1, n2, mu));
    vec3 final_pos = (pos / vec3(ubo.volumeDim.xyz)) * 2.0 - 1.0;

    return VertexData(vec4(final_pos, 1.0), vec4(norm, 0.0));
}

// Computes the marching cubes configuration index for a cell
uint calculate_configuration(ivec3 cell_coord_global) {
    uint configuration = 0;
    for (int i = 0; i < 8; ++i) {
        // Defines the 8 corners of a cube relative to its origin
        ivec3 corner_offset = cornerOffset[i];
        ivec3 neighbor_coord = cell_coord_global + corner_offset;
        float value = float(imageLoad(volumeImage, neighbor_coord).r);
        if (value <= ubo.isovalue) {
            configuration |= (1 << i);
        }
    }
    return configuration;
}


// --- Shared Memory ---
shared VertexData  shVerts[MAX_VERTS_PER_MESHLET];
shared uint        shVertCount;
// Map from [x][y][z][edge_type] in the 5x5x5 context block to a vertex index in shVerts
shared uint        shVertMap[BLOCK_DIM_X][BLOCK_DIM_Y][BLOCK_DIM_Z][3];
shared uint        shPrimCount;
shared uint        shIdx[MAX_PRIMS_PER_MESHLET*3];

shared uvec3 sh_temp_tris[MAX_PRIMS_PER_MESHLET * MAX_PRIMS_PER_THREAD];
shared uint sh_vert_subgroup_sums[WORKGROUP_SIZE];
shared uint sh_prim_subgroup_sums[WORKGROUP_SIZE];

/* helper telling whether a cell in the 5x5x5 context grid owns an edge */
bool ownsX(uvec3 c) { return c.x < BX; } // Core cells 0..3 own their +X edge
bool ownsY(uvec3 c) { return c.y < BY; } // Core cells 0..3 own their +Y edge
bool ownsZ(uvec3 c) { return c.z < BZ; } // Core cells 0..3 own their +Z edge

// --- Workgroup size ---
// NOTE: A workgroup size of 64 is small for this task. 128 is often better.
// The code is written to be flexible, but this must match the host application.
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Output limits ---
layout(max_vertices = MAX_VERTS_PER_MESHLET, max_primitives = MAX_PRIMS_PER_MESHLET) out;
layout(triangles) out;

void main ()
{
    uint meshlet_idx_in_block = gl_WorkGroupID.x;
    if (meshlet_idx_in_block >= taskPayloadIn.meshletCount) return;

    // --- 0. Initialize Shared Memory ---
    if (gl_LocalInvocationIndex == 0) {
        shVertCount = 0u;
        shPrimCount = 0u;
    }
    // Initialize the vertex map. Each thread clears a portion.
    for (uint i = gl_LocalInvocationIndex; i < (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z); i += WORKGROUP_SIZE) {
        uvec3 c = uvec3(i % BLOCK_DIM_X, (i / BLOCK_DIM_X) % BLOCK_DIM_Y, i / (BLOCK_DIM_X * BLOCK_DIM_Y));
        shVertMap[c.x][c.y][c.z][0] = 0xFFFFFFFFu;
        shVertMap[c.x][c.y][c.z][1] = 0xFFFFFFFFu;
        shVertMap[c.x][c.y][c.z][2] = 0xFFFFFFFFu;
    }
    barrier();

    // --- 1. Generate all unique vertices for the 5x5x5 context region ---
    uvec3 blk_coord = unpack_block_id(taskPayloadIn.blockID);
    ivec3 base_coord = ivec3(blk_coord) * int(STRIDE);

    // =================================================================================
    // SECTION 1: THREE-PASS VERTEX GENERATION (SUBGROUP-BASED)
    // This entire section replaces the atomic-based vertex generation.
    // =================================================================================

    // --- PASS 1A: COUNT - Each thread counts how many vertices it owns. ---
    uint local_vert_count = 0;
    const uint TOTAL_CONTEXT_CELLS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    for (uint cell_idx = gl_LocalInvocationIndex; cell_idx < TOTAL_CONTEXT_CELLS; cell_idx += WORKGROUP_SIZE)
    {
        uvec3 cLoc = uvec3(cell_idx % BLOCK_DIM_X, (cell_idx / BLOCK_DIM_X) % BLOCK_DIM_Y, cell_idx / (BLOCK_DIM_X * BLOCK_DIM_Y));
        if (any(greaterThanEqual(base_coord + ivec3(cLoc) + ivec3(1), ivec3(ubo.volumeDim.xyz)))) continue;
        uint eMask = mcEdgeTable.edgeTable[calculate_configuration(base_coord + ivec3(cLoc))];
        if (eMask == 0u) {
            continue;                     // nothing else to do for this cell
        }
        if ((eMask & (1u << PMB_EDGE_X)) != 0u && ownsX(cLoc)) local_vert_count++;
        if ((eMask & (1u << PMB_EDGE_Y)) != 0u && ownsY(cLoc)) local_vert_count++;
        if ((eMask & (1u << PMB_EDGE_Z)) != 0u && ownsZ(cLoc)) local_vert_count++;
    }
    barrier();

    // --- PASS 1B: SCAN - Perform a workgroup-wide scan on the vertex counts. ---
    uint subgroup_vert_offset = subgroupExclusiveAdd(local_vert_count);
    uint subgroup_vert_total  = subgroupAdd(local_vert_count);
    if (subgroupElect()) { 
        sh_vert_subgroup_sums[gl_SubgroupID] = subgroup_vert_total; 
    }
    barrier();

    if (gl_SubgroupID == 0) {
        uint subgroup_sum_val = (gl_SubgroupInvocationID < gl_NumSubgroups) ? sh_vert_subgroup_sums[gl_SubgroupInvocationID] : 0;
        uint subgroup_base_offset = subgroupExclusiveAdd(subgroup_sum_val);
        if (gl_SubgroupInvocationID < gl_NumSubgroups) { sh_vert_subgroup_sums[gl_SubgroupInvocationID] = subgroup_base_offset; }
    }
    barrier();

    uint final_vert_offset = sh_vert_subgroup_sums[gl_SubgroupID] + subgroup_vert_offset;
    if (gl_LocalInvocationIndex == WORKGROUP_SIZE - 1) { 
        shVertCount = final_vert_offset + local_vert_count; 
    }
    barrier();

    // --- PASS 1C: GENERATE & WRITE - Each thread generates its vertices and writes them to the calculated offsets. ---
    uint running_vert_offset = 0;
    for (uint cell_idx = gl_LocalInvocationIndex; cell_idx < TOTAL_CONTEXT_CELLS; cell_idx += WORKGROUP_SIZE) {
        uvec3 cLoc = uvec3(cell_idx % BLOCK_DIM_X, (cell_idx / BLOCK_DIM_X) % BLOCK_DIM_Y, cell_idx / (BLOCK_DIM_X * BLOCK_DIM_Y));
        ivec3 gLoc = base_coord + ivec3(cLoc);
        if (any(greaterThanEqual(gLoc + ivec3(1), ivec3(ubo.volumeDim.xyz)))) continue;

        uint cfg   = calculate_configuration(gLoc);
        uint eMask = mcEdgeTable.edgeTable[cfg];
        if (eMask == 0u) {
            continue;                     // nothing else to do for this cell
        }
        // -- X owner edge --
        if ((eMask & (1u << PMB_EDGE_X)) != 0u && ownsX(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex(ubo.isovalue, gLoc, gLoc + ivec3(1,0,0));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][0] = write_idx;
            }
            running_vert_offset++;
        }
        if ((eMask & (1u << PMB_EDGE_Y)) != 0u && ownsY(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex(ubo.isovalue, gLoc, gLoc + ivec3(0,1,0));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][1] = write_idx;
            }
            running_vert_offset++;
        }
        if ((eMask & (1u << PMB_EDGE_Z)) != 0u && ownsZ(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex(ubo.isovalue, gLoc, gLoc + ivec3(0,0,1));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][2] = write_idx;
            }
            running_vert_offset++;
        }
    }
    barrier();

    // --- 2. Generate triangles for cells belonging to this specific meshlet ---
    uint first_cell_idx = taskPayloadIn.firstCell[meshlet_idx_in_block];
    uint num_cells      = taskPayloadIn.cellCount[meshlet_idx_in_block];
    uint local_prim_count = 0;

    for (uint i = gl_LocalInvocationIndex; i < num_cells; i += WORKGROUP_SIZE)
    {
        uint packed_data = taskPayloadIn.packedCellData[first_cell_idx + i];
        uint cellID = (packed_data >> 16) & 0xFFFFu;
        uint prims  = (packed_data >>  8) & 0xFFu;
        
        // Configuration must be recalculated as it's not passed from task shader
        uvec3 c_core = uvec3(cellID % BX, (cellID / BX) % BY, cellID / (BX*BY));
        ivec3 g_core = base_coord + ivec3(c_core);
        uint  cfg    = calculate_configuration(g_core);

        for (uint t = 0; t < prims; ++t) {
            if (local_prim_count >= MAX_PRIMS_PER_THREAD) continue;
            uvec3 tri_indices;
            int e0 = mcTriangleTable.triTable[cfg*16 + t*3 + 0];
            int e1 = mcTriangleTable.triTable[cfg*16 + t*3 + 1];
            int e2 = mcTriangleTable.triTable[cfg*16 + t*3 + 2];
            
            // Look up the vertex index from the owner cell in the 5x5x5 map
            uvec4 owner0 = edgeOwner[e0];
            uvec4 owner1 = edgeOwner[e1];
            uvec4 owner2 = edgeOwner[e2];

            // c_core is 0..3, owner.xyz is 0..1, result is 0..4
            tri_indices.x = shVertMap[c_core.x+owner0.x][c_core.y+owner0.y][c_core.z+owner0.z][owner0.w];
            tri_indices.y = shVertMap[c_core.x+owner1.x][c_core.y+owner1.y][c_core.z+owner1.z][owner1.w];
            tri_indices.z = shVertMap[c_core.x+owner2.x][c_core.y+owner2.y][c_core.z+owner2.z][owner2.w];

            // Skip triangle if any of its vertices couldn't be generated (e.g., overflow)
            if (tri_indices.x == 0xFFFFFFFFu || tri_indices.y == 0xFFFFFFFFu || tri_indices.z == 0xFFFFFFFFu) {
                continue;
            }

            if (tri_indices.x != 0xFFFFFFFFu && tri_indices.y != 0xFFFFFFFFu && tri_indices.z != 0xFFFFFFFFu) {
                sh_temp_tris[gl_LocalInvocationIndex * MAX_PRIMS_PER_THREAD + local_prim_count] = tri_indices;
                local_prim_count++;
            }
        }
    }
    barrier();

    // --- PASS 2B: SCAN ---
    uint subgroup_prim_offset = subgroupExclusiveAdd(local_prim_count);
    uint subgroup_prim_total  = subgroupAdd(local_prim_count);

    if (subgroupElect()) { sh_prim_subgroup_sums[gl_SubgroupID] = subgroup_prim_total; }
    barrier();
    if (gl_SubgroupID == 0) {
        uint subgroup_sum_val = (gl_SubgroupInvocationID < gl_NumSubgroups) ? sh_prim_subgroup_sums[gl_SubgroupInvocationID] : 0;
        uint subgroup_base_offset = subgroupExclusiveAdd(subgroup_sum_val);
        if (gl_SubgroupInvocationID < gl_NumSubgroups) { sh_prim_subgroup_sums[gl_SubgroupInvocationID] = subgroup_base_offset; }
    }
    barrier();
    uint final_prim_offset = sh_prim_subgroup_sums[gl_SubgroupID] + subgroup_prim_offset;
    if (gl_LocalInvocationIndex == WORKGROUP_SIZE - 1) { shPrimCount = final_prim_offset + local_prim_count; }
    barrier();
    
    // --- PASS 2C: WRITE ---
    for (uint i = 0; i < local_prim_count; ++i) {
        uint write_idx = final_prim_offset + i;
        if (write_idx < MAX_PRIMS_PER_MESHLET) {
            uvec3 tri = sh_temp_tris[gl_LocalInvocationIndex * MAX_PRIMS_PER_THREAD + i];
            shIdx[write_idx * 3 + 0] = tri.x;
            shIdx[write_idx * 3 + 1] = tri.y;
            shIdx[write_idx * 3 + 2] = tri.z;
        }
    }
    barrier();

    // --- 3. Reserve global buffer space and copy data ---
    uint vBase, iBase;
    if (gl_LocalInvocationIndex == 0)
    {
        vBase = atomicAdd(vCount.vertexCounter, shVertCount);
        iBase = atomicAdd(iCount.indexCounter,  shPrimCount * 3u);
    }
    vBase = subgroupBroadcastFirst(vBase);
    iBase = subgroupBroadcastFirst(iBase);

    for (uint v = gl_LocalInvocationIndex; v < shVertCount; v += WORKGROUP_SIZE)
        vertices.data[vBase + v] = shVerts[v];

    for (uint k = gl_LocalInvocationIndex; k < shPrimCount * 3u; k += WORKGROUP_SIZE)
        indices.data[iBase + k] = shIdx[k] + vBase; // Add base offset to local indices

    // --- 4. Write Meshlet Descriptor and dummy output ---
    if (gl_LocalInvocationIndex == 0)
    {
        // This assumes one block produces one meshlet, but the task shader can create more.
        // This part needs to be handled carefully if a block can be split.
        // For now, writing one descriptor per workgroup.
        uint desc_id = atomicAdd(meshletCount.meshletCounter, 1u);
        meshlets.descriptors[desc_id] = MeshletDescriptor(vBase, iBase, shVertCount, shPrimCount);

        // This is an extraction-only shader, so we output zero geometry.
        SetMeshOutputsEXT(0u, 0u);
    }
}