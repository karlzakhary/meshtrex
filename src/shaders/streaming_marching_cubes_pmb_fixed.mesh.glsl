#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_ballot: require

// --- Configurable Parameters ---
#define WORKGROUP_SIZE 128u
/* core-cell grid ---------------------------------------------------- */
#define BX 4u
#define BY 4u
#define BZ 4u
#define CELLS_PER_BLOCK 64u  /* 4×4×4 */

/* voxel region you must read (core + 1-voxel halo) ----------------- */
#define BLOCK_DIM_X 5u
#define BLOCK_DIM_Y 5u
#define BLOCK_DIM_Z 5u
#define STRIDE      4u           /* overlap = 1 voxel */
#define MAX_PRIMS_PER_CELL 5u
#define MAX_CELLS_IN_BLOCK 64u
#define MAX_CELLS_PER_THREAD (MAX_CELLS_IN_BLOCK + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
#define MAX_PRIMS_PER_THREAD (MAX_CELLS_PER_THREAD * MAX_PRIMS_PER_CELL)
#define MAX_VERTS_PER_MESHLET 256u
#define MAX_PRIMS_PER_MESHLET 256u

/* ------------------  PMB edge ownership -------------------------- */
const uint PMB_EDGE_X = 0u;
const uint PMB_EDGE_Y = 3u;
const uint PMB_EDGE_Z = 8u;

/* neighbour mapping: <dx,dy,dz,edgeType(0=x 1=y 2=z)>               */
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
    uint packedCellData[BX * BY * BZ];
    uvec3 pageCoord;
    uint mipLevel;
} taskPayloadIn;

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
    uint globalPadding;
} pc;

// --- Descriptor Set Bindings ---
layout(set = 0, binding = 0, std430) readonly buffer PageTable {
    uvec2 pageEntries[];
} pageTable;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeAtlas;

layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { 
    int triTable[]; 
} mcTriangleTable;

layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesEdgeTable { 
    int edgeTable[]; 
} mcEdgeTable;

layout(set = 0, binding = 6, std430) buffer VertexBuffer { 
    VertexData data[]; 
} vertices;

layout(set = 0, binding = 7, std430) buffer VertexCount { 
    uint vertexCounter; 
} vCount;

layout(set = 0, binding = 8, std430) buffer IndexBuffer { 
    uint data[]; 
} indices;

layout(set = 0, binding = 9, std430) buffer IndexCount { 
    uint indexCounter; 
} iCount;

layout(set = 0, binding = 10, std430) buffer MeshletDescriptorBuffer { 
    MeshletDescriptor descriptors[]; 
} meshlets;

layout(set = 0, binding = 11, std430) buffer MeshletDescriptorCount { 
    uint meshletCounter; 
} meshletCount;

// --- Shared Memory for PMB ---
shared VertexData shVerts[MAX_VERTS_PER_MESHLET];
shared uint shVertMap[BLOCK_DIM_X][BLOCK_DIM_Y][BLOCK_DIM_Z][3];  // 5x5x5x3 for edge ownership
shared uint shVertCount;
shared uint shIdx[MAX_PRIMS_PER_MESHLET * 3];
shared uint shPrimCount;
shared uint sh_subgroup_sums[32];
shared uint sh_prim_subgroup_sums[32];
shared uvec3 sh_temp_tris[WORKGROUP_SIZE * MAX_PRIMS_PER_THREAD];

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

// Edge ownership functions
bool ownsX(uvec3 cLoc) { return cLoc.x < BX; }
bool ownsY(uvec3 cLoc) { return cLoc.y < BY; }
bool ownsZ(uvec3 cLoc) { return cLoc.z < BZ; }

// --- Helper Functions ---
uint getPageIndex(uvec3 pageCoord, uint mipLevel) {
    uint pagesX = (256 + pc.pageSizeX - 1) / pc.pageSizeX;
    uint pagesY = (256 + pc.pageSizeY - 1) / pc.pageSizeY;
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
    uvec3 pageCoord = uvec3(worldCoord.x / pc.pageSizeX, worldCoord.y / pc.pageSizeY, worldCoord.z / pc.pageSizeZ);
    
    if (!isPageResident(pageCoord, pc.mipLevel)) {
        return 0;
    }
    
    uvec3 atlasCoord = getAtlasCoord(pageCoord, pc.mipLevel);
    uvec3 localCoord = uvec3(worldCoord.x % pc.pageSizeX, worldCoord.y % pc.pageSizeY, worldCoord.z % pc.pageSizeZ);
    
    ivec3 atlasTexel = ivec3(atlasCoord.x * 64 + localCoord.x,
                             atlasCoord.y * 32 + localCoord.y,
                             atlasCoord.z * 32 + localCoord.z);
    
    return imageLoad(volumeAtlas, atlasTexel).r;
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

vec3 calculate_normal_streaming(ivec3 p) {
    float s1 = float(sampleVolumeAtlas(uvec3(max(p + ivec3(-1, 0, 0), ivec3(0)))));
    float s2 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 1, 0, 0), ivec3(0)))));
    float s3 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0,-1, 0), ivec3(0)))));
    float s4 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0, 1, 0), ivec3(0)))));
    float s5 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0, 0,-1), ivec3(0)))));
    float s6 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0, 0, 1), ivec3(0)))));
    return normalize(vec3(s1 - s2, s3 - s4, s5 - s6));
}

VertexData interpolate_vertex_streaming(float isolevel, ivec3 p1_coord, ivec3 p2_coord) {
    float v1_val = float(sampleVolumeAtlas(uvec3(max(p1_coord, ivec3(0)))));
    float v2_val = float(sampleVolumeAtlas(uvec3(max(p2_coord, ivec3(0)))));

    vec3 n1 = calculate_normal_streaming(p1_coord);
    vec3 n2 = calculate_normal_streaming(p2_coord);

    float mu = 0.5;
    float denominator = v2_val - v1_val;
    if (abs(denominator) > 0.00001) {
        mu = (isolevel - v1_val) / denominator;
    }
    mu = clamp(mu, 0.0, 1.0);
    
    vec3 pos = mix(vec3(p1_coord), vec3(p2_coord), mu);
    vec3 norm = normalize(mix(n1, n2, mu));
    
    vec3 volumeSize = vec3(pc.pageSizeX, pc.pageSizeY, pc.pageSizeZ);
    vec3 final_pos = (pos / volumeSize) * 2.0 - 1.0;

    return VertexData(vec4(final_pos, 1.0), vec4(norm, 0.0));
}

uint calculate_configuration_streaming(ivec3 cell_coord_global) {
    uint configuration = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 corner_offset = cornerOffset[i];
        ivec3 neighbor_coord = cell_coord_global + corner_offset;
        uint value = sampleVolumeAtlas(uvec3(max(neighbor_coord, ivec3(0))));
        
        if (float(value) <= pc.isoValue) {
            configuration |= (1 << i);
        }
    }
    return configuration;
}

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Output limits ---
layout(max_vertices = 256, max_primitives = 256) out;
layout(triangles) out;

void main()
{
    const uint lane = gl_LocalInvocationIndex;
    const uint meshletID = gl_WorkGroupID.x;
    
    if (meshletID >= taskPayloadIn.meshletCount) {
        return;
    }

    // Initialize shared memory
    if (lane == 0) {
        shVertCount = 0;
        shPrimCount = 0;
    }
    
    // Initialize vertex map to invalid
    for (uint i = lane; i < BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z * 3; i += WORKGROUP_SIZE) {
        uint z = i / (BLOCK_DIM_X * BLOCK_DIM_Y * 3);
        uint rem = i % (BLOCK_DIM_X * BLOCK_DIM_Y * 3);
        uint y = rem / (BLOCK_DIM_X * 3);
        uint rem2 = rem % (BLOCK_DIM_X * 3);
        uint x = rem2 / 3;
        uint edge = rem2 % 3;
        
        if (x < BLOCK_DIM_X && y < BLOCK_DIM_Y && z < BLOCK_DIM_Z && edge < 3) {
            shVertMap[x][y][z][edge] = 0xFFFFFFFFu;
        }
    }
    barrier();

    uint blockID = taskPayloadIn.blockID;
    uvec3 blockCoord = unpack_block_id(blockID);
    
    // Calculate world coordinates for this block
    ivec3 pageStart = ivec3(taskPayloadIn.pageCoord.x * pc.pageSizeX,
                            taskPayloadIn.pageCoord.y * pc.pageSizeY,
                            taskPayloadIn.pageCoord.z * pc.pageSizeZ);
    ivec3 base = pageStart + ivec3(blockCoord * pc.blockSize);

    // === PASS 1: Generate vertices for owned edges ===
    uint firstCell = taskPayloadIn.firstCell[meshletID];
    uint cellCount = taskPayloadIn.cellCount[meshletID];
    
    // Count vertices needed
    uint localVertCount = 0;
    for (uint cellIdx = firstCell + lane; cellIdx < firstCell + cellCount; cellIdx += WORKGROUP_SIZE) {
        if (cellIdx >= CELLS_PER_BLOCK) continue;
        
        uint packed_data = taskPayloadIn.packedCellData[cellIdx];
        uint owner_verts = packed_data & 0xFFu;
        localVertCount += owner_verts;
    }
    
    // Scan to get offsets
    uint subgroup_offset = subgroupExclusiveAdd(localVertCount);
    uint subgroup_total = subgroupAdd(localVertCount);
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
    
    uint final_vert_offset = sh_subgroup_sums[gl_SubgroupID] + subgroup_offset;
    if (lane == WORKGROUP_SIZE - 1) {
        shVertCount = final_vert_offset + localVertCount;
    }
    barrier();

    // Generate vertices for owned edges
    uint running_vert_offset = 0;
    for (uint cellIdx = firstCell + lane; cellIdx < firstCell + cellCount; cellIdx += WORKGROUP_SIZE) {
        if (cellIdx >= CELLS_PER_BLOCK) continue;
        
        uint packed_data = taskPayloadIn.packedCellData[cellIdx];
        uint cell = (packed_data >> 16) & 0xFFFFu;
        uint owner_verts = packed_data & 0xFFu;
        
        if (owner_verts == 0) continue;
        
        uvec3 cLoc = uvec3(cell % BX, (cell / BX) % BY, cell / (BX * BY));
        ivec3 cellWorldCoord = base + ivec3(cLoc);
        
        uint cfg = calculate_configuration_streaming(cellWorldCoord);
        uint eMask = mcEdgeTable.edgeTable[cfg];
        
        // Generate owned vertices
        if ((eMask & (1u << PMB_EDGE_X)) != 0 && ownsX(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex_streaming(pc.isoValue, cellWorldCoord, cellWorldCoord + ivec3(1,0,0));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][0] = write_idx;
            }
            running_vert_offset++;
        }
        
        if ((eMask & (1u << PMB_EDGE_Y)) != 0 && ownsY(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex_streaming(pc.isoValue, cellWorldCoord, cellWorldCoord + ivec3(0,1,0));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][1] = write_idx;
            }
            running_vert_offset++;
        }
        
        if ((eMask & (1u << PMB_EDGE_Z)) != 0 && ownsZ(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex_streaming(pc.isoValue, cellWorldCoord, cellWorldCoord + ivec3(0,0,1));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][2] = write_idx;
            }
            running_vert_offset++;
        }
    }
    barrier();

    // === PASS 2: Generate triangles using vertex map ===
    uint local_prim_count = 0;
    
    for (uint cellIdx = firstCell + lane; cellIdx < firstCell + cellCount; cellIdx += WORKGROUP_SIZE) {
        if (cellIdx >= CELLS_PER_BLOCK) continue;
        
        uint packed_data = taskPayloadIn.packedCellData[cellIdx];
        uint cell = (packed_data >> 16) & 0xFFFFu;
        uint prims = (packed_data >> 8) & 0xFFu;
        
        if (prims == 0) continue;
        
        uvec3 cLoc = uvec3(cell % BX, (cell / BX) % BY, cell / (BX * BY));
        ivec3 cellWorldCoord = base + ivec3(cLoc);
        uint cfg = calculate_configuration_streaming(cellWorldCoord);
        
        for (uint t = 0; t < prims; ++t) {
            if (local_prim_count >= MAX_PRIMS_PER_THREAD) continue;
            
            int e0 = mcTriangleTable.triTable[cfg * 16 + t * 3 + 0];
            int e1 = mcTriangleTable.triTable[cfg * 16 + t * 3 + 1];
            int e2 = mcTriangleTable.triTable[cfg * 16 + t * 3 + 2];
            
            if (e0 == -1 || e1 == -1 || e2 == -1) break;
            
            // Look up vertices from owner cells
            uvec4 owner0 = edgeOwner[e0];
            uvec4 owner1 = edgeOwner[e1];
            uvec4 owner2 = edgeOwner[e2];
            
            uvec3 tri_indices;
            tri_indices.x = shVertMap[cLoc.x + owner0.x][cLoc.y + owner0.y][cLoc.z + owner0.z][owner0.w];
            tri_indices.y = shVertMap[cLoc.x + owner1.x][cLoc.y + owner1.y][cLoc.z + owner1.z][owner1.w];
            tri_indices.z = shVertMap[cLoc.x + owner2.x][cLoc.y + owner2.y][cLoc.z + owner2.z][owner2.w];
            
            // Skip triangle if any vertex is invalid
            if (tri_indices.x != 0xFFFFFFFFu && tri_indices.y != 0xFFFFFFFFu && tri_indices.z != 0xFFFFFFFFu) {
                sh_temp_tris[gl_LocalInvocationIndex * MAX_PRIMS_PER_THREAD + local_prim_count] = tri_indices;
                local_prim_count++;
            }
        }
    }
    barrier();

    // Scan primitives
    uint subgroup_prim_offset = subgroupExclusiveAdd(local_prim_count);
    uint subgroup_prim_total = subgroupAdd(local_prim_count);
    
    if (subgroupElect()) { sh_prim_subgroup_sums[gl_SubgroupID] = subgroup_prim_total; }
    barrier();
    
    if (gl_SubgroupID == 0) {
        uint subgroup_sum_val = (gl_SubgroupInvocationID < gl_NumSubgroups) ? sh_prim_subgroup_sums[gl_SubgroupInvocationID] : 0;
        uint subgroup_base_offset = subgroupExclusiveAdd(subgroup_sum_val);
        if (gl_SubgroupInvocationID < gl_NumSubgroups) { 
            sh_prim_subgroup_sums[gl_SubgroupInvocationID] = subgroup_base_offset; 
        }
    }
    barrier();
    
    uint final_prim_offset = sh_prim_subgroup_sums[gl_SubgroupID] + subgroup_prim_offset;
    if (gl_LocalInvocationIndex == WORKGROUP_SIZE - 1) { 
        shPrimCount = final_prim_offset + local_prim_count; 
    }
    barrier();
    
    // Write triangles to shared memory
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

    // === PASS 3: Write to global buffers ===
    uint vBase, iBase;
    if (gl_LocalInvocationIndex == 0) {
        vBase = pc.globalVertexOffset + atomicAdd(vCount.vertexCounter, shVertCount);
        iBase = pc.globalIndexOffset + atomicAdd(iCount.indexCounter, shPrimCount * 3u);
    }
    vBase = subgroupBroadcastFirst(vBase);
    iBase = subgroupBroadcastFirst(iBase);

    // Copy vertices to global buffer
    for (uint v = gl_LocalInvocationIndex; v < shVertCount; v += WORKGROUP_SIZE) {
        vertices.data[vBase + v] = shVerts[v];
    }

    // Copy indices to global buffer (with offset)
    for (uint k = gl_LocalInvocationIndex; k < shPrimCount * 3u; k += WORKGROUP_SIZE) {
        indices.data[iBase + k] = shIdx[k] + vBase;
    }

    // Write meshlet descriptor
    if (gl_LocalInvocationIndex == 0) {
        uint meshletIndex = pc.globalMeshletOffset + atomicAdd(meshletCount.meshletCounter, 1);
        meshlets.descriptors[meshletIndex] = MeshletDescriptor(vBase, iBase, shVertCount, shPrimCount);
        
        // This is an extraction-only shader, so we output zero geometry.
        SetMeshOutputsEXT(0u, 0u);
    }
}