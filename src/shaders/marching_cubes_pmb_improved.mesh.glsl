#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_ballot: require

// --- Debug Configuration (should match task shader) ---
#define DEBUG_ENABLED 0
#define DEBUG_BOUNDS_CHECK 0
#define DEBUG_MEMORY_USAGE 0
#define DEBUG_OVERFLOW_DETECTION 0
#define DEBUG_VERTEX_VALIDATION 0
#define DEBUG_PRIMITIVE_VALIDATION 0

// --- Compile-time Maximum Constants (must match task shader) ---
#define MAX_BLOCK_DIM 3u
#define MAX_CELLS_IN_BLOCK (MAX_BLOCK_DIM * MAX_BLOCK_DIM * MAX_BLOCK_DIM)
#define MAX_BLOCK_CONTEXT_DIM (MAX_BLOCK_DIM + 1u)

// --- Specialization Constants ---
layout(constant_id = 0) const uint BX = 3u;
layout(constant_id = 1) const uint BY = 3u;
layout(constant_id = 2) const uint BZ = 3u;

// --- Workgroup Configuration ---
#define WORKGROUP_SIZE 64u

// --- Algorithm Constants ---
const uint PMB_EDGE_X = 0u;
const uint PMB_EDGE_Y = 3u;
const uint PMB_EDGE_Z = 8u;

#define MAX_VERTS_PER_MESHLET 64u
#define MAX_PRIMS_PER_MESHLET 126u
#define MAX_MESHLETS_PER_BLOCK 4u
#define MAX_PRIMS_PER_CELL 5u

// Maximum primitives any thread might generate
const uint MAX_CELLS_PER_THREAD = (MAX_CELLS_IN_BLOCK + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
const uint MAX_PRIMS_PER_THREAD = MAX_CELLS_PER_THREAD * MAX_PRIMS_PER_CELL;

// --- Edge ownership mapping ---
const uvec4 edgeOwner[12] = uvec4[12](
    uvec4(0,0,0,0), uvec4(1,0,0,1), uvec4(0,1,0,0), uvec4(0,0,0,1),
    uvec4(0,0,1,0), uvec4(1,0,1,1), uvec4(0,1,1,0), uvec4(0,0,1,1),
    uvec4(0,0,0,2), uvec4(1,0,0,2), uvec4(1,1,0,2), uvec4(0,1,0,2)
);

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

// --- Task Payload ---
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
    uint meshletCount;
    uint firstCell[MAX_MESHLETS_PER_BLOCK];
    uint cellCount[MAX_MESHLETS_PER_BLOCK];
    uint packedCellData[MAX_CELLS_IN_BLOCK];
    // Debug fields always present but only used when debugging
    uint debugInfo[4];  // Store various debug counters
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

// --- Helper Constants ---
const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1)
);

// --- Shared Memory ---
shared VertexData shVerts[MAX_VERTS_PER_MESHLET];
shared uint shVertCount;
// Vertex map sized to maximum possible context region
shared uint shVertMap[MAX_BLOCK_CONTEXT_DIM][MAX_BLOCK_CONTEXT_DIM][MAX_BLOCK_CONTEXT_DIM][3];
shared uint shPrimCount;
shared uint shIdx[MAX_PRIMS_PER_MESHLET * 3];

// Temporary storage for parallel primitive generation
shared uvec3 sh_temp_tris[WORKGROUP_SIZE * MAX_PRIMS_PER_THREAD];
shared uint sh_vert_subgroup_sums[WORKGROUP_SIZE];
shared uint sh_prim_subgroup_sums[WORKGROUP_SIZE];

#if DEBUG_OVERFLOW_DETECTION
    shared uint sh_max_vert_index;
    shared uint sh_max_prim_index;
    shared uint sh_vert_overflow_count;
    shared uint sh_prim_overflow_count;
    shared uint sh_invalid_vertex_refs;
#endif

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

// Check if a cell owns a particular edge type
bool ownsX(uvec3 c) { return c.x < BX; }
bool ownsY(uvec3 c) { return c.y < BY; }
bool ownsZ(uvec3 c) { return c.z < BZ; }

// --- Workgroup Configuration ---
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Output Configuration ---
layout(max_vertices = MAX_VERTS_PER_MESHLET, max_primitives = MAX_PRIMS_PER_MESHLET) out;
layout(triangles) out;

void main()
{
    uint meshlet_idx_in_block = gl_WorkGroupID.x;
    if (meshlet_idx_in_block >= taskPayloadIn.meshletCount) return;

    // Calculate runtime dimensions
    const uint ACTUAL_BLOCK_DIM_X = BX + 1u;
    const uint ACTUAL_BLOCK_DIM_Y = BY + 1u;
    const uint ACTUAL_BLOCK_DIM_Z = BZ + 1u;
    const uint ACTUAL_CONTEXT_CELLS = ACTUAL_BLOCK_DIM_X * ACTUAL_BLOCK_DIM_Y * ACTUAL_BLOCK_DIM_Z;
    const uint ACTUAL_STRIDE = BX;

    // --- Initialize Shared Memory ---
    if (gl_LocalInvocationIndex == 0) {
        shVertCount = 0u;
        shPrimCount = 0u;
        
        #if DEBUG_OVERFLOW_DETECTION
            sh_max_vert_index = 0;
            sh_max_prim_index = 0;
            sh_vert_overflow_count = 0;
            sh_prim_overflow_count = 0;
            sh_invalid_vertex_refs = 0;
        #endif
        
        #if DEBUG_MEMORY_USAGE
            debugPrintfEXT("Mesh shader processing meshlet %u of block %u",
                          meshlet_idx_in_block, taskPayloadIn.blockID);
        #endif
    }

    // Initialize vertex map with sentinel values
    for (uint i = gl_LocalInvocationIndex; i < MAX_BLOCK_CONTEXT_DIM * MAX_BLOCK_CONTEXT_DIM * MAX_BLOCK_CONTEXT_DIM; i += WORKGROUP_SIZE) {
        uint x = i % MAX_BLOCK_CONTEXT_DIM;
        uint y = (i / MAX_BLOCK_CONTEXT_DIM) % MAX_BLOCK_CONTEXT_DIM;
        uint z = i / (MAX_BLOCK_CONTEXT_DIM * MAX_BLOCK_CONTEXT_DIM);
        
        // Only initialize cells within actual context bounds
        if (x < ACTUAL_BLOCK_DIM_X && y < ACTUAL_BLOCK_DIM_Y && z < ACTUAL_BLOCK_DIM_Z) {
            shVertMap[x][y][z][0] = 0xFFFFFFFFu;
            shVertMap[x][y][z][1] = 0xFFFFFFFFu;
            shVertMap[x][y][z][2] = 0xFFFFFFFFu;
        }
    }
    barrier();

    // --- PASS 1: Vertex Generation ---
    uvec3 blk_coord = unpack_block_id(taskPayloadIn.blockID);
    ivec3 base_coord = ivec3(blk_coord) * int(ACTUAL_STRIDE);

    // PASS 1A: Count vertices
    uint local_vert_count = 0;
    for (uint cell_idx = gl_LocalInvocationIndex; cell_idx < ACTUAL_CONTEXT_CELLS; cell_idx += WORKGROUP_SIZE) {
        uvec3 cLoc = uvec3(
            cell_idx % ACTUAL_BLOCK_DIM_X,
            (cell_idx / ACTUAL_BLOCK_DIM_X) % ACTUAL_BLOCK_DIM_Y,
            cell_idx / (ACTUAL_BLOCK_DIM_X * ACTUAL_BLOCK_DIM_Y)
        );
        
        #if DEBUG_BOUNDS_CHECK
            if (cLoc.x >= ACTUAL_BLOCK_DIM_X || cLoc.y >= ACTUAL_BLOCK_DIM_Y || cLoc.z >= ACTUAL_BLOCK_DIM_Z) {
                debugPrintfEXT("ERROR: Cell index %u produced invalid coordinates (%u,%u,%u)",
                              cell_idx, cLoc.x, cLoc.y, cLoc.z);
                continue;
            }
        #endif
        
        ivec3 gLoc = base_coord + ivec3(cLoc);
        if (any(greaterThanEqual(gLoc + ivec3(1), ivec3(ubo.volumeDim.xyz)))) continue;
        
        uint eMask = mcEdgeTable.edgeTable[calculate_configuration(gLoc)];
        if (eMask == 0u) continue;
        
        if ((eMask & (1u << PMB_EDGE_X)) != 0u && ownsX(cLoc)) local_vert_count++;
        if ((eMask & (1u << PMB_EDGE_Y)) != 0u && ownsY(cLoc)) local_vert_count++;
        if ((eMask & (1u << PMB_EDGE_Z)) != 0u && ownsZ(cLoc)) local_vert_count++;
    }
    barrier();

    // PASS 1B: Parallel scan for vertex offsets
    uint subgroup_vert_offset = subgroupExclusiveAdd(local_vert_count);
    uint subgroup_vert_total = subgroupAdd(local_vert_count);
    
    if (subgroupElect()) {
        sh_vert_subgroup_sums[gl_SubgroupID] = subgroup_vert_total;
    }
    barrier();
    
    if (gl_SubgroupID == 0) {
        uint subgroup_sum_val = (gl_SubgroupInvocationID < gl_NumSubgroups) ? 
                                sh_vert_subgroup_sums[gl_SubgroupInvocationID] : 0;
        uint subgroup_base_offset = subgroupExclusiveAdd(subgroup_sum_val);
        if (gl_SubgroupInvocationID < gl_NumSubgroups) {
            sh_vert_subgroup_sums[gl_SubgroupInvocationID] = subgroup_base_offset;
        }
    }
    barrier();
    
    uint final_vert_offset = sh_vert_subgroup_sums[gl_SubgroupID] + subgroup_vert_offset;
    if (gl_LocalInvocationIndex == WORKGROUP_SIZE - 1) {
        shVertCount = final_vert_offset + local_vert_count;
        
        #if DEBUG_OVERFLOW_DETECTION
            if (shVertCount > MAX_VERTS_PER_MESHLET) {
                debugPrintfEXT("WARNING: Vertex count %u exceeds maximum %u",
                              shVertCount, MAX_VERTS_PER_MESHLET);
            }
        #endif
    }
    barrier();

    // PASS 1C: Generate vertices
    uint running_vert_offset = 0;
    for (uint cell_idx = gl_LocalInvocationIndex; cell_idx < ACTUAL_CONTEXT_CELLS; cell_idx += WORKGROUP_SIZE) {
        uvec3 cLoc = uvec3(
            cell_idx % ACTUAL_BLOCK_DIM_X,
            (cell_idx / ACTUAL_BLOCK_DIM_X) % ACTUAL_BLOCK_DIM_Y,
            cell_idx / (ACTUAL_BLOCK_DIM_X * ACTUAL_BLOCK_DIM_Y)
        );
        
        ivec3 gLoc = base_coord + ivec3(cLoc);
        if (any(greaterThanEqual(gLoc + ivec3(1), ivec3(ubo.volumeDim.xyz)))) continue;
        
        uint cfg = calculate_configuration(gLoc);
        uint eMask = mcEdgeTable.edgeTable[cfg];
        if (eMask == 0u) continue;
        
        // Generate X edge vertex
        if ((eMask & (1u << PMB_EDGE_X)) != 0u && ownsX(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex(ubo.isovalue, gLoc, gLoc + ivec3(1,0,0));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][0] = write_idx;
                
                #if DEBUG_VERTEX_VALIDATION
                    if (any(isnan(shVerts[write_idx].position.xyz)) || 
                        any(isinf(shVerts[write_idx].position.xyz))) {
                        debugPrintfEXT("Invalid vertex generated at index %u", write_idx);
                    }
                #endif
            } else {
                #if DEBUG_OVERFLOW_DETECTION
                    atomicAdd(sh_vert_overflow_count, 1);
                #endif
            }
            running_vert_offset++;
        }
        
        // Generate Y edge vertex
        if ((eMask & (1u << PMB_EDGE_Y)) != 0u && ownsY(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex(ubo.isovalue, gLoc, gLoc + ivec3(0,1,0));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][1] = write_idx;
            } else {
                #if DEBUG_OVERFLOW_DETECTION
                    atomicAdd(sh_vert_overflow_count, 1);
                #endif
            }
            running_vert_offset++;
        }
        
        // Generate Z edge vertex
        if ((eMask & (1u << PMB_EDGE_Z)) != 0u && ownsZ(cLoc)) {
            uint write_idx = final_vert_offset + running_vert_offset;
            
            if (write_idx < MAX_VERTS_PER_MESHLET) {
                shVerts[write_idx] = interpolate_vertex(ubo.isovalue, gLoc, gLoc + ivec3(0,0,1));
                shVertMap[cLoc.x][cLoc.y][cLoc.z][2] = write_idx;
            } else {
                #if DEBUG_OVERFLOW_DETECTION
                    atomicAdd(sh_vert_overflow_count, 1);
                #endif
            }
            running_vert_offset++;
        }
        
        #if DEBUG_OVERFLOW_DETECTION
            atomicMax(sh_max_vert_index, final_vert_offset + running_vert_offset);
        #endif
    }
    barrier();

    // --- PASS 2: Triangle Generation ---
    uint first_cell_idx = taskPayloadIn.firstCell[meshlet_idx_in_block];
    uint num_cells = taskPayloadIn.cellCount[meshlet_idx_in_block];
    uint local_prim_count = 0;

    #if DEBUG_PRIMITIVE_VALIDATION
        if (gl_LocalInvocationIndex == 0) {
            debugPrintfEXT("Generating triangles for %u cells starting at %u",
                          num_cells, first_cell_idx);
        }
    #endif

    for (uint i = gl_LocalInvocationIndex; i < num_cells; i += WORKGROUP_SIZE) {
        uint packed_data = taskPayloadIn.packedCellData[first_cell_idx + i];
        uint cellID = (packed_data >> 16) & 0xFFFFu;
        uint prims = (packed_data >> 8) & 0xFFu;
        
        uvec3 c_core = uvec3(cellID % BX, (cellID / BX) % BY, cellID / (BX * BY));
        ivec3 g_core = base_coord + ivec3(c_core);
        uint cfg = calculate_configuration(g_core);
        
        for (uint t = 0; t < prims; ++t) {
            if (local_prim_count >= MAX_PRIMS_PER_THREAD) {
                #if DEBUG_OVERFLOW_DETECTION
                    atomicAdd(sh_prim_overflow_count, 1);
                #endif
                continue;
            }
            
            uvec3 tri_indices;
            int e0 = mcTriangleTable.triTable[cfg * 16 + t * 3 + 0];
            int e1 = mcTriangleTable.triTable[cfg * 16 + t * 3 + 1];
            int e2 = mcTriangleTable.triTable[cfg * 16 + t * 3 + 2];
            
            // Get vertex indices from owner cells
            uvec4 owner0 = edgeOwner[e0];
            uvec4 owner1 = edgeOwner[e1];
            uvec4 owner2 = edgeOwner[e2];
            
            uvec3 lookup0 = c_core + owner0.xyz;
            uvec3 lookup1 = c_core + owner1.xyz;
            uvec3 lookup2 = c_core + owner2.xyz;
            
            #if DEBUG_BOUNDS_CHECK
                bool valid = true;
                if (lookup0.x >= ACTUAL_BLOCK_DIM_X || lookup0.y >= ACTUAL_BLOCK_DIM_Y || 
                    lookup0.z >= ACTUAL_BLOCK_DIM_Z) {
                    debugPrintfEXT("Triangle vertex 0 lookup out of bounds: (%u,%u,%u)",
                                  lookup0.x, lookup0.y, lookup0.z);
                    valid = false;
                }
                if (lookup1.x >= ACTUAL_BLOCK_DIM_X || lookup1.y >= ACTUAL_BLOCK_DIM_Y || 
                    lookup1.z >= ACTUAL_BLOCK_DIM_Z) {
                    debugPrintfEXT("Triangle vertex 1 lookup out of bounds: (%u,%u,%u)",
                                  lookup1.x, lookup1.y, lookup1.z);
                    valid = false;
                }
                if (lookup2.x >= ACTUAL_BLOCK_DIM_X || lookup2.y >= ACTUAL_BLOCK_DIM_Y || 
                    lookup2.z >= ACTUAL_BLOCK_DIM_Z) {
                    debugPrintfEXT("Triangle vertex 2 lookup out of bounds: (%u,%u,%u)",
                                  lookup2.x, lookup2.y, lookup2.z);
                    valid = false;
                }
                if (!valid) continue;
            #endif
            
            tri_indices.x = shVertMap[lookup0.x][lookup0.y][lookup0.z][owner0.w];
            tri_indices.y = shVertMap[lookup1.x][lookup1.y][lookup1.z][owner1.w];
            tri_indices.z = shVertMap[lookup2.x][lookup2.y][lookup2.z][owner2.w];
            
            // Validate triangle
            if (tri_indices.x == 0xFFFFFFFFu || tri_indices.y == 0xFFFFFFFFu || 
                tri_indices.z == 0xFFFFFFFFu) {
                #if DEBUG_OVERFLOW_DETECTION
                    atomicAdd(sh_invalid_vertex_refs, 1);
                #endif
                continue;
            }
            
            #if DEBUG_BOUNDS_CHECK
                if (tri_indices.x >= shVertCount || tri_indices.y >= shVertCount || 
                    tri_indices.z >= shVertCount) {
                    debugPrintfEXT("Triangle references invalid vertices: (%u,%u,%u) max=%u",
                                  tri_indices.x, tri_indices.y, tri_indices.z, shVertCount);
                    continue;
                }
            #endif
            
            sh_temp_tris[gl_LocalInvocationIndex * MAX_PRIMS_PER_THREAD + local_prim_count] = tri_indices;
            local_prim_count++;
        }
    }
    barrier();

    // PASS 2B: Scan for primitive offsets
    uint subgroup_prim_offset = subgroupExclusiveAdd(local_prim_count);
    uint subgroup_prim_total = subgroupAdd(local_prim_count);
    
    if (subgroupElect()) {
        sh_prim_subgroup_sums[gl_SubgroupID] = subgroup_prim_total;
    }
    barrier();
    
    if (gl_SubgroupID == 0) {
        uint subgroup_sum_val = (gl_SubgroupInvocationID < gl_NumSubgroups) ? 
                                sh_prim_subgroup_sums[gl_SubgroupInvocationID] : 0;
        uint subgroup_base_offset = subgroupExclusiveAdd(subgroup_sum_val);
        if (gl_SubgroupInvocationID < gl_NumSubgroups) {
            sh_prim_subgroup_sums[gl_SubgroupInvocationID] = subgroup_base_offset;
        }
    }
    barrier();
    
    uint final_prim_offset = sh_prim_subgroup_sums[gl_SubgroupID] + subgroup_prim_offset;
    if (gl_LocalInvocationIndex == WORKGROUP_SIZE - 1) {
        shPrimCount = final_prim_offset + local_prim_count;
        
        #if DEBUG_OVERFLOW_DETECTION
            if (shPrimCount > MAX_PRIMS_PER_MESHLET) {
                debugPrintfEXT("WARNING: Primitive count %u exceeds maximum %u",
                              shPrimCount, MAX_PRIMS_PER_MESHLET);
            }
            debugPrintfEXT("Meshlet stats: %u vertices, %u primitives",
                          shVertCount, shPrimCount);
            debugPrintfEXT("  Vertex overflows: %u, Prim overflows: %u, Invalid refs: %u",
                          sh_vert_overflow_count, sh_prim_overflow_count, sh_invalid_vertex_refs);
        #endif
    }
    barrier();

    // PASS 2C: Write triangles
    for (uint i = 0; i < local_prim_count; ++i) {
        uint write_idx = final_prim_offset + i;
        if (write_idx < MAX_PRIMS_PER_MESHLET) {
            uvec3 tri = sh_temp_tris[gl_LocalInvocationIndex * MAX_PRIMS_PER_THREAD + i];
            shIdx[write_idx * 3 + 0] = tri.x;
            shIdx[write_idx * 3 + 1] = tri.y;
            shIdx[write_idx * 3 + 2] = tri.z;
            
            #if DEBUG_OVERFLOW_DETECTION
                atomicMax(sh_max_prim_index, write_idx);
            #endif
        }
    }
    barrier();

    // --- PASS 3: Write to global buffers ---
    uint vBase, iBase;
    if (gl_LocalInvocationIndex == 0) {
        uint actualVertCount = min(shVertCount, MAX_VERTS_PER_MESHLET);
        uint actualPrimCount = min(shPrimCount, MAX_PRIMS_PER_MESHLET);
        
        vBase = atomicAdd(vCount.vertexCounter, actualVertCount);
        iBase = atomicAdd(iCount.indexCounter, actualPrimCount * 3u);
        
        #if DEBUG_MEMORY_USAGE
            debugPrintfEXT("Allocated global storage: verts[%u-%u], indices[%u-%u]",
                          vBase, vBase + actualVertCount - 1,
                          iBase, iBase + actualPrimCount * 3 - 1);
        #endif
    }
    vBase = subgroupBroadcastFirst(vBase);
    iBase = subgroupBroadcastFirst(iBase);

    // Copy vertices
    uint actualVertCount = min(shVertCount, MAX_VERTS_PER_MESHLET);
    for (uint v = gl_LocalInvocationIndex; v < actualVertCount; v += WORKGROUP_SIZE) {
        vertices.data[vBase + v] = shVerts[v];
    }

    // Copy indices with offset
    uint actualPrimCount = min(shPrimCount, MAX_PRIMS_PER_MESHLET);
    for (uint k = gl_LocalInvocationIndex; k < actualPrimCount * 3u; k += WORKGROUP_SIZE) {
        indices.data[iBase + k] = shIdx[k] + vBase;
    }

    // Write meshlet descriptor
    if (gl_LocalInvocationIndex == 0) {
        uint desc_id = atomicAdd(meshletCount.meshletCounter, 1u);
        meshlets.descriptors[desc_id] = MeshletDescriptor(
            vBase, iBase, actualVertCount, actualPrimCount
        );
        
        #if DEBUG_MEMORY_USAGE
            debugPrintfEXT("Created meshlet descriptor %u", desc_id);
        #endif
        
        // This is an extraction-only shader
        SetMeshOutputsEXT(0u, 0u);
    }
}
