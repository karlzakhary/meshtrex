#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
#define MESH_SHADER_MAX_VERTICES 256u
#define MESH_SHADER_MAX_PRIMITIVES 256u
#define MESH_SHADER_WORKGROUP_SIZE 32

#define MAX_VERTICES_PER_CELL_OUTPUT 15
#define MAX_PRIMS_PER_CELL_OUTPUT 5
#define MAX_TRI_INDICES_PER_CELL (MAX_PRIMS_PER_CELL_OUTPUT * 3)

// From Task Shader
#define MACRO_BLOCK_DIM_X 8 
#define MACRO_BLOCK_DIM_Y 8
#define MACRO_BLOCK_DIM_Z 8
#define MACRO_BLOCK_CELL_COUNT (MACRO_BLOCK_DIM_X * MACRO_BLOCK_DIM_Y * MACRO_BLOCK_DIM_Z)
#define MAX_BATCHES_PER_MACRO_BLOCK (MACRO_BLOCK_CELL_COUNT / 2 + 1)

// --- Structures ---
struct TaskPayload {
    uint macroBlockGlobalVertexBase;
    uint macroBlockGlobalIndexBase;
    uint macroBlockGlobalMeshletDescBase;
    uvec3 macroBlockOrigin_global;
    uvec3 macroBlockDimensions;
    uint numTotalOccupiedCellsInMacroBlock;
    uint denseOccupiedCellIndices[MACRO_BLOCK_CELL_COUNT];
    uint numBatchesToLaunch;
    uvec2 batchCellListInfo[MAX_BATCHES_PER_MACRO_BLOCK];
    uint batchVertexSubOffsets[MAX_BATCHES_PER_MACRO_BLOCK];
    uint batchIndexSubOffsets[MAX_BATCHES_PER_MACRO_BLOCK];
    uint estimatedVerticesPerBatch[MAX_BATCHES_PER_MACRO_BLOCK];
    uint estimatedPrimitivesPerBatch[MAX_BATCHES_PER_MACRO_BLOCK];
    uint originalCompactedBlockID;
};

taskPayloadSharedEXT TaskPayload taskPayloadIn;

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

// --- Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;

// Output SSBOs
layout(set = 0, binding = 6, std430) buffer Vertex_SSBO { VertexData vertex_data[]; } vertices_ssbo;
layout(set = 0, binding = 8, std430) buffer Index_SSBO { uint indices[]; } indices_ssbo;
layout(set = 0, binding = 10, std430) buffer MeshletDescOutput_SSBO { MeshletDescriptor meshletDescriptors[]; } meshletDescOutput;
layout(set = 0, binding = 11, std430) buffer VertexCount_SSBO { uint vertexCounter; } vCount;
layout(set = 0, binding = 13, std430) buffer IndexCount_SSBO { uint indexCounter; } iCount;
layout(set = 0, binding = 12, std430) buffer MeshletDescCount_SSBO { uint meshletCounter; } meshletDescCount;
layout(set = 0, binding = 12, std430) buffer FilledMeshletDescCount_SSBO { uint filledMeshletCounter; } filledMeshletDescCount;

// --- Mesh Shader Output ---
layout(local_size_x = MESH_SHADER_WORKGROUP_SIZE) in;
layout(max_vertices = MESH_SHADER_MAX_VERTICES, max_primitives = MESH_SHADER_MAX_PRIMITIVES) out;
layout(triangles) out;

layout(location = 0) out PerVertexData { vec3 normal; } meshVertexDataOut[];

// --- Shared Memory ---
shared uint shared_thread_vertex_counts[MESH_SHADER_WORKGROUP_SIZE];
shared uint shared_thread_initial_primitive_counts[MESH_SHADER_WORKGROUP_SIZE];
shared uint shared_thread_final_primitive_counts[MESH_SHADER_WORKGROUP_SIZE];

shared uint shared_vertex_write_offsets[MESH_SHADER_WORKGROUP_SIZE];
shared uint shared_primitive_write_offsets[MESH_SHADER_WORKGROUP_SIZE];

shared uint shared_batch_total_uncapped_vertices;
shared uint shared_batch_capped_vertex_count;
shared uint shared_batch_capped_primitive_count;

// For independent allocation
shared uint shared_batch_global_vertex_base;
shared uint shared_batch_global_index_base;
shared uint shared_batch_global_meshlet_slot;

// --- Helper Functions ---
void getFullEdgeInfo(int edgeIndex, out ivec3 p1_offset, out ivec3 p2_offset, out int corner1_idx, out int corner2_idx) {
    switch (edgeIndex) {
        case 0:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(1,0,0); corner1_idx=0; corner2_idx=1; return;
        case 1:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,1,0); corner1_idx=1; corner2_idx=2; return;
        case 2:  p1_offset=ivec3(0,1,0); p2_offset=ivec3(1,1,0); corner1_idx=3; corner2_idx=2; return;
        case 3:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,1,0); corner1_idx=0; corner2_idx=3; return;
        case 4:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(1,0,1); corner1_idx=4; corner2_idx=5; return;
        case 5:  p1_offset=ivec3(1,0,1); p2_offset=ivec3(1,1,1); corner1_idx=5; corner2_idx=6; return;
        case 6:  p1_offset=ivec3(0,1,1); p2_offset=ivec3(1,1,1); corner1_idx=7; corner2_idx=6; return;
        case 7:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(0,1,1); corner1_idx=4; corner2_idx=7; return;
        case 8:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,0,1); corner1_idx=0; corner2_idx=4; return;
        case 9:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,0,1); corner1_idx=1; corner2_idx=5; return;
        case 10: p1_offset=ivec3(1,1,0); p2_offset=ivec3(1,1,1); corner1_idx=2; corner2_idx=6; return;
        case 11: p1_offset=ivec3(0,1,0); p2_offset=ivec3(0,1,1); corner1_idx=3; corner2_idx=7; return;
        default: p1_offset=ivec3(0); p2_offset=ivec3(0); corner1_idx=-1; corner2_idx=-1; return;
    }
}

// CONSISTENT VERTEX GENERATION: Key to eliminating seams
// This function ensures that adjacent blocks generate identical vertices for shared edges
vec3 interpolateVertexConsistent(ivec3 p1_global_voxel_coord, ivec3 p2_global_voxel_coord, float val1, float val2, float isovalue_in) {
    // CRITICAL: Always use the same precision and order for shared edges
    // Ensure p1 is always the "smaller" coordinate to maintain consistency across blocks
    if (any(greaterThan(p1_global_voxel_coord, p2_global_voxel_coord))) {
        // Swap to ensure consistent ordering
        ivec3 temp = p1_global_voxel_coord;
        p1_global_voxel_coord = p2_global_voxel_coord;
        p2_global_voxel_coord = temp;
        float tempVal = val1;
        val1 = val2;
        val2 = tempVal;
    }
    
    // Standard interpolation with consistent precision
    if (abs(isovalue_in - val1) < 1e-6f) return vec3(p1_global_voxel_coord);
    if (abs(isovalue_in - val2) < 1e-6f) return vec3(p2_global_voxel_coord);
    if (abs(val1 - val2) < 1e-6f) return vec3(p1_global_voxel_coord);
    
    // Use high precision for consistent results across blocks
    double t = double(isovalue_in - val1) / double(val2 - val1);
    return vec3(mix(dvec3(p1_global_voxel_coord), dvec3(p2_global_voxel_coord), t));
}

vec3 calculateNormal(vec3 vertexPos_global_coords, float isovalue_in, uvec3 volDim_in) {
    float delta = 0.5f; 
    ivec3 volMax = ivec3(volDim_in) - 1;
    float s_xp1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords + vec3(delta, 0,0)), ivec3(0), volMax)).r);
    float s_xm1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords - vec3(delta, 0,0)), ivec3(0), volMax)).r);
    float s_yp1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords + vec3(0, delta,0)), ivec3(0), volMax)).r);
    float s_ym1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords - vec3(0, delta,0)), ivec3(0), volMax)).r);
    float s_zp1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords + vec3(0,0, delta)), ivec3(0), volMax)).r);
    float s_zm1 = float(imageLoad(volumeImage, clamp(ivec3(vertexPos_global_coords - vec3(0,0, delta)), ivec3(0), volMax)).r);
    vec3 grad = vec3(s_xp1 - s_xm1, s_yp1 - s_ym1, s_zp1 - s_zm1);
    if (length(grad) < 1e-5f) return vec3(0, 1, 0);
    return -normalize(grad);
}

// SIMPLIFIED OWNERSHIP: Only skip cells that are clearly outside the volume or block
bool shouldProcessCell(uvec3 cellOrigin, uvec3 blockOrigin, uvec3 blockDimensions, uvec3 volumeDimensions) {
    // Basic bounds check - only skip cells that are clearly invalid
    if (any(greaterThanEqual(cellOrigin, volumeDimensions))) {
        return false; // Cell starts outside volume
    }
    
    // Allow processing of boundary cells - let duplicate detection happen at vertex level
    // Only skip cells that are completely outside this block's responsibility
    uvec3 localCellCoord = cellOrigin - blockOrigin;
    if (any(greaterThanEqual(localCellCoord, blockDimensions + uvec3(1)))) {
        return false; // Cell is too far outside block
    }
    
    return true; // Process this cell
}

// Fixed prefix scan implementation
void workgroupExclusivePrefix() {
    // Phase 1: Inclusive scan
    uint offset = 1;
    for (uint d = MESH_SHADER_WORKGROUP_SIZE >> 1; d > 0; d >>= 1) {
        barrier();
        if (gl_LocalInvocationIndex < d) {
            uint ai = offset * (2 * gl_LocalInvocationIndex + 1) - 1;
            uint bi = offset * (2 * gl_LocalInvocationIndex + 2) - 1;
            shared_thread_vertex_counts[bi] += shared_thread_vertex_counts[ai];
        }
        offset <<= 1;
    }
    
    // Store total sum
    barrier();
    if (gl_LocalInvocationIndex == 0) {
        shared_batch_total_uncapped_vertices = shared_thread_vertex_counts[MESH_SHADER_WORKGROUP_SIZE - 1];
        shared_thread_vertex_counts[MESH_SHADER_WORKGROUP_SIZE - 1] = 0;
    }
    
    // Phase 2: Convert to exclusive scan
    for (uint d = 1; d < MESH_SHADER_WORKGROUP_SIZE; d <<= 1) {
        offset >>= 1;
        barrier();
        if (gl_LocalInvocationIndex < d) {
            uint ai = offset * (2 * gl_LocalInvocationIndex + 1) - 1;
            uint bi = offset * (2 * gl_LocalInvocationIndex + 2) - 1;
            uint temp = shared_thread_vertex_counts[ai];
            shared_thread_vertex_counts[ai] = shared_thread_vertex_counts[bi];
            shared_thread_vertex_counts[bi] += temp;
        }
    }
    barrier();
    
    // Copy results to offset array
    shared_vertex_write_offsets[gl_LocalInvocationIndex] = shared_thread_vertex_counts[gl_LocalInvocationIndex];
}

void main() {
    uint lid = gl_LocalInvocationIndex;
    uint batchID = gl_WorkGroupID.x;

    // Initialize shared memory
    if (lid == 0) {
        shared_batch_capped_vertex_count = 0;
        shared_batch_capped_primitive_count = 0;
        shared_batch_total_uncapped_vertices = 0;
        shared_batch_global_vertex_base = 0;
        shared_batch_global_index_base = 0;
        shared_batch_global_meshlet_slot = 0;
    }
    
    shared_thread_vertex_counts[lid] = 0;
    shared_thread_initial_primitive_counts[lid] = 0;
    shared_thread_final_primitive_counts[lid] = 0;
    barrier();

    if (batchID >= taskPayloadIn.numBatchesToLaunch) {
        SetMeshOutputsEXT(0, 0);
        return;
    }

    uint cellListOffset = taskPayloadIn.batchCellListInfo[batchID].x;
    uint numCellsInThisBatch = taskPayloadIn.batchCellListInfo[batchID].y;

    // Per-thread local storage
    VertexData thread_local_vertices[MAX_VERTICES_PER_CELL_OUTPUT];
    uint       thread_local_cell_indices[MAX_TRI_INDICES_PER_CELL];
    uvec3      thread_final_batch_triangles[MAX_PRIMS_PER_CELL_OUTPUT];

    uint num_verts_gen_this_thread = 0;
    uint num_prims_gen_this_thread_initial = 0;
    uint num_prims_final_this_thread = 0;

    // PHASE 1: Process assigned cells with LIBERAL ownership (allow boundary overlaps)
    if (lid < numCellsInThisBatch) {
        uint oneDCellIdx = taskPayloadIn.denseOccupiedCellIndices[cellListOffset + lid];
        
        uvec3 localCellCoordsInMacroBlock;
        localCellCoordsInMacroBlock.x = oneDCellIdx % taskPayloadIn.macroBlockDimensions.x;
        localCellCoordsInMacroBlock.y = (oneDCellIdx / taskPayloadIn.macroBlockDimensions.x) % taskPayloadIn.macroBlockDimensions.y;
        localCellCoordsInMacroBlock.z = oneDCellIdx / (taskPayloadIn.macroBlockDimensions.x * taskPayloadIn.macroBlockDimensions.y);
        
        uvec3 globalCellOrigin = taskPayloadIn.macroBlockOrigin_global + localCellCoordsInMacroBlock;

        // SIMPLIFIED CHECK: Only skip clearly invalid cells
        bool shouldProcess = shouldProcessCell(
            globalCellOrigin,
            taskPayloadIn.macroBlockOrigin_global,
            taskPayloadIn.macroBlockDimensions,
            ubo.volumeDim.xyz
        );

        if (shouldProcess) {
            // Sample corner values and determine cube case
            float cornerValuesF[8];
            uint cubeCase = 0;
            for (int c = 0; c < 8; ++c) {
                ivec3 cornerOffset = ivec3((c & 1), (c & 2) >> 1, (c & 4) >> 2);
                ivec3 cornerVolCoord = ivec3(globalCellOrigin) + cornerOffset;
                uint valU = 0;
                if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                    valU = imageLoad(volumeImage, cornerVolCoord).r;
                }
                cornerValuesF[c] = float(valU);
                if (cornerValuesF[c] >= ubo.isovalue) {
                    cubeCase |= (1 << c);
                }
            }

            // Generate geometry using marching cubes with CONSISTENT vertex generation
            if (cubeCase != 0 && cubeCase != 255) {
                uint cell_vtx_map[12];
                for(int k = 0; k < 12; ++k) {
                    cell_vtx_map[k] = 0xFFFFFFFFu;
                }
                
                int triTableBase = int(cubeCase * 16);

                for (int i = 0; i < 15; ++i) {
                    int mcEdgeID = mc.triTable[triTableBase + i];
                    if (mcEdgeID == -1) break;
                    if (mcEdgeID < 0 || mcEdgeID > 11) continue;

                    // Create vertex for this edge if not already created
                    if (cell_vtx_map[mcEdgeID] == 0xFFFFFFFFu) {
                        if (num_verts_gen_this_thread < MAX_VERTICES_PER_CELL_OUTPUT) {
                            ivec3 p1_offset, p2_offset;
                            int c1_idx, c2_idx;
                            getFullEdgeInfo(mcEdgeID, p1_offset, p2_offset, c1_idx, c2_idx);
                            
                            // USE CONSISTENT VERTEX GENERATION
                            vec3 vPos = interpolateVertexConsistent(
                                ivec3(globalCellOrigin) + p1_offset, 
                                ivec3(globalCellOrigin) + p2_offset, 
                                cornerValuesF[c1_idx], 
                                cornerValuesF[c2_idx], 
                                ubo.isovalue
                            );
                            
                            vec3 vNorm = calculateNormal(vPos, ubo.isovalue, ubo.volumeDim.xyz);
                            
                            thread_local_vertices[num_verts_gen_this_thread].position = vec4(vPos, 1.0);
                            thread_local_vertices[num_verts_gen_this_thread].normal = vec4(vNorm, 0.0);
                            
                            cell_vtx_map[mcEdgeID] = num_verts_gen_this_thread;
                            num_verts_gen_this_thread++;
                        }
                    }

                    // Store triangle indices
                    if (num_prims_gen_this_thread_initial * 3 + (i % 3) < MAX_TRI_INDICES_PER_CELL) {
                        thread_local_cell_indices[num_prims_gen_this_thread_initial * 3 + (i % 3)] = cell_vtx_map[mcEdgeID];
                    }

                    // Complete triangle
                    if ((i + 1) % 3 == 0) {
                        if (num_prims_gen_this_thread_initial < MAX_PRIMS_PER_CELL_OUTPUT) {
                            uint i0 = thread_local_cell_indices[num_prims_gen_this_thread_initial * 3 + 0];
                            uint i1 = thread_local_cell_indices[num_prims_gen_this_thread_initial * 3 + 1];
                            uint i2 = thread_local_cell_indices[num_prims_gen_this_thread_initial * 3 + 2];
                            
                            // Validate triangle
                            if (i0 != 0xFFFFFFFFu && i1 != 0xFFFFFFFFu && i2 != 0xFFFFFFFFu &&
                                i0 < num_verts_gen_this_thread && i1 < num_verts_gen_this_thread && i2 < num_verts_gen_this_thread &&
                                i0 != i1 && i0 != i2 && i1 != i2) {
                                num_prims_gen_this_thread_initial++;
                            }
                        }
                    }
                }
            }
        }
    }

    // Store counts in shared memory
    shared_thread_vertex_counts[lid] = num_verts_gen_this_thread;
    shared_thread_initial_primitive_counts[lid] = num_prims_gen_this_thread_initial;
    barrier();

    // PHASE 2: Calculate vertex layout using fixed prefix scan
    workgroupExclusivePrefix();

    // PHASE 3: Determine capped vertex count
    uint batch_capped_vertex_count_val;
    if (lid == 0) {
        batch_capped_vertex_count_val = min(shared_batch_total_uncapped_vertices, MESH_SHADER_MAX_VERTICES);
        shared_batch_capped_vertex_count = batch_capped_vertex_count_val;
    }
    barrier();
    batch_capped_vertex_count_val = shared_batch_capped_vertex_count;

    // PHASE 4: Filter primitives based on vertex capping
    if (lid < numCellsInThisBatch && num_prims_gen_this_thread_initial > 0) {
        uint thread_vertex_base_in_batch = shared_vertex_write_offsets[lid];
        
        for (uint prim_idx = 0; prim_idx < num_prims_gen_this_thread_initial; ++prim_idx) {
            uint cl_idx0 = thread_local_cell_indices[prim_idx * 3 + 0];
            uint cl_idx1 = thread_local_cell_indices[prim_idx * 3 + 1];
            uint cl_idx2 = thread_local_cell_indices[prim_idx * 3 + 2];

            uint batch_vidx0 = thread_vertex_base_in_batch + cl_idx0;
            uint batch_vidx1 = thread_vertex_base_in_batch + cl_idx1;
            uint batch_vidx2 = thread_vertex_base_in_batch + cl_idx2;

            if (batch_vidx0 < batch_capped_vertex_count_val &&
                batch_vidx1 < batch_capped_vertex_count_val &&
                batch_vidx2 < batch_capped_vertex_count_val) {
                if (num_prims_final_this_thread < MAX_PRIMS_PER_CELL_OUTPUT) {
                    thread_final_batch_triangles[num_prims_final_this_thread] = uvec3(batch_vidx0, batch_vidx1, batch_vidx2);
                    num_prims_final_this_thread++;
                }
            }
        }
    }

    shared_thread_final_primitive_counts[lid] = num_prims_final_this_thread;
    barrier();

    // PHASE 5: Calculate final primitive count
    uint batch_capped_primitive_count_val = 0;
    if (lid == 0) {
        for (uint i = 0; i < MESH_SHADER_WORKGROUP_SIZE; ++i) {
            batch_capped_primitive_count_val += shared_thread_final_primitive_counts[i];
        }
        batch_capped_primitive_count_val = min(batch_capped_primitive_count_val, MESH_SHADER_MAX_PRIMITIVES);
        shared_batch_capped_primitive_count = batch_capped_primitive_count_val;
    }
    barrier();
    batch_capped_primitive_count_val = shared_batch_capped_primitive_count;

    // PHASE 6: Allocate global space (INDEPENDENT ALLOCATION)
    if (lid == 0 && batch_capped_vertex_count_val > 0 && batch_capped_primitive_count_val > 0) {
        shared_batch_global_vertex_base = atomicAdd(vCount.vertexCounter, batch_capped_vertex_count_val);
        shared_batch_global_index_base = atomicAdd(iCount.indexCounter, batch_capped_primitive_count_val * 3);
        shared_batch_global_meshlet_slot = atomicAdd(meshletDescCount.meshletCounter, 1);
    }
    barrier();

    // PHASE 7: Set mesh outputs
    if (lid == 0) {
        SetMeshOutputsEXT(batch_capped_vertex_count_val, batch_capped_primitive_count_val);
    }
    barrier();

    // PHASE 8: Write vertices
    uint thread_vertex_base_offset = shared_vertex_write_offsets[lid];
    for (uint i = 0; i < num_verts_gen_this_thread; ++i) {
        uint batch_vertex_idx = thread_vertex_base_offset + i;
        if (batch_vertex_idx < batch_capped_vertex_count_val) {
            VertexData v = thread_local_vertices[i];
            
            gl_MeshVerticesEXT[batch_vertex_idx].gl_Position = v.position;
            meshVertexDataOut[batch_vertex_idx].normal = v.normal.xyz;
            
            if (shared_batch_global_vertex_base + batch_vertex_idx < 10000000) {
                vertices_ssbo.vertex_data[shared_batch_global_vertex_base + batch_vertex_idx] = v;
            }
        }
    }

    // PHASE 9: Write primitives
    uint primitive_write_offset = 0;
    for (uint i = 0; i < lid; ++i) {
        primitive_write_offset += shared_thread_final_primitive_counts[i];
    }
    
    for (uint i = 0; i < num_prims_final_this_thread; ++i) {
        uint batch_primitive_idx = primitive_write_offset + i;
        if (batch_primitive_idx < batch_capped_primitive_count_val) {
            uvec3 batch_indices = thread_final_batch_triangles[i];
            
            gl_PrimitiveTriangleIndicesEXT[batch_primitive_idx] = batch_indices;
            
            uint global_idx_start = shared_batch_global_index_base + batch_primitive_idx * 3;
            if (global_idx_start + 2 < 30000000) {
                indices_ssbo.indices[global_idx_start + 0] = shared_batch_global_vertex_base + batch_indices.x;
                indices_ssbo.indices[global_idx_start + 1] = shared_batch_global_vertex_base + batch_indices.y;
                indices_ssbo.indices[global_idx_start + 2] = shared_batch_global_vertex_base + batch_indices.z;
            }
        }
    }

    // PHASE 10: Write meshlet descriptor
    if (lid == 0 && batch_capped_vertex_count_val > 0 && batch_capped_primitive_count_val > 0) {
        uint descriptorSlot = shared_batch_global_meshlet_slot;
        if (descriptorSlot < 1000000) {
            meshletDescOutput.meshletDescriptors[descriptorSlot].vertexOffset = shared_batch_global_vertex_base;
            meshletDescOutput.meshletDescriptors[descriptorSlot].indexOffset = shared_batch_global_index_base;
            meshletDescOutput.meshletDescriptors[descriptorSlot].vertexCount = batch_capped_vertex_count_val;
            meshletDescOutput.meshletDescriptors[descriptorSlot].primitiveCount = batch_capped_primitive_count_val;
            
            atomicAdd(filledMeshletDescCount.filledMeshletCounter, 1u);
        }
    }
}