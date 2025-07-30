#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_debug_printf : enable

// --- Debug Configuration ---
// Instead of complex conditional compilation, we'll use a simpler approach
#define DEBUG_MODE 0

// --- Compile-time Maximum Constants ---
#define MAX_BLOCK_DIM 3u
#define MAX_CELLS_IN_BLOCK 27u  // MAX_BLOCK_DIM^3
#define MAX_BLOCK_CONTEXT_DIM 4u  // MAX_BLOCK_DIM + 1

// --- Specialization Constants ---
layout(constant_id = 0) const uint BX = 3u;
layout(constant_id = 1) const uint BY = 3u;
layout(constant_id = 2) const uint BZ = 3u;

// --- Workgroup Configuration ---
#define WORKGROUP_SIZE 32
const uint MAX_OCC_CELLS_PER_THREAD = 2u;  // (64 + 32 - 1) / 32
#define MAX_VERTS_PER_MESHLET 64u
#define MAX_PRIMS_PER_MESHLET 126u
#define MAX_MESHLETS_PER_BLOCK 4u

// --- Algorithm Constants ---
const uint PMB_EDGE_X = 0u;
const uint PMB_EDGE_Y = 3u;
const uint PMB_EDGE_Z = 8u;

// --- Task Payload Structure ---
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
    uint meshletCount;
    uint firstCell[MAX_MESHLETS_PER_BLOCK];
    uint cellCount[MAX_MESHLETS_PER_BLOCK];
    uint packedCellData[MAX_CELLS_IN_BLOCK];
    // Debug fields always present but only used when debugging
    uint debugInfo[4];  // Store various debug counters
} TP;

// --- Shared Memory ---
shared uint sh_temp_occ_list[WORKGROUP_SIZE * MAX_OCC_CELLS_PER_THREAD];
shared uint sh_subgroup_sums[WORKGROUP_SIZE];
shared uint total_occ_count;
// Debug tracking variables
shared uint sh_debug_counters[8];  // Various debug counters

// --- Descriptor Set Bindings ---
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

// Simple debug print function that's less likely to cause SPIR-V issues
void debugPrint(uint value1, uint value2, uint value3) {
    // Only print from first thread to avoid overwhelming output
    if (gl_LocalInvocationIndex == 0 && DEBUG_MODE == 1) {
        debugPrintfEXT("Debug: %u %u %u", value1, value2, value3);
    }
}

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const uint lane = gl_LocalInvocationIndex;
    
    // Initialize debug counters
    if (lane < 8) {
        sh_debug_counters[lane] = 0;
    }
    
    // Early exit for inactive workgroups
    if (gl_WorkGroupID.x >= activeBlockCount.count) {
        if (lane == 0) EmitMeshTasksEXT(0, 1, 1);
        return;
    }
    
    barrier();

    // Calculate runtime values
    const uint ACTUAL_CELLS_PER_BLOCK = BX * BY * BZ;
    const uint ACTUAL_STRIDE = BX;
    
    // Single debug output at start
    if (lane == 0 && DEBUG_MODE == 1) {
        debugPrintfEXT("Task: Block %u, Dims %ux%ux%u", 
                      gl_WorkGroupID.x, BX, BY, BZ);
    }

    // =======================================================================
    // PASS 1: Find occupied cells
    // =======================================================================
    uint local_occ_count = 0;
    uint blockID = activeBlockIDs.ids[gl_WorkGroupID.x];
    ivec3 base = ivec3(unpack_block_id(blockID)) * int(ACTUAL_STRIDE);

    for (uint cell = lane; cell < ACTUAL_CELLS_PER_BLOCK; cell += WORKGROUP_SIZE)
    {
        uvec3 cLoc = uvec3(cell % BX, (cell / BX) % BY, cell / (BX * BY));
        
        // Bounds check
        if (any(greaterThanEqual(base + ivec3(cLoc), ivec3(ubo.volumeDim.xyz) - 1))) {
            atomicAdd(sh_debug_counters[0], 1);  // Count out-of-bounds cells
            continue;
        }
        
        uint cfg = calculate_configuration(base + ivec3(cLoc));
        uint prims = getPrimitiveCount(cfg);
        if (prims == 0u) continue;

        uint eMask = mcEdgeTable.edgeTable[cfg];
        if (eMask == 0u) continue;
        
        uint owner_verts = 0;
        if ((eMask & (1u << PMB_EDGE_X)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Y)) != 0) owner_verts++;
        if ((eMask & (1u << PMB_EDGE_Z)) != 0) owner_verts++;

        if (local_occ_count < MAX_OCC_CELLS_PER_THREAD) {
            uint write_slot = lane * MAX_OCC_CELLS_PER_THREAD + local_occ_count;
            
            // Safety check without debug printf in the condition
            bool safe_write = (write_slot < WORKGROUP_SIZE * MAX_OCC_CELLS_PER_THREAD);
            if (safe_write) {
                uint packed_data = (cell & 0xFFFFu) << 16 | (prims & 0xFFu) << 8 | (owner_verts & 0xFFu);
                sh_temp_occ_list[write_slot] = packed_data;
                local_occ_count++;
            } else {
                atomicAdd(sh_debug_counters[1], 1);  // Count write overflows
            }
        } else {
            atomicAdd(sh_debug_counters[2], 1);  // Count per-thread overflows
        }
    }
    barrier();

    // =======================================================================
    // PASS 2: Parallel scan
    // =======================================================================
    uint subgroup_offset = subgroupExclusiveAdd(local_occ_count);
    uint subgroup_total = subgroupAdd(local_occ_count);
    
    if (subgroupElect()) {
        sh_subgroup_sums[gl_SubgroupID] = subgroup_total;
    }
    barrier();
    
    if (gl_SubgroupID == 0) {
        uint subgroup_sum_val = (gl_SubgroupInvocationID < gl_NumSubgroups) ? 
                                sh_subgroup_sums[gl_SubgroupInvocationID] : 0;
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
    // PASS 3: Compact data
    // =======================================================================
    for (uint i = 0; i < local_occ_count; i++) {
        uint read_idx = lane * MAX_OCC_CELLS_PER_THREAD + i;
        uint write_idx = final_occ_offset + i;
        
        if (write_idx < MAX_CELLS_IN_BLOCK && write_idx < total_occ_count) {
            TP.packedCellData[write_idx] = sh_temp_occ_list[read_idx];
        }
    }
    barrier();

    // =======================================================================
    // Final partitioning
    // =======================================================================
    if (lane == 0)
    {
        uint occ = min(total_occ_count, MAX_CELLS_IN_BLOCK);
        uint m = 0;
        
        // Store debug info
        TP.debugInfo[0] = occ;
        TP.debugInfo[1] = sh_debug_counters[0];  // Out of bounds cells
        TP.debugInfo[2] = sh_debug_counters[1];  // Write overflows
        TP.debugInfo[3] = sh_debug_counters[2];  // Thread overflows
        
        if (occ > 0) {
            uint runV = 0, runP = 0, first = 0;
            
            for (uint i = 0; i < occ; ++i) {
                uint packed_data = TP.packedCellData[i];
                uint prims = (packed_data >> 8) & 0xFFu;
                uint owner_verts = packed_data & 0xFFu;

                if (i > first && (runV + owner_verts > MAX_VERTS_PER_MESHLET || 
                                  runP + prims > MAX_PRIMS_PER_MESHLET)) {
                    if (m < MAX_MESHLETS_PER_BLOCK) {
                        TP.firstCell[m] = first;
                        TP.cellCount[m] = i - first;
                        m++;
                    }
                    first = i;
                    runV = 0;
                    runP = 0;
                }
                runV += owner_verts;
                runP += prims;
            }

            if (m < MAX_MESHLETS_PER_BLOCK && first < occ) {
                TP.firstCell[m] = first;
                TP.cellCount[m] = occ - first;
                m++;
            }
        }
        
        TP.meshletCount = m;
        TP.blockID = blockID;
        
        // Final debug output
        if (DEBUG_MODE == 1) {
            debugPrintfEXT("Task complete: %u meshlets, %u cells", m, occ);
        }
        
        EmitMeshTasksEXT(m, 1, 1);
    }
}
