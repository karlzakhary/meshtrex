#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
#define BLOCK_DIM_X 4
#define BLOCK_DIM_Y 4
#define BLOCK_DIM_Z 4
#define CELLS_PER_BLOCK 64 // 4x4x4

// Workgroup size for this Task Shader
#define WORKGROUP_SIZE 32
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Payload passed to Mesh Shader ---
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
} taskPayloadOut;

// --- Descriptor Set Bindings ---
// Binding 0: UBO
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

// Binding 1: Volume Image
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;

// Binding 2: Compacted Active Blocks counter 
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount { uint count; } activeBlockCount;

// Binding 3: Compacted Active Block IDs
layout(set = 0, binding = 3, std430) readonly buffer ActiveBlockIDs { uint ids[]; } activeBlockIDs;

// Binding 4: Triangle-Edge Connectivity Table (Size: 256 * 16)
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;
// --- End Descriptor Set Bindings ---

// --- Shared Memory ---
shared uint s_totalPrimitivesInBlock;

// --- Helper Functions ---

/**
 * Calculates the number of primitives (triangles) for a given Marching Cubes configuration
 * by reading the triTable and counting entries until the '-1' terminator is found.
 */
uint getPrimitiveCount(uint configuration) {
    uint primitiveCount = 0;
    // Max 5 triangles per cell, check each one.
    for (int i = 0; i < 5; i++) {
        // Accessing the flattened 2D array: triTable[configuration][i*3]
        if (mcTriangleTable.triTable[configuration * 16 + i * 3] == -1) {
            break; // Found the end of the list for this configuration.
        }
        primitiveCount++;
    }
    return primitiveCount;
}

// Converts a 1D block ID to 3D block coordinates
uvec3 unpack_block_id(uint id) {
    uint grid_width = ubo.blockGridDim.x;
    uint grid_slice = ubo.blockGridDim.x * ubo.blockGridDim.y;
    return uvec3(
        id % grid_width,
        (id / grid_width) % ubo.blockGridDim.y,
        id / grid_slice
    );
}
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

// --- Main Function ---
void main () {
    uint workgroup_idx = gl_WorkGroupID.x;
    if (workgroup_idx >= activeBlockCount.count) {
        EmitMeshTasksEXT(0, 1, 1);
        return;
    }

    if (gl_LocalInvocationID.x == 0) {
        s_totalPrimitivesInBlock = 0;
    }
    barrier();

    uint blockID = activeBlockIDs.ids[workgroup_idx];
    uvec3 blockCoord = unpack_block_id(blockID);

    // Each invocation processes a subset of the cells in the block
    for (uint i = gl_LocalInvocationID.x; i < CELLS_PER_BLOCK; i += WORKGROUP_SIZE) {
        uvec3 cellCoord_local = uvec3(i % BLOCK_DIM_X, (i / BLOCK_DIM_X) % BLOCK_DIM_Y, i / (BLOCK_DIM_X * BLOCK_DIM_Y));
        ivec3 cellCoord_global = ivec3(blockCoord * ubo.blockDim.xyz + cellCoord_local);

    if (any(greaterThanEqual(cellCoord_global, ivec3(ubo.volumeDim) - 1)))
        continue;

        uint configuration = calculate_configuration(cellCoord_global);
        uint primCount = getPrimitiveCount(configuration);
        
        if (primCount > 0) {
            atomicAdd(s_totalPrimitivesInBlock, primCount);
        }
    }
    barrier();

    if (gl_LocalInvocationID.x == 0) {
        if (s_totalPrimitivesInBlock > 0) {
            taskPayloadOut.blockID = blockID;
            EmitMeshTasksEXT(1, 1, 1);
        } else {
            EmitMeshTasksEXT(0, 1, 1);
        }
    }
}