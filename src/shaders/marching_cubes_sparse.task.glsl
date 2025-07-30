#version 460 core
#extension GL_EXT_mesh_shader : require

// --- Configurable Parameters ---
#define BLOCKS_PER_WORKGROUP 6
#define WORKGROUP_SIZE 32
#define CELLS_PER_BLOCK 64

// --- Payload (Simplified) ---
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
} taskPayloadOut;

// --- Push Constants Block (Single Source of Truth) ---
layout(push_constant, std430) uniform block {
    uvec4 volumeDim;            // Offset 0 - Not used, get from UBO
    uvec4 blockDim;             // Offset 16 - Not used, get from UBO  
    uvec4 blockGridDim;         // Offset 32 - Not used, get from UBO
    vec4 voxelSize;             // Offset 48
    vec4 origin;                // Offset 64
    float isovalue;             // Offset 80 - Not used, get from UBO
    int activeBlockCount;       // Offset 84
    uint globalVertexOffset;    // Offset 88
    uint globalIndexOffset;     // Offset 92
    uint globalMeshletOffset;   // Offset 96
    uint densityClass;          // Offset 100
    uint blockOffset;           // Offset 104
    uint _padding;              // Offset 108
} pc;

// --- Descriptor Set Bindings (Binding 0 is now free) ---
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 3, std430) readonly buffer ActiveBlockIDs { uint ids[]; } activeBlockIDs;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Helper Functions (Using 'pc' members) ---
// --- Helper Functions ---
const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1)
);

uint getPrimitiveCount(uint configuration) {
    uint primitiveCount = 0;
    for (int i = 0; i < 5; i++) {
        if (mcTriangleTable.triTable[configuration * 16 + i * 3] == -1) break;
        primitiveCount++;
    }
    return primitiveCount;
}
uvec3 unpack_block_id(uint id) {
    uint grid_width = pc.blockGridDim.x;
    uint grid_slice = pc.blockGridDim.x * pc.blockGridDim.y;
    return uvec3(id % grid_width, (id / grid_width) % pc.blockGridDim.y, id / grid_slice);
}
uint calculate_configuration(ivec3 cell_coord_global) {
    uint configuration = 0;
    for (int i = 0; i < 8; ++i) {
        float value = float(imageLoad(volumeImage, cell_coord_global + cornerOffset[i]).r);
        if (value <= pc.isovalue) {
            configuration |= (1u << i);
        }
    }
    return configuration;
}

// --- Main Function ---
void main () {
    uint baseBlockIdx = gl_WorkGroupID.x * BLOCKS_PER_WORKGROUP;

    // Each lane is responsible for one of the blocks in the workgroup's batch
    if (gl_LocalInvocationID.x < BLOCKS_PER_WORKGROUP) {
        uint currentBlockLinearIdx = baseBlockIdx + gl_LocalInvocationID.x;

        if (currentBlockLinearIdx < pc.activeBlockCount) {
            uint blockID = activeBlockIDs.ids[pc.blockOffset + currentBlockLinearIdx];

            // A quick check is sufficient for sparse blocks. If any cell has geometry, we dispatch.
            // A full count is overkill here and hurts performance.
            uvec3 blockCoord = unpack_block_id(blockID);
            for (uint i = 0; i < CELLS_PER_BLOCK; ++i) {
                uvec3 cellLocal = uvec3(i % 4, (i/4)%4, i/16);
                ivec3 cellGlobal = ivec3(blockCoord * pc.blockDim.xyz) + ivec3(cellLocal);

                if (all(lessThan(cellGlobal, ivec3(pc.volumeDim.xyz) - 1))) {
                    uint config = calculate_configuration(cellGlobal);
                    if (getPrimitiveCount(config) > 0) {
                        taskPayloadOut.blockID = blockID;
                        EmitMeshTasksEXT(1, 1, 1);
                        // Once we find geometry, we know the block is active. No need to continue checking.
                        return;
                    }
                }
            }
        }
    }
}