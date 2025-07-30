#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable

// --- Configurable Parameters ---
#define WORKGROUP_SIZE 64   // Use a full workgroup to process all cells in parallel
#define CELLS_PER_BLOCK 64

// --- Payload passed to Mesh Shader ---
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
} taskPayloadOut;

// --- Push Constants Block ---
// This receives per-dispatch data from vkCmdPushConstants
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

// --- Descriptor Set Bindings ---
// (Assuming other data is still in descriptor sets)
// Binding 0: UBO
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 3, std430) readonly buffer ActiveBlockIDs { uint ids[]; } activeBlockIDs;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;

// --- Workgroup Definition ---
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
shared uint s_totalPrimitivesInBlock;

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
    uint workgroup_idx = gl_WorkGroupID.x;
    if (workgroup_idx >= pc.activeBlockCount) {
        EmitMeshTasksEXT(0, 1, 1);
        return;
    }

    if (gl_LocalInvocationID.x == 0) {
        s_totalPrimitivesInBlock = 0;
    }
    barrier();

    // Each invocation processes one cell to count primitives
    uint cellIdx = gl_LocalInvocationID.x;
    
    // Get blockID using the offset from push constants
    uint blockID = activeBlockIDs.ids[pc.blockOffset + workgroup_idx];
    uvec3 blockCoord = unpack_block_id(blockID);
    
    uvec3 cellCoord_local = uvec3(cellIdx % 4, (cellIdx / 4) % 4, cellIdx / (4 * 4));
    ivec3 cellCoord_global = ivec3(blockCoord * pc.blockDim.xyz + cellCoord_local);

    uint primCount = 0;
    if (all(lessThan(cellCoord_global, ivec3(pc.volumeDim.xyz) - 1))) {
        uint configuration = calculate_configuration(cellCoord_global);
        primCount = getPrimitiveCount(configuration);
    }
    
    if (primCount > 0) {
        atomicAdd(s_totalPrimitivesInBlock, primCount);
    }
    barrier();

    // Leader thread emits a single mesh task if the block has any geometry
    if (gl_LocalInvocationID.x == 0) {
        if (s_totalPrimitivesInBlock > 0) {
            taskPayloadOut.blockID = blockID;
            EmitMeshTasksEXT(1, 1, 1);
        } else {
            EmitMeshTasksEXT(0, 1, 1);
        }
    }
}