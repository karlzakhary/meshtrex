#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_debug_printf: require

// Specialization constants for dynamic block dimensions
layout(constant_id = 0) const uint BX = 1u;
layout(constant_id = 1) const uint BY = 1u;
layout(constant_id = 2) const uint BZ = 1u;

#define WORKGROUP_SIZE 32
#define CELLS_PER_BLOCK BX * BY * BZ

// Regular marching cubes: each cell can generate up to 15 vertices and 5 triangles
const uint MAX_VERTS_PER_CELL = 15u;
const uint MAX_PRIMS_PER_CELL = 5u;

// --- Payload passed to Mesh Shader ---
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
} taskPayloadOut;

// Descriptor bindings
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 2) uniform usampler3D minMaxImage;
layout(set = 0, binding = 3, std430) readonly buffer ActiveBlockCount { uint count; } activeBlockCount;
layout(set = 0, binding = 4, std430) readonly buffer ActiveBlockIDs { uint ids[]; } activeBlockIDs;
layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;
layout(set = 0, binding = 6, std430) readonly buffer MarchingCubesEdgeTable { int edgeTable[]; } mcEdgeTable;

// Push constants for frustum culling
layout(push_constant) uniform RenderPushConstants {
    mat4 viewProj;
    vec4 frustumPlanes[6];
} pushConsts;

// Utility functions
uvec3 unpack_block_id(uint id) {
    return uvec3(
        id % ubo.blockGridDim.x,
        (id / ubo.blockGridDim.x) % ubo.blockGridDim.y,
        id / (ubo.blockGridDim.x * ubo.blockGridDim.y)
    );
}

float sampleVolume(vec3 coord) {
    if (any(greaterThanEqual(coord, vec3(ubo.volumeDim.xyz)))) return 0.0;
    return float(imageLoad(volumeImage, ivec3(coord)).r);
}

uint calculateConfiguration(uvec3 cellCoord) {
    uint config = 0;
    for (uint i = 0; i < 8; i++) {
        vec3 cornerPos = vec3(cellCoord) + vec3(
            float(i & 1),
            float((i >> 1) & 1),
            float((i >> 2) & 1)
        );
        if (sampleVolume(cornerPos) <= ubo.isovalue) {
            config |= (1u << i);
        }
    }
    return config;
}

uint getPrimitiveCount(uint configuration) {
    uint count = 0;
    for (uint i = 0; i < 5; i++) {
        if (mcTriangleTable.triTable[configuration * 16 + i * 3] == -1) break;
        count++;
    }
    return count;
}

uint getVertexCount(uint configuration) {
    uint edgeMask = uint(mcEdgeTable.edgeTable[configuration]);
    return bitCount(edgeMask);
}

bool isBlockInFrustum(vec3 minPos, vec3 maxPos) {
    for (int i = 0; i < 6; i++) {
        vec4 plane = pushConsts.frustumPlanes[i];
        vec3 p = mix(minPos, maxPos, step(0.0, plane.xyz));
        if (dot(vec4(p, 1.0), plane) < 0.0) {
            return false;
        }
    }
    return true;
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

// --- Shared Memory ---
shared uint s_totalPrimitivesInBlock;

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

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
    
    // Check frustum culling
    vec3 blockMin = vec3(blockCoord * ubo.blockDim.xyz);
    vec3 blockMax = blockMin + vec3(ubo.blockDim.xyz);
    if (!isBlockInFrustum(blockMin, blockMax)) {
        // return;
    }

    // Each invocation processes a subset of the cells in the block
    for (uint i = gl_LocalInvocationID.x; i < CELLS_PER_BLOCK; i += WORKGROUP_SIZE) {
        uvec3 cellCoord_local = uvec3(i % BX, (i / BX) % BY, i / (BX * BY));
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