#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_debug_printf: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

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
layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesTriangleTable { uint8_t triTable[]; } mcTriangleTable;
layout(set = 0, binding = 6, std430) readonly buffer MarchingCubesNumVerticesTable { uint8_t numVerticesTable[]; } mcNumVerticesTable;

// Hardcoded edge table for better performance
const uint edgeTable[256] = uint[256](
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
);

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
        if (mcTriangleTable.triTable[configuration * 16 + i * 3] == 255u) break;
        count++;
    }
    return count;
}

uint getVertexCount(uint configuration) {
    uint edgeMask = edgeTable[configuration];
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