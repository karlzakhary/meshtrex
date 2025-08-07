#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_debug_printf : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

// Unified mesh shader for both transient extraction passes
// Implements Kreskowski-style direct triangle extraction from compact cell lists

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 96, max_primitives = 126) out;

// Input bindings
layout(binding = 0, std140) uniform ViewParameters {
    mat4 viewProj;
    uvec4 volumeDim;      // Volume dimensions (x, y, z, _)
    uvec4 blockDim;       // Block dimensions (x, y, z, _)
    uvec4 blockGridDim;   // Grid dimensions in blocks (x, y, z, _)
    float isovalue;
} viewParams;

// Volume data image
layout(binding = 1, r8ui) uniform readonly uimage3D volumeImage;

// Push constants for additional parameters
layout(push_constant) uniform PushConstants {
    uint renderPass;  // 0 = PVS_prev, 1 = PVS_curr-prev
} pushConstants;

// Task payload from task shader (Kreskowski-style compact list)
struct TaskPayload {
    uint blockID;                    // Block being processed  
    uint halfIndex;                  // Which half of the block (0 or 1)
    uint8_t denseOccupancyIndex[256]; // Dense list of occupied cell indices (8-bit)
    uint8_t offsetAndLength[64];     // Interleaved offset/length pairs for mesh workgroups
};

taskPayloadSharedEXT TaskPayload IN;

// Vertex outputs
layout(location = 0) out vec3 outNormal[];
layout(location = 1) out vec3 outWorldPos[];
layout(location = 2) flat out uint outRenderPass[];

// Marching cubes edge table
const uint edgeTable[256] = uint[256](
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
);

// Number of unique vertices per cell configuration (0-12)
const uint numUniqueVerticesPerCell[256] = uint[256](
    0, 3, 3, 6, 3, 6, 6, 9, 3, 6, 6, 9, 6, 9, 9, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 6, 9, 12, 12, 9, 9, 12, 12, 9, 12, 9, 9, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    9, 12, 12, 9, 12, 12, 12, 9, 12, 12, 12, 9, 12, 9, 9, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 6, 12, 9, 12, 9, 12, 6,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    9, 12, 12, 9, 12, 12, 12, 9, 12, 9, 12, 6, 12, 9, 6, 3,
    6, 9, 9, 12, 9, 12, 12, 12, 9, 12, 12, 12, 12, 12, 12, 9,
    9, 12, 12, 12, 12, 12, 12, 6, 12, 9, 12, 9, 12, 9, 6, 3,
    9, 12, 12, 12, 12, 12, 12, 6, 12, 12, 12, 6, 12, 6, 6, 3,
    6, 9, 9, 6, 9, 6, 6, 3, 9, 6, 6, 3, 6, 3, 3, 0
);

// Marching cubes triangle table - defines triangles for each configuration
const int triTable[256][16] = int[256][16](
    int[16](-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1),
    int[16](8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1),
    int[16](3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1),
    int[16](4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1),
    int[16](4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1),
    int[16](9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1),
    int[16](10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1),
    int[16](5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1),
    int[16](5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1),
    int[16](8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1),
    int[16](2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1),
    int[16](2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1),
    int[16](11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1),
    int[16](5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1),
    int[16](11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1),
    int[16](11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1),
    int[16](2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1),
    int[16](6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1),
    int[16](3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1),
    int[16](6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1),
    int[16](6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1),
    int[16](8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1),
    int[16](7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1),
    int[16](3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1),
    int[16](0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1),
    int[16](9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1),
    int[16](8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1),
    int[16](5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1),
    int[16](0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1),
    int[16](6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1),
    int[16](10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1),
    int[16](1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1),
    int[16](0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1),
    int[16](3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1),
    int[16](6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1),
    int[16](9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1),
    int[16](8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1),
    int[16](3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1),
    int[16](6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1),
    int[16](10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1),
    int[16](10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1),
    int[16](2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1),
    int[16](7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1),
    int[16](7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1),
    int[16](2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1),
    int[16](1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1),
    int[16](11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1),
    int[16](8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1),
    int[16](0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1),
    int[16](7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1),
    int[16](7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1),
    int[16](10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1),
    int[16](0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1),
    int[16](7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1),
    int[16](6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1),
    int[16](6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1),
    int[16](4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1),
    int[16](10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1),
    int[16](8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1),
    int[16](1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1),
    int[16](10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1),
    int[16](10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1),
    int[16](9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1),
    int[16](7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1),
    int[16](3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1),
    int[16](7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1),
    int[16](3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1),
    int[16](6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1),
    int[16](9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1),
    int[16](1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1),
    int[16](4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1),
    int[16](7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1),
    int[16](6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1),
    int[16](0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1),
    int[16](6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1),
    int[16](0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1),
    int[16](11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1),
    int[16](6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1),
    int[16](5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1),
    int[16](9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1),
    int[16](1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1),
    int[16](10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1),
    int[16](0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1),
    int[16](10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1),
    int[16](11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1),
    int[16](9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1),
    int[16](7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1),
    int[16](2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1),
    int[16](9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1),
    int[16](9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1),
    int[16](1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1),
    int[16](5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1),
    int[16](0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1),
    int[16](10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1),
    int[16](2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1),
    int[16](0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1),
    int[16](0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1),
    int[16](9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1),
    int[16](5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1),
    int[16](5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1),
    int[16](8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1),
    int[16](9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1),
    int[16](1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1),
    int[16](3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1),
    int[16](4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1),
    int[16](9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1),
    int[16](11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1),
    int[16](11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1),
    int[16](2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1),
    int[16](9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1),
    int[16](3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1),
    int[16](1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1),
    int[16](4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1),
    int[16](0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1),
    int[16](9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1),
    int[16](1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    int[16](-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
);

// Edge-to-vertex mapping for marching cubes
const uvec2 edgeVertexMap[12] = uvec2[12](
    uvec2(0, 1), uvec2(1, 2), uvec2(2, 3), uvec2(3, 0),
    uvec2(4, 5), uvec2(5, 6), uvec2(6, 7), uvec2(7, 4),
    uvec2(0, 4), uvec2(1, 5), uvec2(2, 6), uvec2(3, 7)
);

// Sample volume data
float sampleVolume(vec3 pos) {
    ivec3 coord = ivec3(pos);
    if (any(lessThan(coord, ivec3(0))) || any(greaterThanEqual(coord, ivec3(viewParams.volumeDim.xyz)))) {
        return 255.0; // Return value > isovalue for outside bounds
    }
    return float(imageLoad(volumeImage, coord).r);
}

// Compute gradient for normal calculation
vec3 computeGradient(vec3 pos) {
    float dx = sampleVolume(pos + vec3(1,0,0)) - sampleVolume(pos - vec3(1,0,0));
    float dy = sampleVolume(pos + vec3(0,1,0)) - sampleVolume(pos - vec3(0,1,0));
    float dz = sampleVolume(pos + vec3(0,0,1)) - sampleVolume(pos - vec3(0,0,1));
    return normalize(vec3(dx, dy, dz));
}

// Linear interpolation along an edge
vec3 vertexInterp(float isolevel, vec3 p1, vec3 p2, float v1, float v2) {
    if (abs(v1 - v2) < 1e-6) return p1;
    float t = (isolevel - v1) / (v2 - v1);
    return mix(p1, p2, clamp(t, 0.0, 1.0));
}

void main() {
    uint threadID = gl_LocalInvocationID.x;
    uint workgroupID = gl_WorkGroupID.x;
    
    // Get offset and length for this mesh workgroup from task shader
    uint baseListOffset = 2 * workgroupID;
    uint occupiedVoxelStart = uint(IN.offsetAndLength[baseListOffset]);
    uint numActiveThreads = uint(IN.offsetAndLength[baseListOffset + 1]);
    
    // Get block information
    uint blockID = IN.blockID;
    uint halfIndex = IN.halfIndex;
    
    
    // Decode block coordinates
    uint blocksPerRow = viewParams.blockGridDim.x;
    uint blocksPerSlice = viewParams.blockGridDim.x * viewParams.blockGridDim.y;
    
    uint blockZ = blockID / blocksPerSlice;
    uint remaining = blockID % blocksPerSlice;
    uint blockY = remaining / blocksPerRow;
    uint blockX = remaining % blocksPerRow;
    
    uvec3 blockCoord = uvec3(blockX, blockY, blockZ);
    uvec3 blockBasePos = blockCoord * viewParams.blockDim.xyz;
    
    // Get cell index for this thread
    int localCellIndex = -1;
    if (threadID < numActiveThreads) {
        localCellIndex = int(IN.denseOccupancyIndex[occupiedVoxelStart + threadID]);
    }
    
    // Marching cubes variables
    float values[8] = float[8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    vec3 vertexPosList[12] = vec3[12](
        vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0),
        vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0),
        vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)
    );
    uint cubeIndex = 0;
    uint numVerticesFromCell = 0;
    
    // Process cell if thread is active
    if (threadID < numActiveThreads && localCellIndex >= 0) {
        // Convert cell index to 3D position within the 8x8x4 half
        uvec3 unflattenedLocalIndex;
        unflattenedLocalIndex.z = uint(localCellIndex) / 64;
        uint rem = uint(localCellIndex) % 64;
        unflattenedLocalIndex.y = rem / 8;
        unflattenedLocalIndex.x = rem % 8;
        
        // Add half-block offset for upper half
        if (halfIndex == 1) {
            unflattenedLocalIndex.z += 4;
        }
        
        uvec3 cellSamplingIndex = blockBasePos + unflattenedLocalIndex;
        
        // Sample 8 corners
        values[0] = sampleVolume(vec3(cellSamplingIndex));
        values[1] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 0, 0));
        values[2] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 1, 0));
        values[3] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 1, 0));
        values[4] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 0, 1));
        values[5] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 0, 1));
        values[6] = sampleVolume(vec3(cellSamplingIndex) + vec3(1, 1, 1));
        values[7] = sampleVolume(vec3(cellSamplingIndex) + vec3(0, 1, 1));
        
        // Compute cube index
        float scaledIsovalue = viewParams.isovalue * 255.0;

        for (uint i = 0; i < 8; i++) {
            if (values[i] < scaledIsovalue) {
                cubeIndex |= (1u << i);
            }
        }
        
        numVerticesFromCell = numUniqueVerticesPerCell[cubeIndex];
        
        // Compute vertex positions following Kreskowski's approach
        // cellSamplingIndex is already in world voxel coordinates
        vec3 voxelSize = vec3(1.0);
        vec3 vertexPosition = vec3(cellSamplingIndex);
        
        // Cell vertex positions (8 corners of the cube)
        vec3 cellVertexPositions[8];
        cellVertexPositions[0] = vertexPosition;                                          // (0,0,0)
        cellVertexPositions[1] = vertexPosition + vec3(voxelSize.x, 0, 0);               // (1,0,0)
        cellVertexPositions[2] = vertexPosition + vec3(voxelSize.x, voxelSize.y, 0);    // (1,1,0)
        cellVertexPositions[3] = vertexPosition + vec3(0, voxelSize.y, 0);               // (0,1,0)
        cellVertexPositions[4] = vertexPosition + vec3(0, 0, voxelSize.z);               // (0,0,1)
        cellVertexPositions[5] = vertexPosition + vec3(voxelSize.x, 0, voxelSize.z);    // (1,0,1)
        cellVertexPositions[6] = vertexPosition + vec3(voxelSize.x, voxelSize.y, voxelSize.z); // (1,1,1)
        cellVertexPositions[7] = vertexPosition + vec3(0, voxelSize.y, voxelSize.z);    // (0,1,1)
        
        // Interpolate vertex positions along edges
        // Edge 0: vertices 0-1
        vertexPosList[0] = vertexInterp(scaledIsovalue, cellVertexPositions[0], cellVertexPositions[1], values[0], values[1]);
        // Edge 1: vertices 1-2
        vertexPosList[1] = vertexInterp(scaledIsovalue, cellVertexPositions[1], cellVertexPositions[2], values[1], values[2]);
        // Edge 2: vertices 2-3
        vertexPosList[2] = vertexInterp(scaledIsovalue, cellVertexPositions[2], cellVertexPositions[3], values[2], values[3]);
        // Edge 3: vertices 3-0
        vertexPosList[3] = vertexInterp(scaledIsovalue, cellVertexPositions[3], cellVertexPositions[0], values[3], values[0]);
        // Edge 4: vertices 4-5
        vertexPosList[4] = vertexInterp(scaledIsovalue, cellVertexPositions[4], cellVertexPositions[5], values[4], values[5]);
        // Edge 5: vertices 5-6
        vertexPosList[5] = vertexInterp(scaledIsovalue, cellVertexPositions[5], cellVertexPositions[6], values[5], values[6]);
        // Edge 6: vertices 6-7
        vertexPosList[6] = vertexInterp(scaledIsovalue, cellVertexPositions[6], cellVertexPositions[7], values[6], values[7]);
        // Edge 7: vertices 7-4
        vertexPosList[7] = vertexInterp(scaledIsovalue, cellVertexPositions[7], cellVertexPositions[4], values[7], values[4]);
        // Edge 8: vertices 0-4
        vertexPosList[8] = vertexInterp(scaledIsovalue, cellVertexPositions[0], cellVertexPositions[4], values[0], values[4]);
        // Edge 9: vertices 1-5
        vertexPosList[9] = vertexInterp(scaledIsovalue, cellVertexPositions[1], cellVertexPositions[5], values[1], values[5]);
        // Edge 10: vertices 2-6
        vertexPosList[10] = vertexInterp(scaledIsovalue, cellVertexPositions[2], cellVertexPositions[6], values[2], values[6]);
        // Edge 11: vertices 3-7
        vertexPosList[11] = vertexInterp(scaledIsovalue, cellVertexPositions[3], cellVertexPositions[7], values[3], values[7]);
    }
    
    // Perform subgroup prefix sum to get vertex write offsets
    uint vertexWriteOffset = numVerticesFromCell;
    vertexWriteOffset = subgroupInclusiveAdd(vertexWriteOffset);
    uint totalNumVertices = subgroupBroadcast(vertexWriteOffset, 31);
    vertexWriteOffset -= numVerticesFromCell;
    
    // Track unique vertices per cell to avoid duplicates
    uint8_t uniqueVertexLocationsPerCell[12];
    uint8_t numUniqueVerticesWritten = uint8_t(0);
    
    // Count total indices per cell for prefix sum
    uint totalIndicesPerCell = 0;
    for (uint tri = 0; tri < 16; tri += 3) {
        if (triTable[cubeIndex][tri] == -1) break;
        totalIndicesPerCell += 3;
    }
    
    uint intraWarpCellIndicesPrefixSum = subgroupInclusiveAdd(totalIndicesPerCell);
    
    // Process triangles following Kreskowski's approach
    if (threadID < numActiveThreads && cubeIndex != 0 && cubeIndex != 255) {
        // Temporary storage for triangle indices
        uint triangleIndices[15]; // Max 5 triangles * 3 vertices per cell
        uint triangleCount = 0;
        
        for (uint triangleBaseVertexIdx = 0; triangleBaseVertexIdx < totalIndicesPerCell; triangleBaseVertexIdx += 3) {
            // Process each vertex of the triangle
            for (int triVertexIdx = 0; triVertexIdx < 3; ++triVertexIdx) {
                int edgeIdx = triTable[cubeIndex][triangleBaseVertexIdx + triVertexIdx];
                
                // Check if this vertex needs to be written (not already processed)
                bool needsWrite = true;
                uint8_t vertexLocation = uint8_t(0);
                
                // Search for existing vertex
                for (uint8_t i = uint8_t(0); i < numUniqueVerticesWritten; i++) {
                    if (uniqueVertexLocationsPerCell[i] == uint8_t(edgeIdx)) {
                        needsWrite = false;
                        vertexLocation = i;
                        break;
                    }
                }
                
                if (needsWrite) {
                    // Write new vertex
                    uint8_t attributeWriteOffset = uint8_t(vertexWriteOffset) + numUniqueVerticesWritten;
                    vec3 vertPos = vertexPosList[edgeIdx];
                    
                    vec4 clipPos = viewParams.viewProj * vec4(vertPos, 1.0);

                    gl_MeshVerticesEXT[attributeWriteOffset].gl_Position = clipPos;
                    outNormal[attributeWriteOffset] = computeGradient(vertPos);
                    outWorldPos[attributeWriteOffset] = vertPos;
                    outRenderPass[attributeWriteOffset] = pushConstants.renderPass;
                    
                    uniqueVertexLocationsPerCell[numUniqueVerticesWritten] = uint8_t(edgeIdx);
                    vertexLocation = numUniqueVerticesWritten;
                    numUniqueVerticesWritten++;
                }
                
                // Store the index (with bounds check)
                if (triangleCount < 15) {
                    triangleIndices[triangleCount++] = uint(vertexWriteOffset + vertexLocation);
                }
            }
        }
        
        // Now write complete triangles
        uint triangleWriteOffset = (intraWarpCellIndicesPrefixSum - totalIndicesPerCell) / 3;
        for (uint t = 0; t < triangleCount; t += 3) {
            uvec3 indices = uvec3(triangleIndices[t], triangleIndices[t + 1], triangleIndices[t + 2]);
            
            gl_PrimitiveTriangleIndicesEXT[triangleWriteOffset + (t / 3)] = indices;
        }
    }
    
    // Get total indices from thread 31's prefix sum
    uint totalIndices = subgroupBroadcast(intraWarpCellIndicesPrefixSum, 31);
    
    barrier();
    
    // Thread 0 sets the mesh outputs (using thread 0 for consistency)
    if (threadID == 0) {
        uint totalPrimitives = totalIndices / 3;
        
        SetMeshOutputsEXT(uint8_t(totalNumVertices), uint8_t(totalPrimitives));
    }
}