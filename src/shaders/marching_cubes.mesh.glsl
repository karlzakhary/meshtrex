#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

// --- Configurable Parameters ---
#define BLOCK_DIM_X 4
#define BLOCK_DIM_Y 4
#define BLOCK_DIM_Z 4
#define CELLS_PER_BLOCK 64

// --- Workgroup size ---
layout(local_size_x = CELLS_PER_BLOCK, local_size_y = 1, local_size_z = 1) in;

// --- Output limits ---
layout(max_vertices = 64, max_primitives = 126) out;
layout(triangles) out;

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

struct VertexData {
    vec4 position;
    vec4 normal;
};

taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
} taskPayloadIn;

// --- Descriptor Set Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { uint8_t triTable[]; } mcTriangleTable;
layout(set = 0, binding = 6, std430) buffer VertexBuffer { VertexData data[]; } vertices;
layout(set = 0, binding = 7, std430) buffer VertexCount { uint vertexCounter; } vCount;
layout(set = 0, binding = 8, std430) buffer IndexBuffer { uint data[]; } indices;
layout(set = 0, binding = 9, std430) buffer IndexCount { uint indexCounter; } iCount;
layout(set = 0, binding = 10, std430) buffer MeshletDescriptorBuffer { MeshletDescriptor descriptors[]; } meshlets;
layout(set = 0, binding = 11, std430) buffer MeshletDescriptorCount { uint meshletCounter; } meshletCount;

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

// --- Hardcoded table ---
const ivec2 corner_edge_table[12] = {
    {0,1}, {1,2}, 
    {2,3}, {3,0}, 
    {4,5}, {5,6}, 
    {6,7}, {7,4},
    {0,4}, {1,5}, 
    {2,6}, {3,7}
};

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

// --- Shared Memory ---
shared uint s_vertexBase;
shared uint s_indexBase;
shared uint s_meshletID;

// --- Helper Functions ---
uint getPrimitiveCount(uint configuration) {
    uint primitiveCount = 0;
    for (int i = 0; i < 5; i++) {
        if (uint(mcTriangleTable.triTable[configuration * 16 + i * 3]) == 255u) break;
        primitiveCount++;
    }
    return primitiveCount;
}

uvec3 unpack_block_id(uint id) {
    uint grid_width = ubo.blockGridDim.x;
    uint grid_slice = ubo.blockGridDim.x * ubo.blockGridDim.y;
    return uvec3(id % grid_width, (id / grid_width) % ubo.blockGridDim.y, id / grid_slice);
}

// Safely calculates the normal vector at a point using a robust combination of
// central, forward, and backward differencing to correctly handle volume boundaries.
vec3 calculate_normal(ivec3 p) {
    ivec3 dims = ivec3(ubo.volumeDim.xyz);
    
    // The value at the point itself, needed for one-sided differences
    float s_center = float(imageLoad(volumeImage, p).r);

    // --- X-axis gradient ---
    float gx = 0.0;
    if (p.x == 0) { // Left boundary: use forward difference
        gx = float(imageLoad(volumeImage, p + ivec3(1, 0, 0)).r) - s_center;
    } else if (p.x == dims.x - 1) { // Right boundary: use backward difference
        gx = s_center - float(imageLoad(volumeImage, p - ivec3(1, 0, 0)).r);
    } else { // Interior: use central difference
        float s1 = float(imageLoad(volumeImage, p - ivec3(1, 0, 0)).r);
        float s2 = float(imageLoad(volumeImage, p + ivec3(1, 0, 0)).r);
        gx = (s2 - s1) / 2.0;
    }

    // --- Y-axis gradient ---
    float gy = 0.0;
    if (p.y == 0) { // Bottom boundary
        gy = float(imageLoad(volumeImage, p + ivec3(0, 1, 0)).r) - s_center;
    } else if (p.y == dims.y - 1) { // Top boundary
        gy = s_center - float(imageLoad(volumeImage, p - ivec3(0, 1, 0)).r);
    } else { // Interior
        float s3 = float(imageLoad(volumeImage, p - ivec3(0, 1, 0)).r);
        float s4 = float(imageLoad(volumeImage, p + ivec3(0, 1, 0)).r);
        gy = (s4 - s3) / 2.0;
    }

    // --- Z-axis gradient ---
    float gz = 0.0;
    if (p.z == 0) { // Front boundary
        gz = float(imageLoad(volumeImage, p + ivec3(0, 0, 1)).r) - s_center;
    } else if (p.z == dims.z - 1) { // Back boundary
        gz = s_center - float(imageLoad(volumeImage, p - ivec3(0, 0, 1)).r);
    } else { // Interior
        float s5 = float(imageLoad(volumeImage, p - ivec3(0, 0, 1)).r);
        float s6 = float(imageLoad(volumeImage, p + ivec3(0, 0, 1)).r);
        gz = (s6 - s5) / 2.0;
    }
    
    // The final gradient vector
    vec3 gradient = vec3(gx, gy, gz);

    // Return a default, valid normal in this case to prevent NaN from normalize().
    if (length(gradient) < 0.00001) {
        return vec3(0.0, 1.0, 0.0);
    }

    // Note: The gradient points from lower to higher density. For an isosurface, the normal
    // should typically point "out", which is towards lower density. Hence, we negate the gradient.
    return -normalize(gradient);
}

// Interpolates vertex data between two points to find the precise surface location.
// Relies on the robust `calculate_normal` function to avoid NaN values.
VertexData interpolate_vertex(float isolevel, ivec3 p1_coord, ivec3 p2_coord) {
    float v1_val = float(imageLoad(volumeImage, p1_coord).r);
    float v2_val = float(imageLoad(volumeImage, p2_coord).r);

    // Calculate normals at the two endpoints using the new, safe function
    vec3 n1 = calculate_normal(p1_coord);
    vec3 n2 = calculate_normal(p2_coord);

    float mu = 0.5; // Default value in case of flat surface
    float denominator = v2_val - v1_val;

    // Safety Check: Avoid division by zero
    if (abs(denominator) > 0.00001) {
        mu = (isolevel - v1_val) / denominator;
    }
    
    // Clamp mu to ensure the position stays on the edge between p1 and p2
    mu = clamp(mu, 0.0, 1.0);
    
    vec3 pos = mix(vec3(p1_coord), vec3(p2_coord), mu);
    vec3 interpolated_norm = mix(n1, n2, mu);

    // Final safety check on the *interpolated* normal before normalizing
    if (length(interpolated_norm) < 0.00001) {
        interpolated_norm = n1; // Fallback to one of the endpoint normals if interpolation results in zero
    }
    
    vec3 norm = normalize(interpolated_norm);

    // Convert position to normalized device coordinates (e.g., [-1, 1])
    vec3 final_pos = (pos / vec3(ubo.volumeDim.xyz)) * 2.0 - 1.0;

    return VertexData(vec4(final_pos, 1.0), vec4(norm, 0.0));
}


void main()
{
    const uint cellID   = gl_LocalInvocationID.x;          // 0 … 63
    const uint sgLane   = gl_SubgroupInvocationID;         // lane in subgroup
    const bool sgLeader = subgroupElect();                 // one TRUE per subgroup

    /*──────────────────────── 1 · classify this cell ────────────────────*/
    uvec3 blockCoord = unpack_block_id(taskPayloadIn.blockID);
    uvec3 cellLocal  = uvec3(cellID % BLOCK_DIM_X,
                             (cellID / BLOCK_DIM_X) % BLOCK_DIM_Y,
                              cellID / (BLOCK_DIM_X * BLOCK_DIM_Y));
    ivec3 cellGlobal = ivec3(blockCoord * ubo.blockDim.xyz + cellLocal);

    uint cfg = 0u, edgeMask = 0u;
    uint vLocal = 0u, pLocal = 0u;          // vertices / triangles
    bool hasGeom = false;                   // does this cell emit geometry?

    if (all(lessThan(cellGlobal, ivec3(ubo.volumeDim) - 1))) {
        for (int c = 0; c < 8; ++c) {
            float s = float(imageLoad(volumeImage,
                                       cellGlobal + cornerOffset[c]).r);
            if (s <= ubo.isovalue) cfg |= 1u << c;
        }
        edgeMask = edgeTable[cfg];
        if (edgeMask != 0u) {               // only if surface actually cuts cell
            vLocal  = bitCount(edgeMask);
            pLocal  = getPrimitiveCount(cfg);
            hasGeom = true;
        }
    }

    /*──────────────────────── 2 · subgroup scan & alloc (all lanes) ─────*/
    uint vPrefix = subgroupExclusiveAdd(vLocal);           // vertices prefix
    uint pPrefix = subgroupExclusiveAdd(pLocal);           // tris    prefix

    uint vTotSG  = subgroupAdd(vLocal);                    // subgroup totals
    uint pTotSG  = subgroupAdd(pLocal);

    uint vBase = 0u, pBase = 0u;
    if (sgLeader) {
        vBase = atomicAdd(vCount.vertexCounter, vTotSG);
        pBase = atomicAdd(iCount.indexCounter,  pTotSG * 3u);
    }
    vBase = subgroupBroadcastFirst(vBase);
    pBase = subgroupBroadcastFirst(pBase);

    uint vDst = vBase + vPrefix;                           // first vertex
    uint pDst = pBase + pPrefix * 3u;                      // first index

    /* first subgroup leader reserves meshlet descriptor base */
    if (gl_SubgroupID == 0 && sgLeader) {
        s_meshletID  = atomicAdd(meshletCount.meshletCounter, 1u);
        s_vertexBase = vBase;
        s_indexBase  = pBase;
    }

    /*──────────────────────── 3 · geometry generation (only if needed) ──*/
    if (hasGeom) {
        int localIndex[12];  for (int e = 0; e < 12; ++e) localIndex[e] = -1;

        /* 3-a  vertices on active edges */
        for (int e = 0; e < 12; ++e)
            if (((edgeMask >> e) & 1) != 0) {
                int rank = bitCount(edgeMask & ((1u << e) - 1u));
                localIndex[e] = rank;

                ivec2 ce = corner_edge_table[e];
                VertexData vd = interpolate_vertex(
                    ubo.isovalue,
                    cellGlobal + cornerOffset[ce.x],
                    cellGlobal + cornerOffset[ce.y]);

                vertices.data[vDst + uint(rank)] = vd;
            }

        /* 3-b  triangles */
        for (uint t = 0; t < pLocal; ++t) {
            uint e0 = uint(mcTriangleTable.triTable[cfg*16 + t*3 + 0]);
            uint e1 = uint(mcTriangleTable.triTable[cfg*16 + t*3 + 1]);
            uint e2 = uint(mcTriangleTable.triTable[cfg*16 + t*3 + 2]);

            indices.data[pDst + t*3 + 0] = vDst + uint(localIndex[e0]);
            indices.data[pDst + t*3 + 1] = vDst + uint(localIndex[e1]);
            indices.data[pDst + t*3 + 2] = vDst + uint(localIndex[e2]);
        }
    }

    /*──────────────────────── 4 · final WG descriptor & dummy output ────*/
    barrier();                 /* all writes/atomics done                 */
    memoryBarrierBuffer();     /* make them visible before we reread      */

    if (cellID == 0) {
        uint primTotal = (atomicAdd(iCount.indexCounter, 0u) - s_indexBase) / 3u;
        if (primTotal > 0u) {
            uint vertTotal = atomicAdd(vCount.vertexCounter, 0u) - s_vertexBase;

            meshlets.descriptors[s_meshletID] =
                MeshletDescriptor(s_vertexBase, s_indexBase,
                                  vertTotal, primTotal);
        }
        SetMeshOutputsEXT(0u, 0u);            // no raster output
    }
}