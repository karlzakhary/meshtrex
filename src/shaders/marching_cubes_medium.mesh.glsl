#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_ballot: require

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
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;
layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesEdgeTable { int edgeTable[]; } mcEdgeTable;
layout(set = 0, binding = 6, std430) buffer VertexBuffer { VertexData data[]; } vertices;
layout(set = 0, binding = 7, std430) buffer VertexCount { uint vertexCounter; } vCount;
layout(set = 0, binding = 8, std430) buffer IndexBuffer { uint data[]; } indices;
layout(set = 0, binding = 9, std430) buffer IndexCount { uint indexCounter; } iCount;
layout(set = 0, binding = 10, std430) buffer MeshletDescriptorBuffer { MeshletDescriptor descriptors[]; } meshlets;
layout(set = 0, binding = 11, std430) buffer MeshletDescriptorCount { uint meshletCounter; } meshletCount;

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
        edgeMask = mcEdgeTable.edgeTable[cfg];
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
            int e0 = mcTriangleTable.triTable[cfg*16 + t*3 + 0];
            int e1 = mcTriangleTable.triTable[cfg*16 + t*3 + 1];
            int e2 = mcTriangleTable.triTable[cfg*16 + t*3 + 2];

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