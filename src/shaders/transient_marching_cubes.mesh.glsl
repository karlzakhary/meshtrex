#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_debug_printf: require

// --- Shader Configuration ---

// Workgroup size should match the task shader
#define WORKGROUP_SIZE 32

// Specialization constants for dynamic block dimensions
// These MUST be configured at pipeline creation time.
layout(constant_id = 0) const uint BX = 1u;
layout(constant_id = 1) const uint BY = 1u;
layout(constant_id = 2) const uint BZ = 1u;

// Total number of cells processed by this workgroup
const uint CELLS_PER_BLOCK = BX * BY * BZ;

// Maximum outputs per workgroup.
// The user MUST ensure that BX*BY*BZ*15 <= MAX_VERTS and BX*BY*BZ*5 <= MAX_PRIMS.
// We'll use 256 here as a common, safe upper bound, suitable for smaller block
// dimensions like 2x2x2 or 2x2x4.
const uint MAX_VERTS = 64u;
const uint MAX_PRIMS = 126u; // Each primitive is a triangle

// --- Mesh Shader Outputs ---
layout(triangles, max_vertices = MAX_VERTS, max_primitives = MAX_PRIMS) out;

// Per-vertex output data (position is built-in)
// This interface now matches the required fragment shader inputs.
layout(location = 0) out PerVertexData {
    vec3 fragNormal;
    vec3 fragPos;
} outVertices[];

// --- Payload from Task Shader ---
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
} taskPayloadIn;

// --- Descriptor Bindings (must match task shader) ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim; // Dimensions of a cell block (e.g., 8x8x8 cells)
    uvec4 blockGridDim; // Number of blocks in each dimension
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
// binding 2 (minMaxImage) is not used in the mesh shader
layout(set = 0, binding = 3, std430) readonly buffer ActiveBlockCount { uint count; } activeBlockCount;
layout(set = 0, binding = 4, std430) readonly buffer ActiveBlockIDs { uint ids[]; } activeBlockIDs;
layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesTriangleTable { int triTable[]; } mcTriangleTable;
layout(set = 0, binding = 6, std430) readonly buffer MarchingCubesEdgeTable { int edgeTable[]; } mcEdgeTable;

// Push constants for transformations
layout(push_constant) uniform RenderPushConstants {
    mat4 viewProj;
    vec4 frustumPlanes[6]; // Not used in mesh shader but part of the layout
} pushConsts;

// --- Shared Memory ---
// Used by the workgroup to aggregate geometry before final output
shared uint s_vertexCount;
shared uint s_primitiveCount;
shared vec3 s_vertexPositions[MAX_VERTS];
shared vec3 s_vertexNormals[MAX_VERTS];
shared uvec3 s_indices[MAX_PRIMS];

// --- Marching Cubes Tables & Helpers ---

// Defines the 8 corners of a cube relative to its origin
const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1)
);

// Maps an edge index (0-11) to the two corner indices it connects
const int edgeToCorner[12][2] = {
    {0,1}, {1,2}, {2,3}, {3,0},
    {4,5}, {5,6}, {6,7}, {7,4},
    {0,4}, {1,5}, {2,6}, {3,7}
};

uvec3 unpack_block_id(uint id) {
    return uvec3(
        id % ubo.blockGridDim.x,
        (id / ubo.blockGridDim.x) % ubo.blockGridDim.y,
        id / (ubo.blockGridDim.x * ubo.blockGridDim.y)
    );
}

float sample_volume(ivec3 coord) {
    // Clamp coordinates to stay within volume bounds
    return float(imageLoad(volumeImage, clamp(coord, ivec3(0), ivec3(ubo.volumeDim.xyz) - 1)).r);
}

// Linearly interpolates to find the vertex position along an edge
vec3 vertex_interp(vec3 p1, vec3 p2, float v1, float v2, float iso) {
    // Avoid division by zero on flat surfaces
    if (abs(v1 - v2) < 1e-6) {
        return p1;
    }

    // Calculate the interpolation factor
    float t = (iso - v1) / (v2 - v1);

    // CRITICAL FIX: Clamp 't' to the [0, 1] range to prevent extrapolation.
    t = clamp(t, 0.0, 1.0);
    
    return mix(p1, p2, t);
}

// Calculates the surface normal using the central differences method (gradient of the scalar field)
vec3 calculate_normal(vec3 pos) {
    vec3 h = vec3(1.0, 1.0, 1.0); // Voxel-sized step
    float nx = sample_volume(ivec3(pos - vec3(h.x, 0, 0))) - sample_volume(ivec3(pos + vec3(h.x, 0, 0)));
    float ny = sample_volume(ivec3(pos - vec3(0, h.y, 0))) - sample_volume(ivec3(pos + vec3(0, h.y, 0)));
    float nz = sample_volume(ivec3(pos - vec3(0, 0, h.z))) - sample_volume(ivec3(pos + vec3(0, 0, h.z)));
    return normalize(vec3(nx, ny, nz));
}

// --- Main Shader Logic ---
void main() {
    // Initialize shared counters once per workgroup
    if (gl_LocalInvocationID.x == 0) {
        s_vertexCount = 0;
        s_primitiveCount = 0;
    }
    barrier();

    // Get the ID for the block this workgroup will process
    uint blockID = taskPayloadIn.blockID;
    uvec3 blockCoord = unpack_block_id(blockID);
    ivec3 blockOrigin = ivec3(blockCoord * ubo.blockDim.xyz);
    

    // Each invocation processes a unique subset of cells within the block
    for (uint cellIndex = gl_LocalInvocationID.x; cellIndex < CELLS_PER_BLOCK; cellIndex += WORKGROUP_SIZE) {
        
        // Calculate cell's position
        uvec3 cellCoord_local = uvec3(cellIndex % BX, (cellIndex / BX) % BY, cellIndex / (BX * BY));
        ivec3 cellCoord_global = blockOrigin + ivec3(cellCoord_local);

        // 1. Calculate Cell Configuration
        uint configuration = 0;
        float cornerValues[8];
        for (int i = 0; i < 8; ++i) {
            cornerValues[i] = sample_volume(cellCoord_global + cornerOffset[i]);
            if (cornerValues[i] <= ubo.isovalue) {
                configuration |= (1 << i);
            }
        }

        if (configuration == 0 || configuration == 255) {
            continue;
        }

        // 2. Generate Vertices and Create Index Map
        uint edgeMask = uint(mcEdgeTable.edgeTable[configuration]);
        uint numCellVerts = bitCount(edgeMask);
        
        uint vertexBaseIndex = atomicAdd(s_vertexCount, numCellVerts);

        // This map is critical. It maps an edge index (0-11) to its local vertex index (0-14).
        int localIndex[12];

        // Generate vertices and populate the edge-to-rank map.
        for (int e = 0; e < 12; ++e) {
            if (((edgeMask >> e) & 1) != 0) {
                // This is the 'rank' - the local index of the vertex on this edge.
                int rank = bitCount(edgeMask & ((1u << e) - 1u));
                localIndex[e] = rank;

                int c1_idx = edgeToCorner[e][0];
                int c2_idx = edgeToCorner[e][1];
                vec3 p1 = vec3(cellCoord_global + cornerOffset[c1_idx]);
                vec3 p2 = vec3(cellCoord_global + cornerOffset[c2_idx]);
                vec3 vertPos = vertex_interp(p1, p2, cornerValues[c1_idx], cornerValues[c2_idx], ubo.isovalue);
                
                uint vertIndex = vertexBaseIndex + rank;
                s_vertexPositions[vertIndex] = vertPos;
                s_vertexNormals[vertIndex] = calculate_normal(vertPos);
            }
        }

        // 3. Generate Triangles using the Map
        uint numCellPrims = 0;
        int tri_table_start = int(configuration * 16);
        for (int i = 0; i < 5; ++i) {
             if (mcTriangleTable.triTable[tri_table_start + i*3] == -1) break;
             numCellPrims++;
        }

        uint primBaseIndex = atomicAdd(s_primitiveCount, numCellPrims);

        for (int i = 0; i < numCellPrims; ++i) {
            // Read the three EDGE indices for this triangle from the table.
            int e0 = mcTriangleTable.triTable[tri_table_start + i*3 + 0];
            int e1 = mcTriangleTable.triTable[tri_table_start + i*3 + 1];
            int e2 = mcTriangleTable.triTable[tri_table_start + i*3 + 2];
            
            // Use the map to find the local rank, then add the base offset.
            uint global_v1 = vertexBaseIndex + uint(localIndex[e0]);
            uint global_v2 = vertexBaseIndex + uint(localIndex[e1]);
            uint global_v3 = vertexBaseIndex + uint(localIndex[e2]);

            s_indices[primBaseIndex + i] = uvec3(global_v1, global_v2, global_v3);
        }
    }

    barrier();

    // --- 4. Final Output ---
    if (gl_LocalInvocationID.x == 0) {
        SetMeshOutputsEXT(s_vertexCount, s_primitiveCount);
    }
    
    // Only thread 0 outputs the vertices to avoid race conditions
    if (gl_LocalInvocationID.x == 0) {
        for (uint i = 0; i < s_vertexCount; i++) {
            vec3 worldPos = s_vertexPositions[i];
            
            // Center the volume around origin
            vec3 centeredPos = worldPos - vec3(ubo.volumeDim.xyz) * 0.5;
            

            // Built-in clip-space position
            vec4 clipPos = pushConsts.viewProj * vec4(centeredPos, 1.0);
            gl_MeshVerticesEXT[i].gl_Position = clipPos;
            
            // Custom outputs for the fragment shader
            outVertices[i].fragNormal = s_vertexNormals[i];
            outVertices[i].fragPos = worldPos;
        }

        for (uint i = 0; i < s_primitiveCount; i++) {
            gl_PrimitiveTriangleIndicesEXT[i] = s_indices[i];
        }
    }
}