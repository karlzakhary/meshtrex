#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// Block dimensions optimized for 64-vertex limit
// 4×4×4 = 64 cells, typical ~50-80 vertices
#define CORE_BLOCK_DIM_X 4
#define CORE_BLOCK_DIM_Y 4  
#define CORE_BLOCK_DIM_Z 4

// Mesh shader limits - following NVIDIA's recommendations
#define MAX_MESHLET_VERTS 64u   // NVIDIA recommended for optimal performance
#define MAX_MESHLET_PRIMS 126u  // NVIDIA recommended (unchanged)

// MC constants
#define MAX_TRI_INDICES 16
#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5

// Global edge map size (adjust based on volume size)
// For a 1024^3 volume with 8x4x4 blocks, this should be sufficient
#define GLOBAL_EDGE_MAP_SIZE 16777216u  // 16M entries
#define INVALID_VERTEX_ID 0xFFFFFFFFu
#define CREATING_VERTEX_ID 0xFFFFFFFEu

// Workgroup sizes - NVIDIA recommended (32 = 1 warp)
#define TASK_WORKGROUP_SIZE 32
#define MESH_WORKGROUP_SIZE 32  // Optimal for mesh shaders

// Structures shared with C++
struct VertexData {
    vec4 position;  // xyz = position, w = 1.0
    vec4 normal;    // xyz = normal, w = 0.0
};

struct TaskPayload {
    uvec3 blockOrigin;      // Global origin of the block in voxel coordinates
    uvec3 blockDim;         // Dimensions of the block to process
    uint originalBlockId;   // Original block index from compacted array
    uint level;             // Subdivision level (0 = full block, 1 = half, 2 = quarter)
};

// Edge vertex offsets for standard MC edges (0-11)
const ivec3 MC_EDGE_VERTICES[12][2] = ivec3[12][2](
    ivec3[2](ivec3(0,0,0), ivec3(1,0,0)), // edge 0
    ivec3[2](ivec3(1,0,0), ivec3(1,1,0)), // edge 1
    ivec3[2](ivec3(0,1,0), ivec3(1,1,0)), // edge 2
    ivec3[2](ivec3(0,0,0), ivec3(0,1,0)), // edge 3
    ivec3[2](ivec3(0,0,1), ivec3(1,0,1)), // edge 4
    ivec3[2](ivec3(1,0,1), ivec3(1,1,1)), // edge 5
    ivec3[2](ivec3(0,1,1), ivec3(1,1,1)), // edge 6
    ivec3[2](ivec3(0,0,1), ivec3(0,1,1)), // edge 7
    ivec3[2](ivec3(0,0,0), ivec3(0,0,1)), // edge 8
    ivec3[2](ivec3(1,0,0), ivec3(1,0,1)), // edge 9
    ivec3[2](ivec3(1,1,0), ivec3(1,1,1)), // edge 10
    ivec3[2](ivec3(0,1,0), ivec3(0,1,1))  // edge 11
);

// Neighbor mapping table from PMB paper (Table 1)
// Maps MC edge ID to neighbor cell offset and distinct edge
// xyz = neighbor offset, w = distinct edge (0=x, 1=y, 2=z)
const uvec4 NEIGHBOR_MAPPING_TABLE[12] = uvec4[12](
    uvec4(0, 0, 0, 0), // edge 0: x-edge at origin
    uvec4(1, 0, 0, 1), // edge 1: y-edge at (1,0,0)
    uvec4(0, 1, 0, 0), // edge 2: x-edge at (0,1,0)
    uvec4(0, 0, 0, 1), // edge 3: y-edge at origin
    uvec4(0, 0, 1, 0), // edge 4: x-edge at (0,0,1)
    uvec4(1, 0, 1, 1), // edge 5: y-edge at (1,0,1)
    uvec4(0, 1, 1, 0), // edge 6: x-edge at (0,1,1)
    uvec4(0, 0, 1, 1), // edge 7: y-edge at (0,0,1)
    uvec4(0, 0, 0, 2), // edge 8: z-edge at origin
    uvec4(1, 0, 0, 2), // edge 9: z-edge at (1,0,0)
    uvec4(1, 1, 0, 2), // edge 10: z-edge at (1,1,0)
    uvec4(0, 1, 0, 2)  // edge 11: z-edge at (0,1,0)
);

// Binding points
#define BINDING_PUSH_CONSTANTS     0
#define BINDING_VOLUME_IMAGE       1
#define BINDING_ACTIVE_BLOCK_COUNT 2
#define BINDING_COMPACTED_BLOCKS   3
#define BINDING_MC_TRI_TABLE       4
#define BINDING_GLOBAL_VERTICES    6
#define BINDING_GLOBAL_INDICES     8
#define BINDING_MESHLET_DESC       10
#define BINDING_VERTEX_COUNTER     11
#define BINDING_MESHLET_COUNTER    12
#define BINDING_INDEX_COUNTER      13
#define BINDING_GLOBAL_EDGE_MAP    14

taskPayloadSharedEXT TaskPayload taskPayloadIn;

// Bindings
layout(set = 0, binding = BINDING_PUSH_CONSTANTS, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = BINDING_VOLUME_IMAGE, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = BINDING_MC_TRI_TABLE, std430) readonly buffer MarchingCubesTriTable_SSBO { 
    int triTable[]; 
} mc;

layout(set = 0, binding = BINDING_GLOBAL_VERTICES, std430) buffer GlobalVertexBuffer_SSBO { 
    VertexData vertex_data[]; 
} globalVerticesSSBO;

layout(set = 0, binding = BINDING_GLOBAL_INDICES, std430) buffer GlobalIndexBuffer_SSBO { 
    uint indices[]; 
} globalIndicesSSBO;

layout(set = 0, binding = BINDING_VERTEX_COUNTER, std430) buffer GlobalVertexIDCounter_SSBO { 
    uint counter; 
} globalVertexIDCounter;

layout(set = 0, binding = BINDING_INDEX_COUNTER, std430) buffer GlobalIndexOutputCount_SSBO { 
    uint counter; 
} globalIndexOutputCount;

// Meshlet descriptors
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

layout(set = 0, binding = 10, std430) buffer MeshletDescOutput_SSBO { 
    MeshletDescriptor meshletDescriptors[]; 
} meshletDescOutput;

layout(set = 0, binding = 12, std430) buffer FilledMeshletDescCount_SSBO { 
    uint filledMeshletCounter; 
} filledMeshletDescCount;

layout(local_size_x = MESH_WORKGROUP_SIZE) in;
layout(max_vertices = 64, max_primitives = 126) out;
layout(triangles) out;

// Shared memory
shared float voxelValues[CORE_BLOCK_DIM_X + 1][CORE_BLOCK_DIM_Y + 1][CORE_BLOCK_DIM_Z + 1];
shared uint totalVertices;
shared uint totalTriangles;
shared uint vertexOffset;
shared uint indexOffset;

// Edge table for standard MC - which edges are cut by the isosurface
const int edgeTable[256] = int[256](
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55,  0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55,  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,  0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
);

// Map from edge index to the two corner indices it connects
const ivec2 edgeToCorners[12] = ivec2[12](
    ivec2(0, 1), // edge 0
    ivec2(1, 2), // edge 1
    ivec2(2, 3), // edge 2
    ivec2(3, 0), // edge 3
    ivec2(4, 5), // edge 4
    ivec2(5, 6), // edge 5
    ivec2(6, 7), // edge 6
    ivec2(7, 4), // edge 7
    ivec2(0, 4), // edge 8
    ivec2(1, 5), // edge 9
    ivec2(2, 6), // edge 10
    ivec2(3, 7)  // edge 11
);

// Helper functions
vec3 interpolateVertex(vec3 p0, vec3 p1, float val0, float val1) {
    if (abs(val0 - val1) < 1e-6f) return p0;
    float t = clamp((ubo.isovalue - val0) / (val1 - val0), 0.0f, 1.0f);
    return mix(p0, p1, t);
}

vec3 calculateNormal(vec3 pos) {
    ivec3 ipos = ivec3(round(pos));
    ivec3 volMax = ivec3(ubo.volumeDim.xyz) - 1;
    
    float dx = float(imageLoad(volumeImage, clamp(ipos + ivec3(1,0,0), ivec3(0), volMax)).r) -
               float(imageLoad(volumeImage, clamp(ipos - ivec3(1,0,0), ivec3(0), volMax)).r);
    float dy = float(imageLoad(volumeImage, clamp(ipos + ivec3(0,1,0), ivec3(0), volMax)).r) -
               float(imageLoad(volumeImage, clamp(ipos - ivec3(0,1,0), ivec3(0), volMax)).r);
    float dz = float(imageLoad(volumeImage, clamp(ipos + ivec3(0,0,1), ivec3(0), volMax)).r) -
               float(imageLoad(volumeImage, clamp(ipos - ivec3(0,0,1), ivec3(0), volMax)).r);
    
    vec3 grad = vec3(dx, dy, dz);
    if (length(grad) < 1e-5f) return vec3(0, 1, 0);
    return -normalize(grad);
}

uint warpScan(uint val) {
    uint laneID = gl_LocalInvocationIndex & 31;
    for (uint i = 1; i < 32; i *= 2) {
        uint n = subgroupShuffleUp(val, i);
        if (laneID >= i) val += n;
    }
    return val;
}

void main() {
    uvec3 blockOrigin = taskPayloadIn.blockOrigin;
    uvec3 blockDim = min(taskPayloadIn.blockDim, 
                         uvec3(CORE_BLOCK_DIM_X, CORE_BLOCK_DIM_Y, CORE_BLOCK_DIM_Z));
    
    // Initialize shared memory
    if (gl_LocalInvocationIndex == 0) {
        totalVertices = 0;
        totalTriangles = 0;
        vertexOffset = 0;
        indexOffset = 0;
    }
    
    barrier();
    
    // Phase 1: Cache voxel values
    uint voxelsToLoad = (blockDim.x + 1) * (blockDim.y + 1) * (blockDim.z + 1);
    for (uint i = gl_LocalInvocationIndex; i < voxelsToLoad; i += MESH_WORKGROUP_SIZE) {
        uint x = i % (blockDim.x + 1);
        uint y = (i / (blockDim.x + 1)) % (blockDim.y + 1);
        uint z = i / ((blockDim.x + 1) * (blockDim.y + 1));
        
        ivec3 voxelPos = ivec3(blockOrigin) + ivec3(x, y, z);
        float val = 0.0;
        
        if (all(greaterThanEqual(voxelPos, ivec3(0))) && 
            all(lessThan(voxelPos, ivec3(ubo.volumeDim.xyz)))) {
            val = float(imageLoad(volumeImage, voxelPos).r);
        }
        
        voxelValues[x][y][z] = val;
    }
    
    barrier();
    
    // Phase 2: Count vertices and triangles needed
    uint myVertexCount = 0;
    uint myTriangleCount = 0;
    
    uint cellsInBlock = blockDim.x * blockDim.y * blockDim.z;
    for (uint cellIdx = gl_LocalInvocationIndex; cellIdx < cellsInBlock; cellIdx += MESH_WORKGROUP_SIZE) {
        uvec3 localCell;
        localCell.x = cellIdx % blockDim.x;
        localCell.y = (cellIdx / blockDim.x) % blockDim.y;
        localCell.z = cellIdx / (blockDim.x * blockDim.y);
        
        // Calculate cube case
        uint cubeCase = 0;
        for (int i = 0; i < 8; ++i) {
            ivec3 offset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            float val = voxelValues[localCell.x + offset.x][localCell.y + offset.y][localCell.z + offset.z];
            if (val >= ubo.isovalue) {
                cubeCase |= (1u << i);
            }
        }
        
        if (cubeCase == 0 || cubeCase == 255) continue;
        
        // Count vertices needed
        int edgeFlags = edgeTable[cubeCase];
        uint vertCount = 0;
        for (int i = 0; i < 12; ++i) {
            if ((edgeFlags & (1 << i)) != 0) {
                vertCount++;
            }
        }
        myVertexCount += vertCount;
        
        // Count triangles
        int base = int(cubeCase) * 16;
        uint triCount = 0;
        for (int i = 0; i < 16; i += 3) {
            if (mc.triTable[base + i] < 0) break;
            triCount++;
        }
        myTriangleCount += triCount;
    }
    
    // Prefix sum to get offsets
    uint myVertexOffset = warpScan(myVertexCount);
    uint myTriangleOffset = warpScan(myTriangleCount);
    
    // Last thread gets total counts
    if (gl_LocalInvocationIndex == MESH_WORKGROUP_SIZE - 1) {
        totalVertices = myVertexOffset;
        totalTriangles = myTriangleOffset;
        
        if (myVertexOffset > 0) {
            vertexOffset = atomicAdd(globalVertexIDCounter.counter, myVertexOffset);
        }
        if (myTriangleOffset > 0) {
            indexOffset = atomicAdd(globalIndexOutputCount.counter, myTriangleOffset * 3);
        }
    }
    
    barrier();
    
    // Phase 3: Generate vertices and triangles
    uint currentVertexOffset = myVertexOffset - myVertexCount;
    uint currentTriangleOffset = myTriangleOffset - myTriangleCount;
    
    for (uint cellIdx = gl_LocalInvocationIndex; cellIdx < cellsInBlock; cellIdx += MESH_WORKGROUP_SIZE) {
        uvec3 localCell;
        localCell.x = cellIdx % blockDim.x;
        localCell.y = (cellIdx / blockDim.x) % blockDim.y;
        localCell.z = cellIdx / (blockDim.x * blockDim.y);
        
        ivec3 globalCell = ivec3(blockOrigin) + ivec3(localCell);
        
        // Get corner values
        float cornerValues[8];
        uint cubeCase = 0;
        for (int i = 0; i < 8; ++i) {
            ivec3 offset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            cornerValues[i] = voxelValues[localCell.x + offset.x][localCell.y + offset.y][localCell.z + offset.z];
            if (cornerValues[i] >= ubo.isovalue) {
                cubeCase |= (1u << i);
            }
        }
        
        if (cubeCase == 0 || cubeCase == 255) continue;
        
        // Generate vertices for this cell
        uint cellVertexIDs[12];
        for (int i = 0; i < 12; ++i) {
            cellVertexIDs[i] = INVALID_VERTEX_ID;
        }
        
        int edgeFlags = edgeTable[cubeCase];
        uint localVertexCounter = 0;
        
        for (int edgeIdx = 0; edgeIdx < 12; ++edgeIdx) {
            if ((edgeFlags & (1 << edgeIdx)) != 0) {
                // Get the two corners this edge connects
                ivec2 corners = edgeToCorners[edgeIdx];
                int c0 = corners.x;
                int c1 = corners.y;
                
                // Get vertex positions for the two corners
                ivec3 p0 = globalCell + ivec3((c0 & 1), (c0 & 2) >> 1, (c0 & 4) >> 2);
                ivec3 p1 = globalCell + ivec3((c1 & 1), (c1 & 2) >> 1, (c1 & 4) >> 2);
                
                // Interpolate vertex position
                vec3 pos = interpolateVertex(vec3(p0), vec3(p1), cornerValues[c0], cornerValues[c1]);
                vec3 normal = calculateNormal(pos);
                
                // Store vertex
                uint globalVertexID = vertexOffset + currentVertexOffset + localVertexCounter;
                
                // Debug check
                if (any(greaterThan(abs(pos), vec3(10000.0)))) {
                    debugPrintfEXT("ERROR: Huge vertex position: (%f,%f,%f) at cell (%d,%d,%d)\n",
                                   pos.x, pos.y, pos.z, globalCell.x, globalCell.y, globalCell.z);
                }
                
                globalVerticesSSBO.vertex_data[globalVertexID].position = vec4(pos, 1.0);
                globalVerticesSSBO.vertex_data[globalVertexID].normal = vec4(normal, 0.0);
                
                cellVertexIDs[edgeIdx] = globalVertexID;
                localVertexCounter++;
            }
        }
        
        currentVertexOffset += localVertexCounter;
        
        // Generate triangles
        int base = int(cubeCase) * 16;
        for (int tri = 0; tri < 5; ++tri) {
            int e0 = mc.triTable[base + tri * 3 + 0];
            if (e0 < 0 || e0 >= 12) break;
            
            int e1 = mc.triTable[base + tri * 3 + 1];
            int e2 = mc.triTable[base + tri * 3 + 2];
            
            if (e1 < 0 || e1 >= 12 || e2 < 0 || e2 >= 12) break;
            
            uint globalIndexOffset = indexOffset + (currentTriangleOffset * 3);
            globalIndicesSSBO.indices[globalIndexOffset + 0] = cellVertexIDs[e0];
            globalIndicesSSBO.indices[globalIndexOffset + 1] = cellVertexIDs[e1];
            globalIndicesSSBO.indices[globalIndexOffset + 2] = cellVertexIDs[e2];
            
            currentTriangleOffset++;
        }
    }
    
    barrier();
    
    // Write meshlet descriptor
    if (gl_LocalInvocationIndex == 0 && totalTriangles > 0) {
        uint descIndex = atomicAdd(filledMeshletDescCount.filledMeshletCounter, 1u);
        
        meshletDescOutput.meshletDescriptors[descIndex].vertexOffset = vertexOffset;
        meshletDescOutput.meshletDescriptors[descIndex].indexOffset = indexOffset / 3;
        meshletDescOutput.meshletDescriptors[descIndex].vertexCount = totalVertices;
        meshletDescOutput.meshletDescriptors[descIndex].primitiveCount = totalTriangles;
    }
    
    SetMeshOutputsEXT(0, 0);
}