#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_ballot: require

// --- Configurable Parameters ---
#define WORKGROUP_SIZE 128u
/* core-cell grid ---------------------------------------------------- */
#define BX 4u
#define BY 4u
#define BZ 4u
#define CELLS_PER_BLOCK 64u  /* 4×4×4 */

/* voxel region you must read (core + 1-voxel halo) ----------------- */
#define BLOCK_DIM_X 5u
#define BLOCK_DIM_Y 5u
#define BLOCK_DIM_Z 5u
#define STRIDE      4u           /* overlap = 1 voxel */
#define MAX_PRIMS_PER_CELL 5u
#define MAX_CELLS_IN_BLOCK 64u
#define MAX_CELLS_PER_THREAD (MAX_CELLS_IN_BLOCK + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
#define MAX_PRIMS_PER_THREAD (MAX_CELLS_PER_THREAD * MAX_PRIMS_PER_CELL)
#define MAX_VERTS_PER_MESHLET 256u
#define MAX_PRIMS_PER_MESHLET 256u

/* ------------------  PMB edge ownership -------------------------- */
const uint PMB_EDGE_X = 0u;
const uint PMB_EDGE_Y = 3u;
const uint PMB_EDGE_Z = 8u;

/* neighbour mapping: <dx,dy,dz,edgeType(0=x 1=y 2=z)>               */
const uvec4 edgeOwner[12] = uvec4[12](
  uvec4(0,0,0,0), uvec4(1,0,0,1), uvec4(0,1,0,0), uvec4(0,0,0,1),
  uvec4(0,0,1,0), uvec4(1,0,1,1), uvec4(0,1,1,0), uvec4(0,0,1,1),
  uvec4(0,0,0,2), uvec4(1,0,0,2), uvec4(1,1,0,2), uvec4(0,1,0,2));

// --- Structures ---
struct VertexData {
    vec4 position;
    vec4 normal;
};

struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

#define MAX_MESHLETS_PER_BLOCK 8u
taskPayloadSharedEXT struct TaskPayload {
    uint blockID;
    uint meshletCount;
    uint firstCell[MAX_MESHLETS_PER_BLOCK];
    uint cellCount[MAX_MESHLETS_PER_BLOCK];
    uint packedCellData[BX * BY * BZ];
    uvec3 pageCoord;
    uint mipLevel;
} taskPayloadIn;

// --- Push Constants ---
layout(push_constant) uniform PushConstants {
    uvec3 pageCoord;      // Page coordinate for this pass
    uint mipLevel;        // Mip level
    float isoValue;       // Isovalue for filtering
    uint blockSize;       // Block size (typically 4)
    uint pageSizeX;       // Page size X (64)
    uint pageSizeY;       // Page size Y (32)
    uint pageSizeZ;       // Page size Z (32)
    // Global buffer offsets for persistent geometry extraction
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletOffset;
    uint volumeSizeX;     // Full volume dimension X
    uint volumeSizeY;     // Full volume dimension Y
    uint volumeSizeZ;     // Full volume dimension Z
    uint padding[2];
} pc;

// --- Descriptor Set Bindings ---
layout(set = 0, binding = 0, std430) readonly buffer PageTable {
    uvec2 pageEntries[];
} pageTable;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeAtlas;

layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriangleTable { 
    int triTable[]; 
} mcTriangleTable;

layout(set = 0, binding = 5, std430) readonly buffer MarchingCubesEdgeTable { 
    int edgeTable[]; 
} mcEdgeTable;

layout(set = 0, binding = 6, std430) buffer VertexBuffer { 
    VertexData data[]; 
} vertices;

layout(set = 0, binding = 7, std430) buffer VertexCount { 
    uint vertexCounter; 
} vCount;

layout(set = 0, binding = 8, std430) buffer IndexBuffer { 
    uint data[]; 
} indices;

layout(set = 0, binding = 9, std430) buffer IndexCount { 
    uint indexCounter; 
} iCount;

layout(set = 0, binding = 10, std430) buffer MeshletDescriptorBuffer { 
    MeshletDescriptor descriptors[]; 
} meshlets;

layout(set = 0, binding = 11, std430) buffer MeshletDescriptorCount { 
    uint meshletCounter; 
} meshletCount;

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

// --- Helper Functions ---

uint getPageIndex(uvec3 pageCoord, uint mipLevel) {
    // Calculate based on actual volume dimensions from push constants
    // This needs to match the CPU-side calculation in streamingSystem.cpp
    uint pagesX = (pc.volumeSizeX + pc.pageSizeX - 1) / pc.pageSizeX;
    uint pagesY = (pc.volumeSizeY + pc.pageSizeY - 1) / pc.pageSizeY;
    return pageCoord.z * pagesX * pagesY + pageCoord.y * pagesX + pageCoord.x;
}

bool isPageResident(uvec3 pageCoord, uint mipLevel) {
    uint pageIndex = getPageIndex(pageCoord, mipLevel);
    
    // Debug page residency check for specific coordinates
    if (pageCoord.x == 2 && pageCoord.y == 1 && pageCoord.z == 2) {
        uint pageTableLength = pageTable.pageEntries.length();
        uint isResidentFlag = (pageIndex < pageTableLength) ? pageTable.pageEntries[pageIndex].y : 999;
        debugPrintfEXT("Mesh isPageResident: pageCoord(%d,%d,%d) pageIndex=%d, tableLength=%d, resident=%d",
                      pageCoord.x, pageCoord.y, pageCoord.z, pageIndex, pageTableLength, isResidentFlag);
    }
    
    if (pageIndex >= pageTable.pageEntries.length()) return false;
    return pageTable.pageEntries[pageIndex].y != 0;
}

uvec3 getAtlasCoord(uvec3 pageCoord, uint mipLevel) {
    uint pageIndex = getPageIndex(pageCoord, mipLevel);
    uint atlasCoord = pageTable.pageEntries[pageIndex].x;
    
    return uvec3(
        atlasCoord & 0x3FF,
        (atlasCoord >> 10) & 0x3FF,
        (atlasCoord >> 20) & 0x3FF
    );
}

uint sampleVolumeAtlas(uvec3 worldCoord) {
    uvec3 pageCoord = uvec3(worldCoord.x / pc.pageSizeX, worldCoord.y / pc.pageSizeY, worldCoord.z / pc.pageSizeZ);
    
    bool isResident = isPageResident(pageCoord, pc.mipLevel);
    
    // Debug specific coordinates
    if (worldCoord.x >= 128 && worldCoord.x <= 129 && worldCoord.y == 32 && worldCoord.z == 78) {
        debugPrintfEXT("Mesh sample: world(%d,%d,%d) page(%d,%d,%d) resident=%d",
                      worldCoord.x, worldCoord.y, worldCoord.z,
                      pageCoord.x, pageCoord.y, pageCoord.z, isResident ? 1 : 0);
    }
    
    if (!isResident) {
        return 0;
    }
    
    uvec3 atlasCoord = getAtlasCoord(pageCoord, pc.mipLevel);
    uvec3 localCoord = uvec3(worldCoord.x % pc.pageSizeX, worldCoord.y % pc.pageSizeY, worldCoord.z % pc.pageSizeZ);
    
    // atlasCoord from page table is in granularity units (divided by granularity during packing)
    // We need to multiply by granularity to get back to texel coordinates
    ivec3 atlasTexel = ivec3(atlasCoord.x * 64 + localCoord.x,
                             atlasCoord.y * 32 + localCoord.y,
                             atlasCoord.z * 32 + localCoord.z);
    
    // Use imageLoad for integer textures
    uint value = imageLoad(volumeAtlas, atlasTexel).r;
    
    // Debug atlas access
    if (worldCoord.x >= 128 && worldCoord.x <= 129 && worldCoord.y == 32 && worldCoord.z == 78) {
        debugPrintfEXT("atlasCoord(%d,%d,%d) localCoord(%d,%d,%d) atlasTexel(%d,%d,%d) value=%d",
                      atlasCoord.x, atlasCoord.y, atlasCoord.z,
                      localCoord.x, localCoord.y, localCoord.z,
                      atlasTexel.x, atlasTexel.y, atlasTexel.z, value);
    }
    
    return value;
}

uint getPrimitiveCount(uint configuration) {
    uint primitiveCount = 0;
    for (int i = 0; i < 5; i++) {
        if (mcTriangleTable.triTable[configuration * 16 + i * 3] == -1) break;
        primitiveCount++;
    }
    return primitiveCount;
}

uvec3 unpack_block_id(uint id) {
    // For streaming, calculate blocks within the current page
    uint blocksPerPageX = pc.pageSizeX / pc.blockSize;
    uint blocksPerPageY = pc.pageSizeY / pc.blockSize;
    uint blocksPerPageZ = pc.pageSizeZ / pc.blockSize;
    uint grid_slice = blocksPerPageX * blocksPerPageY;
    return uvec3(
        id % blocksPerPageX, 
        (id / blocksPerPageX) % blocksPerPageY, 
        id / grid_slice
    );
}

// Calculate normal using volume atlas sampling
vec3 calculate_normal_streaming(ivec3 p) {
    float s1 = float(sampleVolumeAtlas(uvec3(max(p + ivec3(-1, 0, 0), ivec3(0)))));
    float s2 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 1, 0, 0), ivec3(0)))));
    float s3 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0,-1, 0), ivec3(0)))));
    float s4 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0, 1, 0), ivec3(0)))));
    float s5 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0, 0,-1), ivec3(0)))));
    float s6 = float(sampleVolumeAtlas(uvec3(max(p + ivec3( 0, 0, 1), ivec3(0)))));
    return normalize(vec3(s1 - s2, s3 - s4, s5 - s6));
}

VertexData interpolate_vertex_streaming(float isolevel, ivec3 p1_coord, ivec3 p2_coord) {
    float v1_val = float(sampleVolumeAtlas(uvec3(max(p1_coord, ivec3(0)))));
    float v2_val = float(sampleVolumeAtlas(uvec3(max(p2_coord, ivec3(0)))));

    vec3 n1 = calculate_normal_streaming(p1_coord);
    vec3 n2 = calculate_normal_streaming(p2_coord);

    float mu = 0.5;
    float denominator = v2_val - v1_val;
    if (abs(denominator) > 0.00001) {
        mu = (isolevel - v1_val) / denominator;
    }
    mu = clamp(mu, 0.0, 1.0);
    
    vec3 pos = mix(vec3(p1_coord), vec3(p2_coord), mu);
    vec3 norm = normalize(mix(n1, n2, mu));
    
    // Normalize to [-1, 1] range - using full volume dimensions from push constants
    vec3 volumeSize = vec3(float(pc.volumeSizeX), float(pc.volumeSizeY), float(pc.volumeSizeZ));
    vec3 final_pos = (pos / volumeSize) * 2.0 - 1.0;

    return VertexData(vec4(final_pos, 1.0), vec4(norm, 0.0));
}

// Computes the marching cubes configuration index for a cell
uint calculate_configuration_streaming(ivec3 cell_coord_global) {
    uint configuration = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 corner_offset = cornerOffset[i];
        ivec3 neighbor_coord = cell_coord_global + corner_offset;
        uint value = sampleVolumeAtlas(uvec3(max(neighbor_coord, ivec3(0))));
        
        // Debug sampling for specific cell
        if (cell_coord_global == ivec3(128, 32, 78) && i < 2) {
            debugPrintfEXT("Mesh config: corner[%d] at (%d,%d,%d) value=%d, iso=%f, test=%d",
                          i, neighbor_coord.x, neighbor_coord.y, neighbor_coord.z, 
                          value, pc.isoValue, (float(value) <= pc.isoValue) ? 1 : 0);
        }
        
        if (float(value) <= pc.isoValue) {
            configuration |= (1 << i);
        }
    }
    return configuration;
}

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Output limits ---
layout(max_vertices = 256, max_primitives = 256) out;
layout(triangles) out;

void main()
{
    const uint lane = gl_LocalInvocationIndex;
    const uint meshletID = gl_WorkGroupID.x;
    
    if (meshletID >= taskPayloadIn.meshletCount) {
        return;
    }

    uint blockID = taskPayloadIn.blockID;
    uvec3 blockCoord = unpack_block_id(blockID);
    
    // Debug block and page info
    if (gl_WorkGroupID.x == 0 && lane == 0) {
        // debugPrintfEXT("Mesh shader: blockID=%d, blockCoord=(%d,%d,%d), pageCoord=(%d,%d,%d), pageSize=(%d,%d,%d)",
        //               blockID, blockCoord.x, blockCoord.y, blockCoord.z,
        //               taskPayloadIn.pageCoord.x, taskPayloadIn.pageCoord.y, taskPayloadIn.pageCoord.z,
        //               pc.pageSizeX, pc.pageSizeY, pc.pageSizeZ);
    }
    
    // Calculate world coordinates for this block
    ivec3 pageStart = ivec3(taskPayloadIn.pageCoord.x * pc.pageSizeX,
                            taskPayloadIn.pageCoord.y * pc.pageSizeY,
                            taskPayloadIn.pageCoord.z * pc.pageSizeZ);
    ivec3 base = pageStart + ivec3(blockCoord * pc.blockSize);

    uint firstCell = taskPayloadIn.firstCell[meshletID];
    uint cellCount = taskPayloadIn.cellCount[meshletID];

    // Process cells in this meshlet
    for (uint cellIdx = firstCell + lane; cellIdx < firstCell + cellCount; cellIdx += WORKGROUP_SIZE)
    {
        if (cellIdx >= CELLS_PER_BLOCK) continue;
        
        uint packed_data = taskPayloadIn.packedCellData[cellIdx];
        uint cell = (packed_data >> 16) & 0xFFFFu;
        uint prims = (packed_data >> 8) & 0xFFu;
        uint owner_verts = packed_data & 0xFFu;

        uvec3 cLoc = uvec3(cell % BX, (cell / BX) % BY, cell / (BX * BY));
        ivec3 cellWorldCoord = base + ivec3(cLoc);
        
        // Debug cell unpacking
        if (gl_WorkGroupID.x == 0 && cellIdx < firstCell + 2) {
            // debugPrintfEXT("Mesh unpacking: cellIdx=%d, cell=%d, cLoc=(%d,%d,%d), base=(%d,%d,%d), world=(%d,%d,%d)",
            //               cellIdx, cell, cLoc.x, cLoc.y, cLoc.z,
            //               base.x, base.y, base.z,
            //               cellWorldCoord.x, cellWorldCoord.y, cellWorldCoord.z);
        }
        
        // Skip page residency check - the task shader already verified these cells are resident
        // All cells passed from task shader should be in the current page being processed

        uint cfg = calculate_configuration_streaming(cellWorldCoord);
        uint eMask = mcEdgeTable.edgeTable[cfg];
        
        // Debug: Check if configuration matches what task shader found
        if (gl_WorkGroupID.x == 0 && cellIdx < firstCell + 3) {
            uint recalcPrims = 0;
            for (int i = 0; i < 5; i++) {
                if (mcTriangleTable.triTable[cfg * 16 + i * 3] == -1) break;
                recalcPrims++;
            }
            // debugPrintfEXT("Mesh shader: Cell at world(%d,%d,%d) - task prims=%d, recalc cfg=%d, recalc prims=%d, eMask=%d", 
            //               cellWorldCoord.x, cellWorldCoord.y, cellWorldCoord.z, prims, cfg, recalcPrims, eMask);
        }

        if (eMask == 0u || prims == 0u) continue;
        
        // Debug output for first workgroup
        if (gl_WorkGroupID.x == 0 && cellIdx < 2) {
            debugPrintfEXT("Mesh shader: Cell %d, cfg=%d, prims=%d, processing triangles", 
                          cell, cfg, prims);
        }

        // Process triangles for this cell
        for (uint triIdx = 0; triIdx < prims; triIdx++)
        {
            int e0 = mcTriangleTable.triTable[cfg * 16 + triIdx * 3 + 0];
            int e1 = mcTriangleTable.triTable[cfg * 16 + triIdx * 3 + 1];
            int e2 = mcTriangleTable.triTable[cfg * 16 + triIdx * 3 + 2];

            if (e0 == -1 || e1 == -1 || e2 == -1) {
                if (gl_WorkGroupID.x == 0 && cellIdx < 2) {
                    debugPrintfEXT("Mesh shader: Triangle %d has invalid edge (-1), breaking", triIdx);
                }
                break;
            }

            // Generate vertices for edges with global buffer offsets
            uint baseVertexIndex = pc.globalVertexOffset + atomicAdd(vCount.vertexCounter, 3);
            uint baseIndex = pc.globalIndexOffset + atomicAdd(iCount.indexCounter, 3);
            
            // Debug atomic operations - show which thread is generating triangles
            if (gl_WorkGroupID.x < 2 && triIdx == 0 && cellIdx < firstCell + 2) {
                debugPrintfEXT("Mesh: WG=%d lane=%d meshlet=%d cell=%d cfg=%d baseVertex=%d", 
                              gl_WorkGroupID.x, lane, meshletID, cell, cfg, baseVertexIndex);
            }

            // Generate proper vertices using edge interpolation (PMB implementation)
            // Edge-to-vertex mapping for marching cubes
            const ivec2 edgeToVertex[12] = ivec2[12](
                ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0),  // bottom face
                ivec2(4,5), ivec2(5,6), ivec2(6,7), ivec2(7,4),  // top face  
                ivec2(0,4), ivec2(1,5), ivec2(2,6), ivec2(3,7)   // vertical edges
            );
            
            ivec3 e0_coords[2], e1_coords[2], e2_coords[2];
            
            // Edge 0
            e0_coords[0] = cellWorldCoord + cornerOffset[edgeToVertex[e0].x];
            e0_coords[1] = cellWorldCoord + cornerOffset[edgeToVertex[e0].y];
            
            // Edge 1  
            e1_coords[0] = cellWorldCoord + cornerOffset[edgeToVertex[e1].x];
            e1_coords[1] = cellWorldCoord + cornerOffset[edgeToVertex[e1].y];
            
            // Edge 2
            e2_coords[0] = cellWorldCoord + cornerOffset[edgeToVertex[e2].x];
            e2_coords[1] = cellWorldCoord + cornerOffset[edgeToVertex[e2].y];

            VertexData v0_data = interpolate_vertex_streaming(pc.isoValue, e0_coords[0], e0_coords[1]);
            VertexData v1_data = interpolate_vertex_streaming(pc.isoValue, e1_coords[0], e1_coords[1]);
            VertexData v2_data = interpolate_vertex_streaming(pc.isoValue, e2_coords[0], e2_coords[1]);

            vertices.data[baseVertexIndex + 0] = v0_data;
            vertices.data[baseVertexIndex + 1] = v1_data;
            vertices.data[baseVertexIndex + 2] = v2_data;

            indices.data[baseIndex + 0] = baseVertexIndex + 0;
            indices.data[baseIndex + 1] = baseVertexIndex + 1;
            indices.data[baseIndex + 2] = baseVertexIndex + 2;
        }
    }

    // Update meshlet descriptor with global offsets
    if (lane == 0)
    {
        uint meshletIndex = pc.globalMeshletOffset + atomicAdd(meshletCount.meshletCounter, 1);
        
        // Calculate actual vertex and index counts for this meshlet
        uint totalVertices = 0;
        uint totalIndices = 0;
        for (uint cellIdx = firstCell; cellIdx < firstCell + cellCount; cellIdx++)
        {
            uint packed_data = taskPayloadIn.packedCellData[cellIdx];
            uint prims = (packed_data >> 8) & 0xFFu;
            totalVertices += prims * 3; // 3 vertices per triangle
            totalIndices += prims * 3;  // 3 indices per triangle
        }
        
        meshlets.descriptors[meshletIndex] = MeshletDescriptor(
            pc.globalVertexOffset,  // vertexOffset 
            pc.globalIndexOffset,   // indexOffset
            totalVertices,          // vertexCount
            totalIndices / 3        // primitiveCount
        );
        
        // This is an extraction-only shader, so we output zero geometry.
        SetMeshOutputsEXT(0u, 0u);
    }
}