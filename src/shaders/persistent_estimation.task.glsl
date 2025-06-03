#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_control_flow_attributes2 : require
#extension GL_EXT_debug_printf : require

// --- Configuration ---
#define CORE_BLOCK_SIZE_X 8
#define CORE_BLOCK_SIZE_Y 8  
#define CORE_BLOCK_SIZE_Z 8

// Extended region for context cells
#define EXTENDED_BLOCK_SIZE_X (CORE_BLOCK_SIZE_X + 2)
#define EXTENDED_BLOCK_SIZE_Y (CORE_BLOCK_SIZE_Y + 2)
#define EXTENDED_BLOCK_SIZE_Z (CORE_BLOCK_SIZE_Z + 2)

#define TOTAL_CORE_CELLS (CORE_BLOCK_SIZE_X * CORE_BLOCK_SIZE_Y * CORE_BLOCK_SIZE_Z)
#define TOTAL_EXTENDED_CELLS (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y * EXTENDED_BLOCK_SIZE_Z)

#define MAX_MESHLET_VERTICES 256
#define MAX_MESHLET_PRIMITIVES 256 / 3
#define TASK_WORKGROUP_SIZE 32
#define MAX_SUBBLOCKS 64

// PMB Edge ownership
#define PMB_EDGE_X 0
#define PMB_EDGE_Y 3
#define PMB_EDGE_Z 8

// --- Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;      // Actual block dimensions (8x8x8)
    uvec4 blockGridDim;  // Grid dimensions using stride
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount { uint count; } activeBlockCount;
layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs { uint blockIds[]; } compactedBlocks;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable { int triTable[256 * 16]; } mcTriTable;
layout(set = 0, binding = 14, std430) readonly buffer MarchingCubesEdgeTable { uint edgeTable[256]; } mcEdgeTable;
layout(set = 0, binding = 11, std430) buffer VertexCount { uint vertexCount; } vertexCount;
layout(set = 0, binding = 13, std430) buffer IndexCount { uint indexCount; } indexCount;

// --- Data Structures ---
struct CellData {
    uint cubeCase;
    uint vertexMask;
    uint primitiveCount;
};

struct SubblockInfo {
    uint mortonStart;
    uint mortonCount;
    uvec3 minBounds;
    uvec3 maxBounds;
    uint estimatedVertices;
    uint estimatedPrimitives;
    uint globalVertexOffset;
    uint globalIndexOffset;
};

struct TaskPayload {
    uvec3 blockOrigin;
    uint blockId;
    uint numSubblocks;
    SubblockInfo subblocks[MAX_SUBBLOCKS];
    CellData cellData[EXTENDED_BLOCK_SIZE_X][EXTENDED_BLOCK_SIZE_Y][EXTENDED_BLOCK_SIZE_Z];
    uint occupiedMortonIndices[TOTAL_CORE_CELLS];
    uint occupiedCount;
};

taskPayloadSharedEXT TaskPayload taskPayload;

// --- Shared Memory ---
shared CellData shared_cellData[EXTENDED_BLOCK_SIZE_X][EXTENDED_BLOCK_SIZE_Y][EXTENDED_BLOCK_SIZE_Z];
shared uint shared_occupiedMorton[TOTAL_CORE_CELLS];
shared uint shared_occupiedCount;

// --- Morton Encoding Functions ---
uint expandBits(uint v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

uint mortonEncode3D(uvec3 coord) {
    return (expandBits(coord.z) << 2) | (expandBits(coord.y) << 1) | expandBits(coord.x);
}

uint compactBits(uint v) {
    v &= 0x09249249;
    v = (v | (v >> 2)) & 0x030C30C3;
    v = (v | (v >> 4)) & 0x0300F00F;
    v = (v | (v >> 8)) & 0x030000FF;
    v = (v | (v >> 16)) & 0x000003FF;
    return v;
}
uvec3 mortonDecode3D(uint morton) {    
    return uvec3(
        compactBits(morton >> 0),
        compactBits(morton >> 1),
        compactBits(morton >> 2)
    );
}

// --- Helper Functions ---
uint sampleMarchingCubesCase(uvec3 cellOrigin) {
    uint cubeCase = 0;
    [[unroll]] for (int i = 0; i < 8; ++i) {
        ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
        ivec3 voxelCoord = ivec3(cellOrigin) + cornerOffset;
        
        if (all(greaterThanEqual(voxelCoord, ivec3(0))) && 
            all(lessThan(voxelCoord, ivec3(ubo.volumeDim.xyz)))) {
            uint voxelValue = imageLoad(volumeImage, voxelCoord).r;
            if (float(voxelValue) >= ubo.isovalue) {
                cubeCase |= (1 << i);
            }
        }
    }
    return cubeCase;
}

uint computeVertexMask(uint cubeCase) {
    if (cubeCase == 0 || cubeCase == 255) return 0;
    
    uint edgeMask = mcEdgeTable.edgeTable[cubeCase];
    uint vertexMask = 0;
    
    if ((edgeMask & (1 << PMB_EDGE_X)) != 0) vertexMask |= 1;
    if ((edgeMask & (1 << PMB_EDGE_Y)) != 0) vertexMask |= 2;
    if ((edgeMask & (1 << PMB_EDGE_Z)) != 0) vertexMask |= 4;
    
    return vertexMask;
}

uint countPrimitives(uint cubeCase) {
    if (cubeCase == 0 || cubeCase == 255) return 0;
    
    uint count = 0;
    int baseIndex = int(cubeCase) * 16;
    [[unroll]] for (int i = 0; i < 15; i += 3) {
        if (mcTriTable.triTable[baseIndex + i] == -1) break;
        count++;
    }
    return count;
}

void partitionBlockMorton() {
    uint numSubblocks = 0;
    uint currentStart = 0;
    
    while (currentStart < shared_occupiedCount && numSubblocks < MAX_SUBBLOCKS) {
        uint currentEnd = currentStart + 1;
        uvec3 coreMin = mortonDecode3D(shared_occupiedMorton[currentStart]);
        uvec3 coreMax = coreMin + uvec3(1);
        
        uint coreVertexCount = 0;
        uint corePrimitiveCount = 0;
        
        // Count initial cell
        {
            CellData cell = shared_cellData[coreMin.x][coreMin.y][coreMin.z];
            coreVertexCount = bitCount(cell.vertexMask);
            corePrimitiveCount = cell.primitiveCount;
        }
        
        // Try to add more cells to this subblock
        while (currentEnd < shared_occupiedCount) {
            uvec3 nextCell = mortonDecode3D(shared_occupiedMorton[currentEnd]);
            uvec3 newCoreMin = min(coreMin, nextCell);
            uvec3 newCoreMax = max(coreMax, nextCell + uvec3(1));
            
            // Count vertices for new configuration
            uint newVertexCount = 0;
            uint newPrimitiveCount = 0;
            
            for (uint i = currentStart; i <= currentEnd; i++) {
                uvec3 coord = mortonDecode3D(shared_occupiedMorton[i]);
                CellData cell = shared_cellData[coord.x][coord.y][coord.z];
                newVertexCount += bitCount(cell.vertexMask);
                newPrimitiveCount += cell.primitiveCount;
            }
            
            // CRITICAL FIX: Count ALL vertices in the extended region
            // This includes both core and context cells
            uvec3 extendedMin = max(ivec3(newCoreMin) - ivec3(1), ivec3(0));
            uvec3 extendedMax = min(newCoreMax + uvec3(1), uvec3(EXTENDED_BLOCK_SIZE_X, 
                                                                  EXTENDED_BLOCK_SIZE_Y, 
                                                                  EXTENDED_BLOCK_SIZE_Z));
            
            uint totalVerticesInRegion = 0;
            for (uint z = extendedMin.z; z < extendedMax.z; z++) {
                for (uint y = extendedMin.y; y < extendedMax.y; y++) {
                    for (uint x = extendedMin.x; x < extendedMax.x; x++) {
                        CellData cell = shared_cellData[x][y][z];
                        if (cell.vertexMask != 0) {
                            totalVerticesInRegion += bitCount(cell.vertexMask);
                        }
                    }
                }
            }
            
            // Check if this configuration would exceed meshlet limits
            // Use actual vertex count, not estimated
            if (totalVerticesInRegion > MAX_MESHLET_VERTICES || 
                newPrimitiveCount > MAX_MESHLET_PRIMITIVES) {
                debugPrintfEXT("totalVerticesInRegion: %d, newPrimitiveCount: %d", totalVerticesInRegion, newPrimitiveCount);
                break;
            }
            
            coreMin = newCoreMin;
            coreMax = newCoreMax;
            currentEnd++;
        }
        
        // Create subblock with proper bounds
        SubblockInfo subblock;
        subblock.mortonStart = currentStart;
        subblock.mortonCount = currentEnd - currentStart;
        subblock.minBounds = max(ivec3(coreMin) - ivec3(1), ivec3(0));
        subblock.maxBounds = min(coreMax + uvec3(1), uvec3(EXTENDED_BLOCK_SIZE_X, 
                                                           EXTENDED_BLOCK_SIZE_Y, 
                                                           EXTENDED_BLOCK_SIZE_Z));
        
        // CRITICAL: Count ALL vertices that will be generated
        uint totalVerticesInSubblock = 0;
        for (uint z = subblock.minBounds.z; z < subblock.maxBounds.z; z++) {
            for (uint y = subblock.minBounds.y; y < subblock.maxBounds.y; y++) {
                for (uint x = subblock.minBounds.x; x < subblock.maxBounds.x; x++) {
                    CellData cell = shared_cellData[x][y][z];
                    if (cell.vertexMask != 0) {
                        totalVerticesInSubblock += bitCount(cell.vertexMask);
                    }
                }
            }
        }
        
        // Count actual primitives from occupied cells only
        uint actualPrimitiveCount = 0;
        for (uint i = currentStart; i < currentEnd; i++) {
            uvec3 coord = mortonDecode3D(shared_occupiedMorton[i]);
            CellData cell = shared_cellData[coord.x][coord.y][coord.z];
            actualPrimitiveCount += cell.primitiveCount;
        }
        
        // Set estimates with safety margin
        // The total vertices includes all context cells that might be referenced
        subblock.estimatedVertices = totalVerticesInSubblock + 64;  // Safety margin
        subblock.estimatedPrimitives = actualPrimitiveCount + 32;   // Safety margin
        
        // Ensure we don't exceed hardware limits even with safety margin
        subblock.estimatedVertices = min(subblock.estimatedVertices, uint(MAX_MESHLET_VERTICES));
        subblock.estimatedPrimitives = min(subblock.estimatedPrimitives, uint(MAX_MESHLET_PRIMITIVES));
        
        taskPayload.subblocks[numSubblocks] = subblock;
        numSubblocks++;
        currentStart = currentEnd;
    }
    
    taskPayload.numSubblocks = numSubblocks;
}

bool canGenerateAllTrianglesForCell(uvec3 localCoord) {
    // Only check the 8 cells that contribute edges to this cell's triangles
    const ivec3 requiredOffsets[8] = {
        ivec3(0,0,0), ivec3(1,0,0), ivec3(0,1,0), ivec3(0,0,1),
        ivec3(1,1,0), ivec3(1,0,1), ivec3(0,1,1), ivec3(1,1,1)
    };
    
    for (int i = 0; i < 8; i++) {
        ivec3 checkCoord = ivec3(localCoord) + requiredOffsets[i];
        
        if (any(lessThan(checkCoord, ivec3(0))) || 
            any(greaterThanEqual(checkCoord, ivec3(EXTENDED_BLOCK_SIZE_X,
                                                  EXTENDED_BLOCK_SIZE_Y,
                                                  EXTENDED_BLOCK_SIZE_Z)))) {
            return false;
        }
    }
    
    return true;
}

layout(local_size_x = TASK_WORKGROUP_SIZE) in;
void main() {
    uint blockIndex = gl_WorkGroupID.x;
    uint threadId = gl_LocalInvocationIndex;
    
    if (blockIndex >= activeBlockCount.count) {
        if (threadId == 0) EmitMeshTasksEXT(0, 0, 0);
        return;
    }
    
    // Initialize shared memory
    if (threadId == 0) {
        shared_occupiedCount = 0;
    }
    
    // Clear shared cell data
    uint cellsPerThread = (TOTAL_EXTENDED_CELLS + TASK_WORKGROUP_SIZE - 1) / TASK_WORKGROUP_SIZE;
    for (uint i = 0; i < cellsPerThread; i++) {
        uint idx = threadId + i * TASK_WORKGROUP_SIZE;
        if (idx < TOTAL_EXTENDED_CELLS) {
            uint z = idx / (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y);
            uint y = (idx % (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y)) / EXTENDED_BLOCK_SIZE_X;
            uint x = idx % EXTENDED_BLOCK_SIZE_X;
            shared_cellData[x][y][z] = CellData(0, 0, 0);
        }
    }
    
    barrier();
    
    uint compactedBlockId = compactedBlocks.blockIds[blockIndex];
    
    uvec3 blockCoord;
    blockCoord.z = compactedBlockId / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = compactedBlockId % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    blockCoord.y = sliceIndex / ubo.blockGridDim.x;
    blockCoord.x = sliceIndex % ubo.blockGridDim.x;
    
    uvec3 blockOrigin = blockCoord * ubo.blockDim.xyz;
    
    // Phase 1: Process all cells in extended region
    for (uint i = 0; i < cellsPerThread; i++) {
        uint idx = threadId + i * TASK_WORKGROUP_SIZE;
        if (idx < TOTAL_EXTENDED_CELLS) {
            uint z = idx / (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y);
            uint y = (idx % (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y)) / EXTENDED_BLOCK_SIZE_X;
            uint x = idx % EXTENDED_BLOCK_SIZE_X;
            
            uvec3 localCoord = uvec3(x, y, z);
            ivec3 globalCoord = ivec3(blockOrigin) + ivec3(localCoord) - ivec3(1);
            
            CellData cellData;
            cellData.cubeCase = 0;
            cellData.vertexMask = 0;
            cellData.primitiveCount = 0;
            
            if (all(greaterThanEqual(globalCoord, ivec3(0))) && 
                all(lessThan(globalCoord + ivec3(1), ivec3(ubo.volumeDim.xyz)))) {
                cellData.cubeCase = sampleMarchingCubesCase(uvec3(globalCoord));
                cellData.vertexMask = computeVertexMask(cellData.cubeCase);
                cellData.primitiveCount = countPrimitives(cellData.cubeCase);

                if (cellData.primitiveCount > 0 && canGenerateAllTrianglesForCell(localCoord)) {
                    uint morton = mortonEncode3D(localCoord);
                    uint idx = atomicAdd(shared_occupiedCount, 1);
                    if (idx < TOTAL_CORE_CELLS) {
                        shared_occupiedMorton[idx] = morton;
                    }
                }
            }
            shared_cellData[x][y][z] = cellData;
        }
    }
    
    barrier();
    
    // Phase 2: Partition and emit
    if (threadId == 0) {
        if (shared_occupiedCount == 0) {
            EmitMeshTasksEXT(0, 0, 0);
            return;
        }
        
        taskPayload.blockOrigin = blockOrigin;
        taskPayload.blockId = compactedBlockId;
        taskPayload.occupiedCount = shared_occupiedCount;
        
        // Copy data to payload
        for (uint z = 0; z < EXTENDED_BLOCK_SIZE_Z; z++) {
            for (uint y = 0; y < EXTENDED_BLOCK_SIZE_Y; y++) {
                for (uint x = 0; x < EXTENDED_BLOCK_SIZE_X; x++) {
                    taskPayload.cellData[x][y][z] = shared_cellData[x][y][z];
                }
            }
        }
        
        for (uint i = 0; i < shared_occupiedCount; i++) {
            taskPayload.occupiedMortonIndices[i] = shared_occupiedMorton[i];
        }
        
        // Partition using improved algorithm
        partitionBlockMorton();
        
        // Allocate global space
        uint totalVertices = 0;
        uint totalPrimitives = 0;
        for (uint i = 0; i < taskPayload.numSubblocks; i++) {
            totalVertices += taskPayload.subblocks[i].estimatedVertices;
            totalPrimitives += taskPayload.subblocks[i].estimatedPrimitives;
        }
        
        uint globalVertexBase = atomicAdd(vertexCount.vertexCount, totalVertices);
        uint globalIndexBase = atomicAdd(indexCount.indexCount, totalPrimitives * 3);
        
        // Assign offsets
        uint currentVertexOffset = globalVertexBase;
        uint currentIndexOffset = globalIndexBase;
        
        for (uint i = 0; i < taskPayload.numSubblocks; i++) {
            taskPayload.subblocks[i].globalVertexOffset = currentVertexOffset;
            taskPayload.subblocks[i].globalIndexOffset = currentIndexOffset;
            currentVertexOffset += taskPayload.subblocks[i].estimatedVertices;
            currentIndexOffset += taskPayload.subblocks[i].estimatedPrimitives * 3;
        }
        if (taskPayload.numSubblocks > 1) {
            // debugPrintfEXT("taskPayload.numSubblocks: %d", taskPayload.numSubblocks);
        }
        EmitMeshTasksEXT(taskPayload.numSubblocks, 1, 1);
    }
}