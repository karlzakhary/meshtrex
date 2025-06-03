#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_control_flow_attributes2 : require
#extension GL_EXT_debug_printf : require


// Configuration - matching your original
#define CORE_BLOCK_SIZE_X 8
#define CORE_BLOCK_SIZE_Y 8  
#define CORE_BLOCK_SIZE_Z 8

#define BLOCK_STRIDE_X 7
#define BLOCK_STRIDE_Y 7
#define BLOCK_STRIDE_Z 7

#define EXTENDED_BLOCK_SIZE_X (CORE_BLOCK_SIZE_X + 2)
#define EXTENDED_BLOCK_SIZE_Y (CORE_BLOCK_SIZE_Y + 2)
#define EXTENDED_BLOCK_SIZE_Z (CORE_BLOCK_SIZE_Z + 2)

#define TOTAL_CORE_CELLS (CORE_BLOCK_SIZE_X * CORE_BLOCK_SIZE_Y * CORE_BLOCK_SIZE_Z)
#define TOTAL_EXTENDED_CELLS (EXTENDED_BLOCK_SIZE_X * EXTENDED_BLOCK_SIZE_Y * EXTENDED_BLOCK_SIZE_Z)

#define MAX_MESHLET_VERTICES 128
#define MAX_MESHLET_PRIMITIVES 254
#define TASK_WORKGROUP_SIZE 32
#define MAX_SUBBLOCKS 64

// PMB Edge ownership
#define PMB_EDGE_X 0
#define PMB_EDGE_Y 3
#define PMB_EDGE_Z 8

// Bindings - keeping your exact names and numbers
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;      
    uvec4 blockStride;   
    uvec4 blockGridDim;  
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount { uint count; } activeBlockCount;
layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs { uint blockIds[]; } compactedBlocks;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable { int triTable[256 * 16]; } mcTriTable;
layout(set = 0, binding = 14, std430) readonly buffer MarchingCubesEdgeTable { uint edgeTable[256]; } mcEdgeTable;
layout(set = 0, binding = 11, std430) buffer VertexCount { uint vertexCount; } vertexCount;
layout(set = 0, binding = 13, std430) buffer IndexCount { uint indexCount; } indexCount;

// Data structures
struct CellData {
    uint cubeCase;
    uint vertexMask;
    uint primitiveCount;
};

struct VertexOwnership {
    uint globalCellIndex;
    uint edgeType;
    uint localVertexIndex;
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
    
    // Vertex ownership for seamless generation
    VertexOwnership vertexOwnership[TOTAL_EXTENDED_CELLS * 3]; // Max 3 vertices per cell
    uint ownedVertexCount;
};

taskPayloadSharedEXT TaskPayload taskPayload;

// Shared memory
shared CellData shared_cellData[EXTENDED_BLOCK_SIZE_X][EXTENDED_BLOCK_SIZE_Y][EXTENDED_BLOCK_SIZE_Z];
shared uint shared_occupiedMorton[TOTAL_CORE_CELLS];
shared uint shared_occupiedCount;
shared VertexOwnership shared_vertexOwnership[TOTAL_EXTENDED_CELLS * 3];
shared uint shared_ownedVertexCount;

// Morton encoding
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

// Sample marching cubes case
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

// Compute vertex mask using edge table
uint computeVertexMask(uint cubeCase) {
    if (cubeCase == 0 || cubeCase == 255) return 0;
    
    uint edgeMask = mcEdgeTable.edgeTable[cubeCase];
    uint vertexMask = 0;
    
    // Check PMB edges
    if ((edgeMask & (1 << PMB_EDGE_X)) != 0) vertexMask |= 1;
    if ((edgeMask & (1 << PMB_EDGE_Y)) != 0) vertexMask |= 2;
    if ((edgeMask & (1 << PMB_EDGE_Z)) != 0) vertexMask |= 4;
    
    return vertexMask;
}

// Count primitives using tri table
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

// Determine vertex ownership with striding
bool blockOwnsVertex(ivec3 globalCellPos, uint edgeType, uvec3 blockOrigin) {
    // Which block primarily contains this cell based on stride?
    ivec3 primaryBlockCoord = globalCellPos / ivec3(BLOCK_STRIDE_X, BLOCK_STRIDE_Y, BLOCK_STRIDE_Z);
    ivec3 ourBlockCoord = ivec3(blockOrigin) / ivec3(BLOCK_STRIDE_X, BLOCK_STRIDE_Y, BLOCK_STRIDE_Z);
    
    // If we're the primary block for this cell, we own its vertices
    if (all(equal(primaryBlockCoord, ourBlockCoord))) {
        return true;
    }
    
    // For cells in overlap regions, use deterministic ownership
    // The block with the lowest coordinates that contains the cell owns it
    ivec3 cellInOurBlock = globalCellPos - ivec3(blockOrigin) + ivec3(1); // Adjust for extended region
    
    // Check if this cell is in our extended region
    if (all(greaterThanEqual(cellInOurBlock, ivec3(0))) && 
        all(lessThan(cellInOurBlock, ivec3(EXTENDED_BLOCK_SIZE_X, 
                                           EXTENDED_BLOCK_SIZE_Y, 
                                           EXTENDED_BLOCK_SIZE_Z)))) {
        
        // Find all blocks that could contain this cell
        ivec3 minPossibleBlock = max((globalCellPos - ivec3(CORE_BLOCK_SIZE_X-1, 
                                                            CORE_BLOCK_SIZE_Y-1, 
                                                            CORE_BLOCK_SIZE_Z-1)) 
                                    / ivec3(BLOCK_STRIDE_X, BLOCK_STRIDE_Y, BLOCK_STRIDE_Z), 
                                    ivec3(0));
        
        // We own it if we're the minimum block
        return all(equal(ourBlockCoord, minPossibleBlock));
    }
    
    return false;
}

// Get global cell index
uint getGlobalCellIndex(ivec3 cellPos) {
    return cellPos.x + cellPos.y * ubo.volumeDim.x + cellPos.z * ubo.volumeDim.x * ubo.volumeDim.y;
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
        shared_ownedVertexCount = 0;
    }
    barrier();
    
    uint compactedBlockId = compactedBlocks.blockIds[blockIndex];
    
    // Calculate block origin using STRIDE for overlap
    uvec3 blockCoord;
    blockCoord.z = compactedBlockId / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = compactedBlockId % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    blockCoord.y = sliceIndex / ubo.blockGridDim.x;
    blockCoord.x = sliceIndex % ubo.blockGridDim.x;
    
    uvec3 blockOrigin = blockCoord * uvec3(BLOCK_STRIDE_X, BLOCK_STRIDE_Y, BLOCK_STRIDE_Z);
    
    // Phase 1: Process all cells in extended region
    uint cellsPerThread = (TOTAL_EXTENDED_CELLS + TASK_WORKGROUP_SIZE - 1) / TASK_WORKGROUP_SIZE;
    for (uint i = 0; i < cellsPerThread; i++) {
        uint idx = threadId + i * TASK_WORKGROUP_SIZE;
        if (idx >= TOTAL_EXTENDED_CELLS) continue;
        
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
            
            // Check if this is a core cell we can generate triangles for
            bool isCore = all(greaterThanEqual(localCoord, uvec3(1))) && 
                         all(lessThan(localCoord, uvec3(EXTENDED_BLOCK_SIZE_X - 1, 
                                                       EXTENDED_BLOCK_SIZE_Y - 1, 
                                                       EXTENDED_BLOCK_SIZE_Z - 1)));
            
            if (cellData.primitiveCount > 0 && isCore) {
                // Check if we have access to all neighbor vertices
                bool canGenerate = true;
                for (int dz = -1; dz <= 1 && canGenerate; dz++) {
                    for (int dy = -1; dy <= 1 && canGenerate; dy++) {
                        for (int dx = -1; dx <= 1 && canGenerate; dx++) {
                            ivec3 neighbor = ivec3(localCoord) + ivec3(dx, dy, dz);
                            if (any(lessThan(neighbor, ivec3(0))) || 
                                any(greaterThanEqual(neighbor, ivec3(EXTENDED_BLOCK_SIZE_X,
                                                                    EXTENDED_BLOCK_SIZE_Y,
                                                                    EXTENDED_BLOCK_SIZE_Z)))) {
                                canGenerate = false;
                            }
                        }
                    }
                }
                
                if (canGenerate) {
                    uint morton = mortonEncode3D(localCoord);
                    uint idx = atomicAdd(shared_occupiedCount, 1);
                    if (idx < TOTAL_CORE_CELLS) {
                        shared_occupiedMorton[idx] = morton;
                    }
                }
            }
            
            // Check vertex ownership
            if (cellData.vertexMask != 0) {
                for (uint edge = 0; edge < 3; edge++) {
                    if ((cellData.vertexMask & (1u << edge)) != 0) {
                        if (blockOwnsVertex(globalCoord, edge, blockOrigin)) {
                            uint vIdx = atomicAdd(shared_ownedVertexCount, 1);
                            if (vIdx < TOTAL_EXTENDED_CELLS * 3) {
                                shared_vertexOwnership[vIdx].globalCellIndex = getGlobalCellIndex(globalCoord);
                                shared_vertexOwnership[vIdx].edgeType = edge;
                                shared_vertexOwnership[vIdx].localVertexIndex = vIdx;
                            }
                        }
                    }
                }
            }
        }
        
        shared_cellData[x][y][z] = cellData;
    }
    
    barrier();
    
    // Phase 2: Prepare payload and partition
    if (threadId == 0) {
        if (shared_occupiedCount == 0) {
            EmitMeshTasksEXT(0, 0, 0);
            return;
        }
        
        taskPayload.blockOrigin = blockOrigin;
        taskPayload.blockId = compactedBlockId;
        taskPayload.occupiedCount = shared_occupiedCount;
        taskPayload.ownedVertexCount = shared_ownedVertexCount;
        
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
        
        for (uint i = 0; i < shared_ownedVertexCount; i++) {
            taskPayload.vertexOwnership[i] = shared_vertexOwnership[i];
        }
        
        // Simple partitioning - adapt from your original
        uint cellsPerSubblock = 32;
        uint numSubblocks = (shared_occupiedCount + cellsPerSubblock - 1) / cellsPerSubblock;
        numSubblocks = min(numSubblocks, uint(MAX_SUBBLOCKS));
        taskPayload.numSubblocks = numSubblocks;
        
        // Allocate global space for owned vertices
        uint totalVertices = shared_ownedVertexCount;
        uint totalPrimitives = shared_occupiedCount * 5; // Conservative
        
        uint globalVertexBase = atomicAdd(vertexCount.vertexCount, totalVertices);
        uint globalIndexBase = atomicAdd(indexCount.indexCount, totalPrimitives * 3);
        
        // Setup subblocks
        uint verticesPerSubblock = (totalVertices + numSubblocks - 1) / numSubblocks;
        uint primitivesPerSubblock = (totalPrimitives + numSubblocks - 1) / numSubblocks;
        
        for (uint i = 0; i < numSubblocks; i++) {
            taskPayload.subblocks[i].mortonStart = i * cellsPerSubblock;
            taskPayload.subblocks[i].mortonCount = min(cellsPerSubblock, shared_occupiedCount - i * cellsPerSubblock);
            taskPayload.subblocks[i].minBounds = uvec3(0);
            taskPayload.subblocks[i].maxBounds = uvec3(EXTENDED_BLOCK_SIZE_X, EXTENDED_BLOCK_SIZE_Y, EXTENDED_BLOCK_SIZE_Z);
            taskPayload.subblocks[i].estimatedVertices = verticesPerSubblock;
            taskPayload.subblocks[i].estimatedPrimitives = primitivesPerSubblock;
            taskPayload.subblocks[i].globalVertexOffset = globalVertexBase + i * verticesPerSubblock;
            taskPayload.subblocks[i].globalIndexOffset = globalIndexBase + i * primitivesPerSubblock * 3;
        }
        
        EmitMeshTasksEXT(numSubblocks, 1, 1);
    }
}