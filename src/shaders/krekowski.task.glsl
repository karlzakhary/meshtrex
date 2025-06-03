#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters for Task Shader ---
#define MACRO_BLOCK_DIM_X 8
#define MACRO_BLOCK_DIM_Y 8
#define MACRO_BLOCK_DIM_Z 8
#define MACRO_BLOCK_CELL_COUNT (MACRO_BLOCK_DIM_X * MACRO_BLOCK_DIM_Y * MACRO_BLOCK_DIM_Z) // 512 for 8x8x8

#define MAX_CELLS_PER_BATCH 32    // Max cells a single mesh shader workgroup will process
#define MAX_ESTIMATED_VERTS_PER_BATCH 168 // Target for mesh shader
#define MAX_ESTIMATED_PRIMS_PER_BATCH 192 // Similar for primitives

#define MAX_BATCHES_PER_MACRO_BLOCK (MACRO_BLOCK_CELL_COUNT / 2 + 1) // Heuristic upper bound

#define MAX_VERTS_PER_CELL_ESTIMATE 12 // For estimation
#define MAX_PRIMS_PER_CELL_ESTIMATE 5  // For estimation
#define MAX_TRI_INDICES 16             // For mc.triTable access

#define TASK_SHADER_WORKGROUP_SIZE 32 // Must match layout(local_size_x)

// --- Structures ---
// Payload from Task Shader to Mesh Shader (must match mesh shader exactly)
struct TaskPayload {
    // These will be set to 0 since we use independent allocation
    uint macroBlockGlobalVertexBase;
    uint macroBlockGlobalIndexBase;
    uint macroBlockGlobalMeshletDescBase;

    uvec3 macroBlockOrigin_global; // Global voxel coords of cell (0,0,0) of this macro-block
    uvec3 macroBlockDimensions;    // Dimensions of the macro-block (e.g., 8,8,8)

    // Information about all occupied cells in this macro-block
    uint numTotalOccupiedCellsInMacroBlock;
    uint denseOccupiedCellIndices[MACRO_BLOCK_CELL_COUNT]; // 1D local indices of occupied cells

    // Information about how occupied cells are batched for mesh shaders
    uint numBatchesToLaunch;
    uvec2 batchCellListInfo[MAX_BATCHES_PER_MACRO_BLOCK]; // .x = offset, .y = count

    // These will be set to 0 since we use independent allocation
    uint batchVertexSubOffsets[MAX_BATCHES_PER_MACRO_BLOCK];
    uint batchIndexSubOffsets[MAX_BATCHES_PER_MACRO_BLOCK];
    
    // Keep estimates for reference (though mesh shader doesn't strictly need them)
    uint estimatedVerticesPerBatch[MAX_BATCHES_PER_MACRO_BLOCK];
    uint estimatedPrimitivesPerBatch[MAX_BATCHES_PER_MACRO_BLOCK];

    uint originalCompactedBlockID; // For debugging
};

taskPayloadSharedEXT TaskPayload taskPayloadOut;

// --- Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;        // Dimensions of the entire volume
    uvec4 blockDim;         // Dimensions of an "original block" (used as our MACRO_BLOCK_DIM)
    uvec4 blockGridDim;     // Grid dimensions in terms of original blocks
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;

// Input: List of active "original block" indices
layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount_SSBO { uint count; } activeBlockCountBuffer;
layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs_SSBO { uint compactedBlkArray[]; } blockIds;

// Marching Cubes Tables
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;

// NOTE: We don't use global counters in task shader anymore - mesh shaders handle their own allocation

// --- Shared Memory for Task Shader ---
shared uint shared_occupiedCellIndices[MACRO_BLOCK_CELL_COUNT];
shared uint shared_occupiedCellEstimatedVerts[MACRO_BLOCK_CELL_COUNT];
shared uint shared_occupiedCellEstimatedPrims[MACRO_BLOCK_CELL_COUNT];
shared uint shared_numOccupiedCellsInMacroBlock;

// --- Helper Functions ---
uint sampleMcCase(uvec3 cell_global_origin) {
    uint cubeCase = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 corner_offset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
        ivec3 voxel_coord = ivec3(cell_global_origin) + corner_offset;
        uint val = 0;
        if (all(greaterThanEqual(voxel_coord, ivec3(0))) && all(lessThan(voxel_coord, ivec3(ubo.volumeDim.xyz)))) {
            val = imageLoad(volumeImage, voxel_coord).r;
        }
        if (float(val) >= ubo.isovalue) {
            cubeCase |= (1 << i);
        }
    }
    return cubeCase;
}

uint countCellVertices_estimate(uint cubeCase) {
    uint flags = 0u; 
    uint vertCount = 0u;
    int base = int(cubeCase) * MAX_TRI_INDICES;
    for (int i = 0; i < MAX_TRI_INDICES; ++i) {
        int edgeID = mc.triTable[base + i];
        if (edgeID < 0) break;
        uint mask = 1u << uint(edgeID);
        if ((flags & mask) == 0u) { 
            flags |= mask; 
            vertCount++; 
        }
    }
    return vertCount;
}

uint countCellPrimitives_estimate(uint cubeCase) {
    uint primCount = 0; 
    int base = int(cubeCase) * MAX_TRI_INDICES;
    for (int i = 0; i < MAX_TRI_INDICES; i += 3) {
        if (mc.triTable[base + i] < 0) break;
        primCount++;
    }
    return primCount;
}

// --- Task Shader Main ---
layout(local_size_x = TASK_SHADER_WORKGROUP_SIZE) in;
void main() {
    // Each task shader workgroup processes one "original block" which is our macro-block.
    uint currentMacroBlockGlobalID = gl_WorkGroupID.x;

    if (currentMacroBlockGlobalID >= activeBlockCountBuffer.count) {
        if (gl_LocalInvocationIndex == 0) EmitMeshTasksEXT(0, 0, 0);
        return;
    }

    uint originalCompactedBlockIndex = blockIds.compactedBlkArray[currentMacroBlockGlobalID];

    // Calculate the global 3D origin of this macro-block
    uvec3 macroBlockOrigin_voxels;
    macroBlockOrigin_voxels.z = originalCompactedBlockIndex / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = originalCompactedBlockIndex % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    macroBlockOrigin_voxels.y = sliceIndex / ubo.blockGridDim.x;
    macroBlockOrigin_voxels.x = sliceIndex % ubo.blockGridDim.x;
    macroBlockOrigin_voxels *= ubo.blockDim.xyz; // ubo.blockDim defines our macro-block size

    uvec3 currentMacroBlockDims = ubo.blockDim.xyz; // Dimensions of this macro-block in cells
    uint totalCellsInCurrentMacroBlock = currentMacroBlockDims.x * currentMacroBlockDims.y * currentMacroBlockDims.z;

    // Initialize shared memory
    if (gl_LocalInvocationIndex == 0) {
        shared_numOccupiedCellsInMacroBlock = 0;
    }
    barrier();

    // PHASE 1: Identify occupied cells and estimate their geometry
    uint cellsProcessedByThisThread[MACRO_BLOCK_CELL_COUNT / TASK_SHADER_WORKGROUP_SIZE + 1];
    uint numCellsForThisThread = 0;
    uint estimatedVertsForThisThread[MACRO_BLOCK_CELL_COUNT / TASK_SHADER_WORKGROUP_SIZE + 1];
    uint estimatedPrimsForThisThread[MACRO_BLOCK_CELL_COUNT / TASK_SHADER_WORKGROUP_SIZE + 1];

    for (uint i = gl_LocalInvocationIndex; i < totalCellsInCurrentMacroBlock; i += TASK_SHADER_WORKGROUP_SIZE) {
        uint localCell1DIndex = i; // 1D index within this macro-block

        // Convert 1D to 3D coordinates within this macro-block
        uvec3 localCellCoords;
        localCellCoords.x = localCell1DIndex % currentMacroBlockDims.x;
        localCellCoords.y = (localCell1DIndex / currentMacroBlockDims.x) % currentMacroBlockDims.y;
        localCellCoords.z = localCell1DIndex / (currentMacroBlockDims.x * currentMacroBlockDims.y);

        uvec3 globalCellOrigin = macroBlockOrigin_voxels + localCellCoords;
        uint cubeCase = sampleMcCase(globalCellOrigin);

        if (cubeCase != 0 && cubeCase != 255) { // If cell is active
            cellsProcessedByThisThread[numCellsForThisThread] = localCell1DIndex;
            estimatedVertsForThisThread[numCellsForThisThread] = countCellVertices_estimate(cubeCase);
            estimatedPrimsForThisThread[numCellsForThisThread] = countCellPrimitives_estimate(cubeCase);
            numCellsForThisThread++;
        }
    }

    // Write this thread's findings to shared memory
    uint writeOffsetInShared = atomicAdd(shared_numOccupiedCellsInMacroBlock, numCellsForThisThread);
    for (uint i = 0; i < numCellsForThisThread; ++i) {
        if (writeOffsetInShared + i < MACRO_BLOCK_CELL_COUNT) {
            shared_occupiedCellIndices[writeOffsetInShared + i] = cellsProcessedByThisThread[i];
            shared_occupiedCellEstimatedVerts[writeOffsetInShared + i] = estimatedVertsForThisThread[i];
            shared_occupiedCellEstimatedPrims[writeOffsetInShared + i] = estimatedPrimsForThisThread[i];
        }
    }
    barrier(); // Ensure all threads have written their occupied cells

    // PHASE 2: Batch Creation (done by thread 0 only)
    if (gl_LocalInvocationIndex == 0) {
        if (shared_numOccupiedCellsInMacroBlock == 0) {
            EmitMeshTasksEXT(0, 0, 0); // No occupied cells, no mesh tasks
            return;
        }

        // Initialize payload
        taskPayloadOut.numBatchesToLaunch = 0;
        taskPayloadOut.numTotalOccupiedCellsInMacroBlock = shared_numOccupiedCellsInMacroBlock;
        taskPayloadOut.macroBlockOrigin_global = macroBlockOrigin_voxels;
        taskPayloadOut.macroBlockDimensions = currentMacroBlockDims;
        taskPayloadOut.originalCompactedBlockID = originalCompactedBlockIndex;

        // IMPORTANT: Clear global allocation fields (mesh shaders will allocate independently)
        taskPayloadOut.macroBlockGlobalVertexBase = 0;
        taskPayloadOut.macroBlockGlobalIndexBase = 0;
        taskPayloadOut.macroBlockGlobalMeshletDescBase = 0;

        // Copy occupied cell data to payload
        for(uint i = 0; i < shared_numOccupiedCellsInMacroBlock; ++i) {
            taskPayloadOut.denseOccupiedCellIndices[i] = shared_occupiedCellIndices[i];
        }

        // Create batches of cells for mesh shaders
        uint currentBatchCellCount = 0;
        uint currentBatchEstimatedVerts = 0;
        uint currentBatchEstimatedPrims = 0;
        uint currentBatchStartIndexInDenseList = 0;

        for (uint i = 0; i < shared_numOccupiedCellsInMacroBlock; ++i) {
            uint cellEstVerts = shared_occupiedCellEstimatedVerts[i];
            uint cellEstPrims = shared_occupiedCellEstimatedPrims[i];

            // Check if adding this cell would exceed batch limits
            if (currentBatchCellCount > 0 && // Don't start new batch if current is empty
                (currentBatchCellCount >= MAX_CELLS_PER_BATCH ||
                 currentBatchEstimatedVerts + cellEstVerts > MAX_ESTIMATED_VERTS_PER_BATCH ||
                 currentBatchEstimatedPrims + cellEstPrims > MAX_ESTIMATED_PRIMS_PER_BATCH)) {
                
                // Finalize current batch
                if (taskPayloadOut.numBatchesToLaunch < MAX_BATCHES_PER_MACRO_BLOCK) {
                    uint batchIdx = taskPayloadOut.numBatchesToLaunch;
                    
                    taskPayloadOut.batchCellListInfo[batchIdx].x = currentBatchStartIndexInDenseList;
                    taskPayloadOut.batchCellListInfo[batchIdx].y = currentBatchCellCount;
                    
                    // Store estimates for reference
                    taskPayloadOut.estimatedVerticesPerBatch[batchIdx] = currentBatchEstimatedVerts;
                    taskPayloadOut.estimatedPrimitivesPerBatch[batchIdx] = currentBatchEstimatedPrims;
                    
                    // IMPORTANT: Clear offset fields (mesh shaders allocate independently)
                    taskPayloadOut.batchVertexSubOffsets[batchIdx] = 0;
                    taskPayloadOut.batchIndexSubOffsets[batchIdx] = 0;

                    taskPayloadOut.numBatchesToLaunch++;
                } else {
                    debugPrintfEXT("TS WARN: Exceeded MAX_BATCHES_PER_MACRO_BLOCK for block %u\n", originalCompactedBlockIndex);
                    break;
                }
                
                // Start new batch
                currentBatchCellCount = 0;
                currentBatchEstimatedVerts = 0;
                currentBatchEstimatedPrims = 0;
                currentBatchStartIndexInDenseList = i; // Current cell starts new batch
            }

            // Add current cell to the current batch
            currentBatchCellCount++;
            currentBatchEstimatedVerts += cellEstVerts;
            currentBatchEstimatedPrims += cellEstPrims;
        }

        // Finalize the last batch if it has cells
        if (currentBatchCellCount > 0 && taskPayloadOut.numBatchesToLaunch < MAX_BATCHES_PER_MACRO_BLOCK) {
            uint batchIdx = taskPayloadOut.numBatchesToLaunch;
            
            taskPayloadOut.batchCellListInfo[batchIdx].x = currentBatchStartIndexInDenseList;
            taskPayloadOut.batchCellListInfo[batchIdx].y = currentBatchCellCount;
            
            // Store estimates for reference
            taskPayloadOut.estimatedVerticesPerBatch[batchIdx] = currentBatchEstimatedVerts;
            taskPayloadOut.estimatedPrimitivesPerBatch[batchIdx] = currentBatchEstimatedPrims;
            
            // IMPORTANT: Clear offset fields (mesh shaders allocate independently)
            taskPayloadOut.batchVertexSubOffsets[batchIdx] = 0;
            taskPayloadOut.batchIndexSubOffsets[batchIdx] = 0;

            taskPayloadOut.numBatchesToLaunch++;
        }
        
        if (taskPayloadOut.numBatchesToLaunch > 0) {
            // DEBUG: Print batch creation info
            if (originalCompactedBlockIndex % 1000 == 0) {
                debugPrintfEXT("TS DBG: Block %u created %u batches from %u occupied cells\n",
                    originalCompactedBlockIndex, taskPayloadOut.numBatchesToLaunch, shared_numOccupiedCellsInMacroBlock);
            }
            
            EmitMeshTasksEXT(taskPayloadOut.numBatchesToLaunch, 1, 1);
        } else {
            EmitMeshTasksEXT(0, 0, 0); // No batches to launch
        }
    } // End if (gl_LocalInvocationIndex == 0)
}