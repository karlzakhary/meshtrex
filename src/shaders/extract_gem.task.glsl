#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
 #extension GL_KHR_shader_subgroup_ballot : require // Not strictly needed for this version
 #extension GL_KHR_shader_subgroup_shuffle : require // Not strictly needed for this version
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Workgroup Layout ---
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in; // Standard size 32

// --- Constants ---
#define MAX_MESHLET_VERTICES 128
#define MAX_MESHLET_PRIMITIVES 256
// Define block dimensions (assuming 8x8x8 input block per task shader)
#define BLOCK_DIM_X 8
#define MIN_BLOCK_DIM 2

#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 8
#define CELLS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z) // 512
#define SUBGROUP_SIZE 32 // Assuming warp size

// --- Meshlet Descriptor Struct (for size calculation) ---
struct MeshletDescriptor { uint vO; uint iO; uint vC; uint pC; }; // Simplified for sizeof

struct BlockEstimate { uint estVertexCount; uint estPrimCount; };

// --- Descriptor Set Bindings (Ensure C++ matches) ---
layout(set = 0, binding = 0, scalar) uniform ExtractionConstantsUBO {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeTexture; // uint8 data
layout(set = 0, binding = 2, scalar) buffer CompactedBlockIDs {
    uint blockIDs[];
}; // Input active block IDs

layout(set = 0, binding = 3, scalar) buffer MarchingCubesTables {
    int data[];
}; // Combined uint tables

layout(set = 0, binding = 4, scalar) buffer VertexBuffer {
    uint vertexCounter;
    vec3 positions[];
};

layout(set = 0, binding = 5, scalar) buffer IndexBuffer {
    uint indexCounter;
    uint indices[];
};

layout(set = 0, binding = 6, scalar) buffer MeshletDescriptorBuffer {
    uint meshletCounter;
    MeshletDescriptor descriptors[];
};

// --- Task Payload ---
struct SubBlockInfo {
    uvec3 blockOrigin;       // Global origin of the PARENT 8x8x8 block
    uvec3 subBlockOffset;    // Offset WITHIN the parent block (0,0,0 for 8x8x8; variable for smaller)
    uvec3 subBlockDim;       // Dimensions of THIS sub-block (8x8x8, 4x4x4, or 2x2x2)
    uint baseVertexOffset;   // Start offset in global VertexBuffer for THIS task
    uint baseIndexOffset;    // Start offset in global IndexBuffer for THIS task
    uint baseDescriptorOffset;// Index in global MeshletDescriptorBuffer for THIS task
    uint activeCellCount;
    uint estVertices;
    uint estIndices; // Store estimate used for allocation
};

uint localID = gl_LocalInvocationIndex;
uint workgroupID = gl_WorkGroupID.x;
// Max potential mesh tasks if 8x8x8 fully subdivides to 2x2x2 -> 4*4*4 = 64
#define MAX_MESH_TASKS_PER_TASK_SHADER 64
taskPayloadSharedEXT SubBlockInfo taskPayload[MAX_MESH_TASKS_PER_TASK_SHADER];
// --- Shared Memory ---
shared uint shared_cell_estimates[CELLS_PER_BLOCK]; // Packed estimate per cell
shared uint shared_cell_claimed_mask[CELLS_PER_BLOCK / 32]; // Bitmask (1 bit per cell) indicating if claimed by a meshlet task
shared SubBlockInfo shared_final_sub_blocks[MAX_MESH_TASKS_PER_TASK_SHADER]; // Final list populated possibly by multiple threads
shared uint shared_num_final_sub_blocks; // Atomic counter for final list index

// --- Helper Functions ---
// Morton encode/decode helpers (3D -> 1D and back for 8x8x8)
// Source: Various examples online (e.g., NVIDIA Dev Blog, Real-Time Rendering)
uint part1By2(uint n) { n &= 0x000003ff; n = (n ^ (n << 16)) & 0xff0000ff; n = (n ^ (n << 8)) & 0x0300f00f; n = (n ^ (n << 4)) & 0x030c30c3; n = (n ^ (n << 2)) & 0x09249249; return n; }
uint mortonEncode3(uvec3 pc) { return (part1By2(pc.z) << 2) | (part1By2(pc.y) << 1) | part1By2(pc.x); }
uint compact1By2(uint n) { n &= 0x09249249; n = (n ^ (n >> 2)) & 0x030c30c3; n = (n ^ (n >> 4)) & 0x0300f00f; n = (n ^ (n >> 8)) & 0xff0000ff; n = (n ^ (n >> 16)) & 0x000003ff; return n; }
uvec3 mortonDecode3(uint mc) { return uvec3(compact1By2(mc), compact1By2(mc >> 1), compact1By2(mc >> 2)); }

// Table Access Helpers (using int tables)
const uint triTableOffsetElements = 0;
const uint numVertsTableOffsetElements = 256 * 16;
int getTriTableEntry(uint caseIndex, uint vertexNum) { return data[triTableOffsetElements + caseIndex * 16 + vertexNum]; }
uint getNumVertsEntry(uint caseIndex) { return uint(data[numVertsTableOffsetElements + caseIndex]); }

// Estimate geometry for a SINGLE cell (uses -1 terminator)
uint estimateGeometryForCell(uint mc_case) {
    if (mc_case == 0 || mc_case == 255) return 0;
    uint vertCount = getNumVertsEntry(mc_case);
    uint primCount = 0; uint idx = 0;
    while (idx < 15 && getTriTableEntry(mc_case, idx) != -1) { primCount++; idx += 3; }
    primCount /= 3;
    return (min(primCount, 0xFFFFu) << 16) | min(vertCount, 0xFFFFu);
}

// Atomically test and set a bit in the shared_cell_claimed_mask
// Returns true if the bit was successfully set (i.e., it was 0 before)
// Requires careful synchronization if multiple threads might claim overlapping regions.
// For hierarchical subdivision, typically a leader thread for a sub-block claims.
bool claimCell(uint cellIdx1D) {
    uint maskIndex = cellIdx1D / 32;
    uint bitIndex = cellIdx1D % 32;
    uint bitMask = (1u << bitIndex);
    // Attempt to set the bit: atomicOr returns the OLD value
    uint oldMask = atomicOr(shared_cell_claimed_mask[maskIndex], bitMask);
    // Return true if the bit was NOT set previously
    return (oldMask & bitMask) == 0;
}

// Check if a cell is already claimed
bool isCellClaimed(uint cellIdx1D) {
    uint maskIndex = cellIdx1D / 32;
    uint bitIndex = cellIdx1D % 32;
    // Need atomic load if other threads might be setting concurrently?
    // Or rely on barriers. Assume barriers are sufficient for read after potential write.
    return (shared_cell_claimed_mask[maskIndex] & (1u << bitIndex)) != 0;
}

// Claim all cells within a sub-block. Returns true if ALL cells were successfully claimed.
// Called by the leader thread for a sub-block found to fit.
bool claimSubBlockCells(uvec3 subBlockOffset, uvec3 subBlockDim) {
    bool allClaimed = true;
    for (uint z = 0; z < subBlockDim.z; ++z) {
        for (uint y = 0; y < subBlockDim.y; ++y) {
            for (uint x = 0; x < subBlockDim.x; ++x) {
                uint cell_lx = subBlockOffset.x + x;
                uint cell_ly = subBlockOffset.y + y;
                uint cell_lz = subBlockOffset.z + z;
                uint cellIdx1D = cell_lz*BLOCK_DIM_X*BLOCK_DIM_Y + cell_ly*BLOCK_DIM_X + cell_lx;
                if (!claimCell(cellIdx1D)) {
                    // This cell was already claimed by another larger/concurrent block!
                    // This indicates an issue with the partitioning logic flow if it happens often.
                    // For now, we signal failure to claim the whole block consistently.
                    allClaimed = false;
                    // We might need logic to un-claim previously claimed cells for this block
                    // if we abort adding it, which is complex.
                }
            }
        }
    }
    return allClaimed;
}

// Estimate geometry for a sub-block by summing estimates of UNCLAIMED cells
// This function now needs to be callable by multiple threads potentially in parallel checks.
BlockEstimate estimateAndCheckClaimed(uvec3 subBlockOffset, uvec3 subBlockDim) {
    BlockEstimate estimate = BlockEstimate(0, 0);
    bool containsUnclaimed = false;

    // Parallel reduction within the workgroup / subgroup for the sub-block sum?
    // Simplification: Still done sequentially by the checking thread for now.
    for (uint z = 0; z < subBlockDim.z; ++z) {
        for (uint y = 0; y < subBlockDim.y; ++y) {
            for (uint x = 0; x < subBlockDim.x; ++x) {
                uint cell_lx = subBlockOffset.x + x;
                uint cell_ly = subBlockOffset.y + y;
                uint cell_lz = subBlockOffset.z + z;
                uint cellIdx1D = cell_lz*BLOCK_DIM_X*BLOCK_DIM_Y + cell_ly*BLOCK_DIM_X + cell_lx;

                if (cellIdx1D < CELLS_PER_BLOCK) {
                    // *** Check if claimed BEFORE summing ***
                    if (!isCellClaimed(cellIdx1D)) {
                        containsUnclaimed = true; // Mark that this sub-block is potentially processable
                        uint packed = shared_cell_estimates[cellIdx1D];
                        estimate.estPrimCount += (packed >> 16);
                        estimate.estVertexCount += (packed & 0xFFFFu);
                    }
                }
            }
        }
    }
    // If block had no unclaimed cells with geometry, treat as empty
    if (!containsUnclaimed) {
        estimate.estPrimCount = 0;
        estimate.estVertexCount = 0;
    }
    return estimate;
}


// --- Main ---
void main() {
    uint taskID = gl_WorkGroupID.x;
    uint localID = gl_LocalInvocationIndex;

    // --- Phase 0: Initialization ---
    if (localID == 0) { shared_num_final_sub_blocks = 0; }
    // Init cell estimates AND claimed mask
    for (uint i = localID; i < CELLS_PER_BLOCK; i += gl_WorkGroupSize.x) {
        shared_cell_estimates[i] = 0;
    }
    for (uint i = localID; i < (CELLS_PER_BLOCK / 32); i += gl_WorkGroupSize.x) {
        shared_cell_claimed_mask[i] = 0u; // Clear claimed bits
    }
    barrier();

    // --- Phase 1: Calculate Per-Cell Estimates (Morton Order) ---
    if (taskID >= blockIDs.length()) return;
    uint blockIndex1D = blockIDs[taskID];
    uvec3 blockCoord; // Calculate blockCoord
    blockCoord.x = blockIndex1D % ubo.blockGridDim.x; blockCoord.y = (blockIndex1D / ubo.blockGridDim.x) % ubo.blockGridDim.y; blockCoord.z = blockIndex1D / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uvec3 blockOrigin = blockCoord * uvec3(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);

    // Each thread processes a chunk of Morton-ordered cells
    // uint cellsPerThread = (CELLS_PER_BLOCK + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    // uint startMortonIdx = localID * cellsPerThread;
    // uint endMortonIdx = min(startMortonIdx + cellsPerThread, CELLS_PER_BLOCK);

    // Simpler: Each thread handles 16 cells, assumes workgroup size 32
    if (gl_WorkGroupSize.x != 32) { /* Need different cell distribution */ }
    uint startMortonIdx = localID * 16;
    uint endMortonIdx = startMortonIdx + 16;


    for (uint mortonIdx = startMortonIdx; mortonIdx < endMortonIdx; ++mortonIdx) {
        uvec3 localCellCoord = mortonDecode3(mortonIdx); // Decode Morton index to local coords
        ivec3 globalCellCoord = ivec3(blockOrigin) + ivec3(localCellCoord);

        uint mc_case = 0; bool bdry = false;
        for (int bit = 0; bit < 8; ++bit) { // Calculate mc_case
                                            ivec3 off = ivec3(bit & 1, (bit >> 1) & 1, (bit >> 2) & 1); ivec3 pos = globalCellCoord + off;
                                            if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, ivec3(ubo.volumeDim.xyz)))) { bdry = true; break; }
                                            uint v = imageLoad(volumeTexture, pos).r; if (float(v) >= ubo.isovalue) mc_case |= (1 << bit);
        }
        if (!bdry) shared_cell_estimates[mortonIdx] = estimateGeometryForCell(mc_case);
    }
    barrier(); // Ensure all estimates are written


    // --- Phase 2: Hierarchical Partitioning & Greedy Claiming (Parallel Attempt) ---
    // Process levels top-down (8x8x8 -> 4x4x4 -> 2x2x2)

    uvec3 levelDims[3] = { uvec3(8u), uvec3(4u), uvec3(2u) };

    for (int level = 0; level < 3; ++level) {
        uvec3 subDim = levelDims[level];
        uint numSubX = BLOCK_DIM_X / subDim.x;
        uint numSubY = BLOCK_DIM_Y / subDim.y;
        uint numSubZ = BLOCK_DIM_Z / subDim.z;
        uint totalSubs = numSubX * numSubY * numSubZ; // Total potential sub-blocks at this level

        // Distribute sub-block checks across the workgroup
        for (uint sb_idx = localID; sb_idx < totalSubs; sb_idx += gl_WorkGroupSize.x) {
            // Calculate offset for this sub-block
            uint sx = sb_idx % numSubX;
            uint sy = (sb_idx / numSubX) % numSubY;
            uint sz = sb_idx / (numSubX * numSubY);
            uvec3 subOffset = uvec3(sx, sy, sz) * subDim;

            // *** CRITICAL: Check if ANY cell in this sub-block is already claimed ***
            // This prevents processing children of an already claimed parent.
            // This requires reading the shared mask - ensure visibility with barriers.
            bool parentClaimed = false;
            // Optimization: only need to check one cell (e.g., origin) if claiming is hierarchical?
            // Safer: check all cells? Too slow. Let's check origin cell for simplicity.
            uint originCellIdx1D = subOffset.z*BLOCK_DIM_X*BLOCK_DIM_Y + subOffset.y*BLOCK_DIM_X + subOffset.x;
            if (isCellClaimed(originCellIdx1D)) {
                parentClaimed = true; // If origin is claimed, assume parent was claimed
                // This simplification might miss cases if claiming isn't perfectly hierarchical.
            }


            if (!parentClaimed) {
                // Estimate geometry for this sub-block using ONLY unclaimed cells
                BlockEstimate estimate = estimateAndCheckClaimed(subOffset, subDim);

                // If it fits the limits and contains unclaimed geometry...
                if ((estimate.estVertexCount > 0 || estimate.estPrimCount > 0) &&
                estimate.estVertexCount <= MAX_MESHLET_VERTICES &&
                estimate.estPrimCount <= MAX_MESHLET_PRIMITIVES)
                {
                    // Try to claim all cells in this sub-block
                    // This needs atomics or careful subgroup logic to avoid races if multiple threads
                    // claim overlapping regions (e.g. a 4x4 fitting vs a 2x2 fitting inside it).
                    // Simplification: Assume leader thread of a conceptual subgroup for this sub-block does the claim.
                    // For now, use simple atomicAdd on final list and hope claimSubBlockCells works (it has races).
                    // A better way needs a lock or more complex subgroup coordination for claiming.

                    // Attempt to add to final list atomically
                    uint final_idx = atomicAdd(shared_num_final_sub_blocks, 1);
                    if (final_idx < MAX_MESH_TASKS_PER_TASK_SHADER) {
                        // Claim the cells for this block *after* securing a spot in the list.
                        // NOTE: Potential race condition here! If claimSubBlockCells fails because
                        // another thread claimed *some* cells concurrently, this entry is invalid.
                        bool claimSuccess = claimSubBlockCells(subOffset, subDim);

                        if (claimSuccess) {
                            shared_final_sub_blocks[final_idx].blockOrigin = blockOrigin;
                            shared_final_sub_blocks[final_idx].subBlockOffset = subOffset;
                            shared_final_sub_blocks[final_idx].subBlockDim = subDim;
                            shared_final_sub_blocks[final_idx].estVertices = estimate.estVertexCount;
                            shared_final_sub_blocks[final_idx].estIndices = estimate.estPrimCount * 3;
                        } else {
                            // Failed to claim all cells (race condition likely occurred). Revert atomic add.
                            atomicAdd(shared_num_final_sub_blocks, -1);
                            // Need to potentially un-claim the partially claimed cells - complex!
                            // For now, just debug print if this happens.
                            if (localID == 0 && workgroupID < 1) {
                                debugPrintfEXT("W%u T%u: Claim failed for block off=(%u,%u,%u) dim=(%u,%u,%u)", workgroupID, localID, subOffset.x, subOffset.y, subOffset.z, subDim.x, subDim.y, subDim.z);
                            }
                        }
                    } else {
                        // Revert atomic add if list is full
                        atomicAdd(shared_num_final_sub_blocks, -1);
                    }
                }
                // else: estimate doesn't fit (will be checked at finer level) or was empty
            } // end if !parentClaimed
        } // end loop over sub-blocks for this thread
        barrier(); // Synchronize after each level check to ensure claims are visible
    } // End loop over levels

    // --- Phase 3 & 4: Allocation & Dispatch (Leader Thread) ---
    // This part remains largely the same as the previous "divisive" version,
    // using the final list built in shared_final_sub_blocks and counted by shared_num_final_sub_blocks.
    barrier(); // Ensure partitioning and counter updates are complete
    uint final_task_count = shared_num_final_sub_blocks;
    if (subgroupElect() && final_task_count > 0) {

        // Calculate accurate total allocation size based on the final list estimates
        uint total_verts_needed = 0;
        uint total_indices_needed = 0;
        for (uint i = 0; i < final_task_count; ++i) {
            total_verts_needed += shared_final_sub_blocks[i].estVertices; // Uses estimate stored when added
            total_indices_needed += shared_final_sub_blocks[i].estIndices;
        }

        // Perform atomic allocations
        uint baseDescOffset = atomicAdd(meshletCounter, final_task_count);
        uint baseVertOffset = atomicAdd(vertexCounter, total_verts_needed);
        uint baseIdxOffset = atomicAdd(indexCounter, total_indices_needed);

        // Assign specific offsets to each task's payload entry
        uint current_vert_offset = baseVertOffset;
        uint current_idx_offset = baseIdxOffset;
        for (uint i = 0; i < final_task_count; ++i) {
            taskPayload[i] = shared_final_sub_blocks[i]; // Copy spatial info
            taskPayload[i].baseDescriptorOffset = baseDescOffset + i;
            taskPayload[i].baseVertexOffset = current_vert_offset;
            taskPayload[i].baseIndexOffset = current_idx_offset;
            // Advance offsets using the stored clamped estimates
            current_vert_offset += shared_final_sub_blocks[i].estVertices;
            current_idx_offset += shared_final_sub_blocks[i].estIndices;
            taskPayload[i].activeCellCount = 0; // Not accurately tracked here
        }

        // Dispatch
        EmitMeshTasksEXT(final_task_count, 1, 1);

    } // End leader thread allocation/dispatch block
}