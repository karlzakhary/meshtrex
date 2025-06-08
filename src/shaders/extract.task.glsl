#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_shader_atomic_int64 : enable // If using 64-bit offsets/counters
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Workgroup Layout ---
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// --- Constants ---
#define MAX_MESHLET_VERTICES 128u
#define MAX_MESHLET_PRIMITIVES 256u
#define INPUT_BLOCK_DIM_X 8u
#define INPUT_BLOCK_DIM_Y 8u
#define INPUT_BLOCK_DIM_Z 8u
#define CELLS_PER_INPUT_BLOCK (INPUT_BLOCK_DIM_X * INPUT_BLOCK_DIM_Y * INPUT_BLOCK_DIM_Z) // 512
#define MIN_SUB_BLOCK_DIM 2u // Smallest dimension for subdivision

// For the temporary edge hash set in the estimator function
#define ESTIMATE_MAX_EDGES 2048u // Max unique edges for one sub-block estimate (tune if needed)
#define ESTIMATE_EDGE_HASH_WORDS (ESTIMATE_MAX_EDGES / 32u) // Bitmask array size

#define MAX_MESH_TASKS_PER_TASK_SHADER 64u // Max 2x2x2 sub-blocks in an 8x8x8

// --- Structs ---
struct MeshletDescriptorLayout { uint vO; uint iO; uint vC; uint pC; }; // For C++ layout matching
struct BlockEstimate { uint estVertexCount; uint estPrimCount; };

// Stored in shared_final_sub_blocks. Includes estimate used for allocation.
struct FinalSubBlockInfo {
    uvec3 subBlockOffsetInParent; // Offset relative to the 8x8x8 parent block
    uvec3 subBlockDim;
    BlockEstimate estimate;
};
// Actual payload sent to Mesh Shader
struct SubBlockInfoPayload {
    uvec3 parentBlockOrigin;
    uvec3 subBlockOffset; // Relative to parentBlockOrigin
    uvec3 subBlockDim;
    uint baseVertexOffset;
    uint baseIndexOffset;
    uint baseDescriptorOffset;
};
taskPayloadSharedEXT SubBlockInfoPayload taskPayload[MAX_MESH_TASKS_PER_TASK_SHADER];

// --- Bindings ---
layout(set = 0, binding = 0, scalar) uniform ExtractionConstantsUBO {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeTexture; // uint8 data

layout(binding = 2, std430) buffer ActiveBlockCount {
    uint count; // Stores the total number of active blocks found
} activeBlockCount;

layout(set = 0, binding = 3, scalar) buffer CompactedBlockIDs {
    uint blockIDs[];
}; // Input active block IDs

layout(set = 0, binding = 4, scalar) buffer MarchingCubesTriangleTable {
    int triTable[];
};

layout(set = 0, binding = 5, scalar) buffer MarchingCubesNumberVertices {
    int numVertices[];
};

layout(set = 0, binding = 6, scalar) buffer VertexBuffer {
    uint vertexCounter;
    vec3 positions[];
};

layout(set = 0, binding = 7, scalar) buffer IndexBuffer {
    uint indexCounter;
    uint indices[];
};

layout(set = 0, binding = 8, scalar) buffer MeshletDescriptorBuffer {
    uint meshletCounter;
    MeshletDescriptorLayout descriptors[];
};

// --- Shared Memory ---
shared uint shared_cell_mc_cases[CELLS_PER_INPUT_BLOCK]; // Morton indexed
// Claim Mask: 0 = unclaimed, 1 = finally claimed
shared uint shared_cell_claimed_mask[CELLS_PER_INPUT_BLOCK / 32u];
// Final list of sub-blocks chosen for dispatch
shared FinalSubBlockInfo shared_final_sub_blocks[MAX_MESH_TASKS_PER_TASK_SHADER];
// Atomically incremented counter for the final list
shared uint shared_num_final_sub_blocks;

// --- Helper Functions ---
// Corrected Morton coding for 3-bit coordinates (0-7 range)
// --- Morton encode/decode for 3-bit coords ---
uint Morton_SpreadBits_3bit(uint x) {
    x &= 0x7u;
    x = (x | (x << 8))  & 0x0000F00Fu;
    x = (x | (x << 4))  & 0x000C30C3u;
    x = (x | (x << 2))  & 0x00249249u;
    return x;
}
uint mortonEncode3D_3bit(uvec3 c) {
    return Morton_SpreadBits_3bit(c.x)
    | (Morton_SpreadBits_3bit(c.y) << 1)
    | (Morton_SpreadBits_3bit(c.z) << 2);
}
uint Morton_CompactBits_3bit(uint x) {
    x &= 0x00249249u;
    x = (x | (x >> 2)) & 0x000C30C3u;
    x = (x | (x >> 4)) & 0x0000F00Fu;
    x = (x | (x >> 8)) & 0x0000007Fu;
    return x;
}
uvec3 mortonDecode3D_3bit(uint m) {
    return uvec3(
    Morton_CompactBits_3bit(m),
    Morton_CompactBits_3bit(m >> 1),
    Morton_CompactBits_3bit(m >> 2)
    );
}

const uint triTableOffsetElements = 0;
int getTriTableEntry(uint mc_case, uint vertNum) {
    return triTable[triTableOffsetElements + mc_case * 16 + vertNum];
}

// Reads claimed status non-atomically. Requires prior memoryBarrierShared().
bool isCellClaimed_ReadOnly_AfterBarrier(uint mortonCellIdx) {
    uint maskWordIndex = mortonCellIdx / 32u;
    uint bitInWord = mortonCellIdx % 32u;
    if (maskWordIndex >= (CELLS_PER_INPUT_BLOCK / 32u)) return true;
    return (shared_cell_claimed_mask[maskWordIndex] & (1u << bitInWord)) != 0;
}

void rollbackClaimedCells(uvec3 subBlockOffsetInParent, uvec3 subBlockDim) {
    for (uint z = 0; z < subBlockDim.z; ++z) {
        for (uint y = 0; y < subBlockDim.y; ++y) {
            for (uint x = 0; x < subBlockDim.x; ++x) {
                uint m = mortonEncode3D_3bit(subBlockOffsetInParent + uvec3(x, y, z));
                uint w = m / 32u, b = m % 32u;
                atomicAnd(shared_cell_claimed_mask[w], ~(1u << b));
            }
        }
    }
}

// Attempts to claim all cells in a sub-block using atomicCompSwap.
// Returns true if ALL cells were successfully changed from 0 to 1 by THIS CALL.
// If any cell was already 1, or if a CAS fails (another thread claimed it),
// it attempts to roll back bits IT set during this attempt and returns false.
bool tryClaimCells_SubBlock_AllOrNothing_Atomic(uvec3 subBlockOffsetInParent, uvec3 subBlockDim) {
    uint cells_to_claim_morton[CELLS_PER_INPUT_BLOCK];
    uint num_cells_this_block = 0;
    uint bits_successfully_set_by_this_attempt_mask[CELLS_PER_INPUT_BLOCK / 32u];
    for(uint i=0; i < CELLS_PER_INPUT_BLOCK / 32u; ++i) bits_successfully_set_by_this_attempt_mask[i] = 0u;

    // Phase 1: Collect Morton indices and check if any cell is ALREADY claimed
    for (uint z = 0; z < subBlockDim.z; ++z) {
        for (uint y = 0; y < subBlockDim.y; ++y) {
            for (uint x = 0; x < subBlockDim.x; ++x) {
                uint mortonIdx = mortonEncode3D_3bit(subBlockOffsetInParent + uvec3(x,y,z));
                if (isCellClaimed_ReadOnly_AfterBarrier(mortonIdx)) {
                    return false; // Part of this block is already definitively claimed
                }
                cells_to_claim_morton[num_cells_this_block++] = mortonIdx;
            }
        }
    }

    // Phase 2: Attempt to atomically claim all collected cells from 0 to 1
    bool overall_claim_successful = true;
    for (uint i = 0; i < num_cells_this_block; ++i) {
        uint mortonCellIdx = cells_to_claim_morton[i];
        uint maskWordIndex = mortonCellIdx / 32u;
        uint bitInWord = mortonCellIdx % 32u;
        uint claim_bit = (1u << bitInWord);

        // Read the current word, prepare expected old and desired new
        uint current_word_val = shared_cell_claimed_mask[maskWordIndex]; // Non-atomic read before CAS
        uint expected_val = current_word_val & (~claim_bit); // Expect this bit to be 0
        uint desired_val  = current_word_val | claim_bit;   // Want to set this bit to 1

        if ((current_word_val & claim_bit) != 0) {
            // If bit is already 1, this sub-block cannot claim it
            overall_claim_successful = false;
            break;
        }

        uint original_word_val_from_cas = atomicCompSwap(shared_cell_claimed_mask[maskWordIndex], expected_val, desired_val);

        if (original_word_val_from_cas != expected_val) {
            // CAS failed. The word changed between our read and the CAS, or the bit was already 1.
            overall_claim_successful = false;
            break;
        }
        // If CAS succeeded, this thread set the bit. Record it for potential rollback.
        bits_successfully_set_by_this_attempt_mask[maskWordIndex] |= claim_bit;
    }

    if (!overall_claim_successful) {
        // Rollback: Clear ONLY the bits that THIS specific attempt had successfully set.
        for (uint i = 0; i < num_cells_this_block; ++i) {
            uint mortonCellIdx = cells_to_claim_morton[i];
            uint maskWordIndex = mortonCellIdx / 32u;
            uint bitInWord = mortonCellIdx % 32u;
            uint bit_to_clear = (1u << bitInWord);
            if ((bits_successfully_set_by_this_attempt_mask[maskWordIndex] & bit_to_clear) != 0) {
                atomicAnd(shared_cell_claimed_mask[maskWordIndex], ~bit_to_clear);
            }
        }
        return false;
    }
    return true; // All cells successfully claimed by this call
}

// Context-Aware Estimator - Parallelized across the workgroup/subgroup
BlockEstimate estimateGeometryForSubBlock_WithContext_Parallel(
    uvec3 parentBlockOrigin, uvec3 subBlockOffsetInParent, uvec3 subBlockDim
) {
    uint temp_edge_seen_mask[ESTIMATE_EDGE_HASH_WORDS]; // Thread-local
    for(uint i=0; i < ESTIMATE_EDGE_HASH_WORDS; ++i) temp_edge_seen_mask[i] = 0u;
    uint localUniqueVertexEstimate = 0;
    uint localPrimEstimate = 0;
    bool localHasUnclaimedActiveCells = false;

    uint cells_in_subblock = subBlockDim.x * subBlockDim.y * subBlockDim.z;
    uint cells_per_thread_estimate = (cells_in_subblock + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
    uint start_cell_loop_idx = gl_LocalInvocationIndex * cells_per_thread_estimate;
    uint end_cell_loop_idx = min(start_cell_loop_idx + cells_per_thread_estimate, cells_in_subblock);

    for (uint i_cell_local = start_cell_loop_idx; i_cell_local < end_cell_loop_idx; ++i_cell_local) {
        uvec3 cellOffsetInSubBlock;
        cellOffsetInSubBlock.x = i_cell_local % subBlockDim.x;
        cellOffsetInSubBlock.y = (i_cell_local / subBlockDim.x) % subBlockDim.y;
        cellOffsetInSubBlock.z = i_cell_local / (subBlockDim.x * subBlockDim.y);
        uvec3 coreCellOffsetInParent = subBlockOffsetInParent + cellOffsetInSubBlock;
        uint coreCellMortonIdx = mortonEncode3D_3bit(coreCellOffsetInParent);

        if (isCellClaimed_ReadOnly_AfterBarrier(coreCellMortonIdx)) continue;

        uint mc_case = shared_cell_mc_cases[coreCellMortonIdx];
        if (mc_case == 0 || mc_case == 255) continue;
        localHasUnclaimedActiveCells = true;

        uint primsInCell = 0;
        for (int v_idx = 0; v_idx < 12; ++v_idx) {
            int edgeIdx = getTriTableEntry(mc_case, v_idx);
            if (edgeIdx == -1) break;
            if ((v_idx % 3) == 0) primsInCell++;

            ivec3 p1_offset_in_cell; int edge_axis;
            switch (edgeIdx) { // Full 12-edge switch
                case 0:  p1_offset_in_cell=ivec3(0,0,0); edge_axis=0; break; case 1:  p1_offset_in_cell=ivec3(1,0,0); edge_axis=1; break;
                case 2:  p1_offset_in_cell=ivec3(0,1,0); edge_axis=0; break; case 3:  p1_offset_in_cell=ivec3(0,0,0); edge_axis=1; break;
                case 4:  p1_offset_in_cell=ivec3(0,0,1); edge_axis=0; break; case 5:  p1_offset_in_cell=ivec3(1,0,1); edge_axis=1; break;
                case 6:  p1_offset_in_cell=ivec3(0,1,1); edge_axis=0; break; case 7:  p1_offset_in_cell=ivec3(0,0,1); edge_axis=1; break;
                case 8:  p1_offset_in_cell=ivec3(0,0,0); edge_axis=2; break; case 9:  p1_offset_in_cell=ivec3(1,0,0); edge_axis=2; break;
                case 10: p1_offset_in_cell=ivec3(1,1,0); edge_axis=2; break; case 11: p1_offset_in_cell=ivec3(0,1,0); edge_axis=2; break;
                default: continue;
            }
            ivec3 globalEdgeP1 = ivec3(parentBlockOrigin + coreCellOffsetInParent) + p1_offset_in_cell;
            uint edgeGlobalHash = uint(globalEdgeP1.x) + uint(globalEdgeP1.y)*ubo.volumeDim.x + uint(globalEdgeP1.z)*ubo.volumeDim.x*ubo.volumeDim.y;
            edgeGlobalHash = (edgeGlobalHash << 2) | uint(edge_axis);
            uint hashWordIdx = (edgeGlobalHash % ESTIMATE_MAX_EDGES) / 32u;
            uint hashBitIdx  = (edgeGlobalHash % ESTIMATE_MAX_EDGES) % 32u;
            if (hashWordIdx < ESTIMATE_EDGE_HASH_WORDS) {
                uint hashBit = (1u << hashBitIdx);
                if ((temp_edge_seen_mask[hashWordIdx] & hashBit) == 0) {
                    temp_edge_seen_mask[hashWordIdx] |= hashBit;
                    localUniqueVertexEstimate++;
                }
            }
        }
        localPrimEstimate += primsInCell;
    }

    BlockEstimate totalEstimate;
    totalEstimate.estVertexCount = subgroupAdd(localUniqueVertexEstimate);
    totalEstimate.estPrimCount   = subgroupAdd(localPrimEstimate);
    // Use subgroupBallot to check if any thread found unclaimed active cells
    uvec4 ballot_result = subgroupBallot(localHasUnclaimedActiveCells);
    bool anyUnclaimedActive = (ballot_result.x != 0u || ballot_result.y != 0u || ballot_result.z != 0u || ballot_result.w != 0u);

    if (!anyUnclaimedActive) {
        totalEstimate.estVertexCount = 0;
        totalEstimate.estPrimCount = 0;
    }
    return totalEstimate;
}

uint getNumVertsEntry(uint mc_case) {
    // Add bounds check if needed
    // if (mc_case >= 256) return 0;
    uint index = mc_case;
    return uint(numVertices[index]); // Read int, cast to uint
}

BlockEstimate estimateGeometryForSubBlock_Parallel_NumVerts(
    uvec3 subBlockOffsetInParent, uvec3 subBlockDim
) {
    uint localVertexEstimate = 0;
    uint localPrimEstimate = 0;
    bool localHasUnclaimedActiveCells = false;

    uint cells_in_subblock = subBlockDim.x * subBlockDim.y * subBlockDim.z;
    uint cells_per_thread_estimate = (cells_in_subblock + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
    uint start_cell_loop_idx = gl_LocalInvocationIndex * cells_per_thread_estimate;
    uint end_cell_loop_idx = min(start_cell_loop_idx + cells_per_thread_estimate, cells_in_subblock);

    for (uint i_cell_local = start_cell_loop_idx; i_cell_local < end_cell_loop_idx; ++i_cell_local) {
        uvec3 cellOffsetInSubBlock;
        cellOffsetInSubBlock.x = i_cell_local % subBlockDim.x;
        cellOffsetInSubBlock.y = (i_cell_local / subBlockDim.x) % subBlockDim.y;
        cellOffsetInSubBlock.z = i_cell_local / (subBlockDim.x * subBlockDim.y);
        uvec3 coreCellOffsetInParent = subBlockOffsetInParent + cellOffsetInSubBlock;
        uint coreCellMortonIdx = mortonEncode3D_3bit(coreCellOffsetInParent);

        // Check against shared claim mask (needs visibility from prior levels)
        if (isCellClaimed_ReadOnly_AfterBarrier(coreCellMortonIdx)) continue;

        uint mc_case = shared_cell_mc_cases[coreCellMortonIdx]; // Read after barrier
        if (mc_case == 0 || mc_case == 255) continue;
        localHasUnclaimedActiveCells = true;

        // *** Use numVertsTable for vertex estimate ***
        localVertexEstimate += getNumVertsEntry(mc_case);

        // *** Count primitives using triTable ***
        uint primsInCell = 0;
        for (int v_idx = 0; v_idx < 15; v_idx += 3) { // Iterate potential triangles
                                                      if (getTriTableEntry(mc_case, v_idx) == -1) break;
                                                      primsInCell++;
        }
        localPrimEstimate += primsInCell;
    }

    // Reduce estimates across the workgroup
    BlockEstimate totalEstimate;
    totalEstimate.estVertexCount = subgroupAdd(localVertexEstimate);
    totalEstimate.estPrimCount   = subgroupAdd(localPrimEstimate);

    // Check if any thread found geometry using ballot
    uvec4 ballot_result = subgroupBallot(localHasUnclaimedActiveCells);
    bool anyUnclaimedActive = (ballot_result.x != 0u || ballot_result.y != 0u || ballot_result.z != 0u || ballot_result.w != 0u);

    if (!anyUnclaimedActive) {
        totalEstimate.estVertexCount = 0;
        totalEstimate.estPrimCount = 0;
    }
    // All threads in the workgroup now have the same totalEstimate
    return totalEstimate;
}

// --- Main ---
void main() {
    uint taskID = gl_WorkGroupID.x;
    uint localID = gl_LocalInvocationIndex;

    // --- Phase 0: Initialization ---
    if (localID == 0) { shared_num_final_sub_blocks = 0u; }
    uint items_per_thread_init = (CELLS_PER_INPUT_BLOCK + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
    for (uint i = 0; i < items_per_thread_init; ++i) {
        uint idx = localID + i * gl_WorkGroupSize.x;
        if (idx < CELLS_PER_INPUT_BLOCK) shared_cell_mc_cases[idx] = 0u;
    }
    uint words_in_mask = CELLS_PER_INPUT_BLOCK / 32u;
    items_per_thread_init = (words_in_mask + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
    for (uint i = 0; i < items_per_thread_init; ++i) { uint idx = localID + i * gl_WorkGroupSize.x; if (idx < words_in_mask) shared_cell_claimed_mask[idx] = 0u;}
    memoryBarrierShared(); barrier();

    // --- Phase 1: Parallel Per-Cell MC Case Calculation (Morton Order) ---
    if (taskID >= activeBlockCount.count) return;
    uvec3 parentBlockOrigin;
    uint blockIndex1D = blockIDs[taskID];
    uvec3 blockGridDim = ubo.blockGridDim.xyz;
    parentBlockOrigin.x = (blockIndex1D % blockGridDim.x) * INPUT_BLOCK_DIM_X;
    parentBlockOrigin.y = ((blockIndex1D / blockGridDim.x) % blockGridDim.y) * INPUT_BLOCK_DIM_Y;
    parentBlockOrigin.z = (blockIndex1D / (blockGridDim.x * blockGridDim.y)) * INPUT_BLOCK_DIM_Z;

    uint cells_per_thread_p1 = (CELLS_PER_INPUT_BLOCK + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
    for (uint i = 0; i < cells_per_thread_p1; ++i) {
        uint mortonCellIdx = localID + i * gl_WorkGroupSize.x;
        if (mortonCellIdx < CELLS_PER_INPUT_BLOCK) {
            uvec3 localCellCoord = mortonDecode3D_3bit(mortonCellIdx);
            ivec3 globalCellCoord = ivec3(parentBlockOrigin) + ivec3(localCellCoord);
            uint mc_case = 0; bool bdry = false;
            for (int bit = 0; bit < 8; ++bit) {
                ivec3 off = ivec3(bit & 1, (bit >> 1) & 1, (bit >> 2) & 1); ivec3 pos = globalCellCoord + off;
                if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, ivec3(ubo.volumeDim.xyz)))) { bdry = true; break; }
                uint v_val = imageLoad(volumeTexture, pos).r;
                if (float(v_val) >= ubo.isovalue) {
                    mc_case |= (1u << bit);
                }
            }
            if (!bdry) { shared_cell_mc_cases[mortonCellIdx] = mc_case; }
        }
    }
    memoryBarrierShared(); barrier();

    // --- Phase 2: Parallel Hierarchical Partitioning & Greedy Claiming ---
    for (uint levelDim = INPUT_BLOCK_DIM_X; levelDim >= MIN_SUB_BLOCK_DIM; levelDim /= 2u) {
        memoryBarrierShared(); // Ensure claims from previous level (or init) are visible
        barrier();

        uint numSubPerAxis = INPUT_BLOCK_DIM_X / levelDim;
        uint numSubBlocksAtThisLevel = numSubPerAxis * numSubPerAxis * numSubPerAxis;

        uint subBlocksToEvalPerThread = (numSubBlocksAtThisLevel + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
        uint startSubBlockLoopIdx = localID * subBlocksToEvalPerThread;
        uint endSubBlockLoopIdx = min(startSubBlockLoopIdx + subBlocksToEvalPerThread, numSubBlocksAtThisLevel);

        for (uint sb_1d_idx = startSubBlockLoopIdx; sb_1d_idx < endSubBlockLoopIdx; ++sb_1d_idx) {
            uvec3 sb_grid_coord;
            sb_grid_coord.x = sb_1d_idx % numSubPerAxis;
            sb_grid_coord.y = (sb_1d_idx / numSubPerAxis) % numSubPerAxis;
            sb_grid_coord.z = sb_1d_idx / (numSubPerAxis * numSubPerAxis);
            uvec3 subBlockOffsetInParent = sb_grid_coord * levelDim;

            // Greedy Check: Test representative cell (origin) if covered by a larger block
            if (isCellClaimed_ReadOnly_AfterBarrier(mortonEncode3D_3bit(subBlockOffsetInParent))) {
                continue;
            }

            // All threads in the workgroup collaboratively estimate for THIS sub-block
            // The result will be identical for all threads after subgroupAdd.
//            BlockEstimate estimate = estimateGeometryForSubBlock_WithContext_Parallel(parentBlockOrigin, subBlockOffsetInParent, uvec3(levelDim));
            BlockEstimate estimate = estimateGeometryForSubBlock_Parallel_NumVerts(subBlockOffsetInParent, uvec3(levelDim));
            bool fits = (estimate.estVertexCount <= MAX_MESHLET_VERTICES && estimate.estPrimCount <= MAX_MESHLET_PRIMITIVES);
            bool hasGeometry = (estimate.estVertexCount > 0 || estimate.estPrimCount > 0);

            if (hasGeometry && fits) {
                // This thread (as representative for this sb_1d_idx) attempts to claim.
                if (tryClaimCells_SubBlock_AllOrNothing_Atomic(subBlockOffsetInParent, uvec3(levelDim))) {
                    uint final_idx = atomicAdd(shared_num_final_sub_blocks, 1u);
                    if (final_idx < MAX_MESH_TASKS_PER_TASK_SHADER) {
                        shared_final_sub_blocks[final_idx].subBlockOffsetInParent = subBlockOffsetInParent;
                        shared_final_sub_blocks[final_idx].subBlockDim = uvec3(levelDim);
                        shared_final_sub_blocks[final_idx].estimate = estimate;
                    } else {
                        // List full. CRITICAL: Rollback the claim!
                        // This is the hard part. The cells are claimed but we can't store the record.
                        // The current tryClaim... does a rollback if its *own* CAS loop fails,
                        // but not if shared_num_final_sub_blocks is full *after* a successful claim.
                        // For now, we accept this limitation: cells might be claimed but not used if list overflows.
                        // A truly robust solution would need to undo the claim here.
                        // The atomicAdd(-1u) was removed as it's unsafe.
                        rollbackClaimedCells(subBlockOffsetInParent, uvec3(levelDim));
                        // clamp the counter so it never exceeds the limit
                        atomicAdd(shared_num_final_sub_blocks, uint(-1));
                        debugPrintfEXT("warning");
                    }
                }
            } else if (hasGeometry && !fits && levelDim == MIN_SUB_BLOCK_DIM) {
                // Smallest level, doesn't fit, but has geometry. Add if claimable.
                if (tryClaimCells_SubBlock_AllOrNothing_Atomic(subBlockOffsetInParent, uvec3(levelDim))) {
                    uint final_idx = atomicAdd(shared_num_final_sub_blocks, 1u);
                    if (final_idx < MAX_MESH_TASKS_PER_TASK_SHADER) {
                        shared_final_sub_blocks[final_idx].subBlockOffsetInParent = subBlockOffsetInParent;
                        shared_final_sub_blocks[final_idx].subBlockDim = uvec3(levelDim);
                        shared_final_sub_blocks[final_idx].estimate.estVertexCount = min(estimate.estVertexCount, MAX_MESHLET_VERTICES);
                        shared_final_sub_blocks[final_idx].estimate.estPrimCount = min(estimate.estPrimCount, MAX_MESHLET_PRIMITIVES);
                    } else {
                        rollbackClaimedCells(subBlockOffsetInParent, uvec3(levelDim));
                        // clamp the counter so it never exceeds the limit
                        atomicAdd(shared_num_final_sub_blocks, uint(-1));
                        debugPrintfEXT("warn");
                    }
                }
            }
        }
        memoryBarrierShared(); // Ensure all shared_cell_claimed_mask writes from THIS level are visible
        barrier();           // Synchronize all threads before processing next smaller level
    }

    // --- Phase 3 & 4: Allocation & Dispatch (Leader Thread) ---
    memoryBarrierShared(); barrier(); // Ensure shared_num_final_sub_blocks is up-to-date
    uint final_task_count_snapshot = shared_num_final_sub_blocks; // Read once by leader
    uint final_task_count = min(final_task_count_snapshot, MAX_MESH_TASKS_PER_TASK_SHADER); // Clamp

    if (localID == 0 && final_task_count > 0) {
        uint total_verts_needed = 0; uint total_indices_needed = 0;
        for (uint i = 0; i < final_task_count; ++i) {
            total_verts_needed += shared_final_sub_blocks[i].estimate.estVertexCount;
            total_indices_needed += shared_final_sub_blocks[i].estimate.estPrimCount * 3u;
        }
        uint baseDescOffset = atomicAdd(meshletCounter, final_task_count);
        uint baseVertOffset = atomicAdd(vertexCounter, total_verts_needed);
        uint baseIdxOffset = atomicAdd(indexCounter, total_indices_needed);
        uint current_vert_offset = baseVertOffset; uint current_idx_offset = baseIdxOffset;
        for (uint i = 0; i < final_task_count; ++i) {
            taskPayload[i].parentBlockOrigin = parentBlockOrigin;
            taskPayload[i].subBlockOffset = shared_final_sub_blocks[i].subBlockOffsetInParent;
            taskPayload[i].subBlockDim = shared_final_sub_blocks[i].subBlockDim;
            taskPayload[i].baseDescriptorOffset = baseDescOffset + i;
            taskPayload[i].baseVertexOffset = current_vert_offset;
            taskPayload[i].baseIndexOffset = current_idx_offset;
            current_vert_offset += shared_final_sub_blocks[i].estimate.estVertexCount;
            current_idx_offset += shared_final_sub_blocks[i].estimate.estPrimCount * 3u;
        }
        EmitMeshTasksEXT(final_task_count, 1, 1); // Dispatch with clamped count
    }
}