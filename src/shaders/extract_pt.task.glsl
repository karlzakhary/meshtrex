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
};

// Max potential mesh tasks if 8x8x8 fully subdivides to 2x2x2 -> 4*4*4 = 64
#define MAX_MESH_TASKS_PER_TASK_SHADER 64
taskPayloadSharedEXT SubBlockInfo taskPayload[MAX_MESH_TASKS_PER_TASK_SHADER];

// --- Shared Memory ---
// Store per-cell estimates (vertex count in lower 16 bits, prim count in upper 16)
shared uint shared_cell_estimates[CELLS_PER_BLOCK];
// Store final sub-block decisions before dispatch
shared SubBlockInfo shared_final_sub_blocks[MAX_MESH_TASKS_PER_TASK_SHADER];
shared uint shared_num_final_sub_blocks; // Atomic counter for filling the final list

// --- Helper Functions ---
// Offsets within the combined MC table buffer (in uint elements)
const uint triTableOffsetElements = 0;
const uint numVertsTableOffsetElements = 256 * 16;

int getTriTableEntry(uint caseIndex, uint vertexNum) {
    return data[triTableOffsetElements + caseIndex * 16 + vertexNum];
}
// Removed edgeTable access - not used here
uint getNumVertsEntry(uint caseIndex) {
    return uint(data[numVertsTableOffsetElements + caseIndex]);
}

// Estimate geometry for a SINGLE cell using numVertsTable
// Returns packed uint: (primitiveCount << 16) | vertexCount
uint estimateGeometryForCell(uint mc_case) {
    if (mc_case == 0 || mc_case == 255) return 0;

    uint vertCount = getNumVertsEntry(mc_case); // Get accurate vertex count

    // Estimate primitive count (still need triTable for this)
    uint primCount = 0;
    uint idx = 0;
    while (idx < 12 && getTriTableEntry(mc_case, idx) != -1) {
        primCount++;
        idx += 3;
    }
    primCount /= 3;

    // Return packed estimate (ensure counts fit in 16 bits if necessary)
    return (min(primCount, 0xFFFFu) << 16) | min(vertCount, 0xFFFFu);
}

BlockEstimate estimateGeometryForSubBlock(
    uvec3 subBlockOffset, // Offset of this sub-block within the parent 8x8x8 block
    uvec3 subBlockDim     // Dimensions of this sub-block (e.g., 8x8x8, 4x4x4, 2x2x2)
) {
    BlockEstimate estimate;
    estimate.estVertexCount = 0;
    estimate.estPrimCount = 0;

    // Loop through all cells within the logical sub-block dimensions
    for (uint z = 0; z < subBlockDim.z; ++z) {
        for (uint y = 0; y < subBlockDim.y; ++y) {
            for (uint x = 0; x < subBlockDim.x; ++x) {
                // Calculate the 1D index into the original 8x8x8 shared_cell_estimates array
                // corresponding to the current cell within this sub-block.
                uint cell_lx = subBlockOffset.x + x;
                uint cell_ly = subBlockOffset.y + y;
                uint cell_lz = subBlockOffset.z + z;

                uint cellIdx1D = cell_lz * BLOCK_DIM_X * BLOCK_DIM_Y +
                cell_ly * BLOCK_DIM_X +
                cell_lx;

                // Bounds check (optional, but good practice if logic is complex)
                if (cellIdx1D < CELLS_PER_BLOCK) { // CELLS_PER_BLOCK = 512
                                                   uint idx = (subBlockOffset.z + z) * BLOCK_DIM_Y * BLOCK_DIM_X + (subBlockOffset.y + y) * BLOCK_DIM_X + (subBlockOffset.x + x);
                                                   uint packed = shared_cell_estimates[idx];
                                                   estimate.estPrimCount   += (packed >> 16);
                                                   estimate.estVertexCount += (packed & 0xFFFFu);
                }
            }
        }
    }
    return estimate;
}

// --- Main ---
void main() {
    uint taskID = gl_WorkGroupID.x;
    uint localID = gl_LocalInvocationIndex;

    // Initialize shared memory
    if (localID == 0) {
        shared_num_final_sub_blocks = 0;
    }
    // Initialize per-cell estimates (distribute work)
    for (uint i = localID; i < CELLS_PER_BLOCK; i += gl_WorkGroupSize.x) {
        shared_cell_estimates[i] = 0;
    }
    barrier(); // Ensure initialization is complete

    // --- Phase 1: Calculate Per-Cell Estimates for the 8x8x8 Block ---
    uint blockIndex1D = blockIDs[taskID]; // Assume taskID is valid index
    uvec3 blockCoord; // Calculate 3D block coord from 1D index
    blockCoord.x = blockIndex1D % ubo.blockGridDim.x;
    blockCoord.y = (blockIndex1D / ubo.blockGridDim.x) % ubo.blockGridDim.y;
    blockCoord.z = blockIndex1D / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uvec3 blockOrigin = blockCoord * uvec3(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);

    uint cellsPerThread = (CELLS_PER_BLOCK + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    uint startCell = localID * cellsPerThread;
    uint endCell   = min(startCell + cellsPerThread, CELLS_PER_BLOCK);
    for (uint idx = startCell; idx < endCell; ++idx) {
        ivec3 c;
        c.x = int(idx % BLOCK_DIM_X);
        c.y = int((idx / BLOCK_DIM_X) % BLOCK_DIM_Y);
        c.z = int(idx / (BLOCK_DIM_X * BLOCK_DIM_Y));
        ivec3 gc = ivec3(blockOrigin) + c;
        uint mc_case = 0;
        bool bdry = false;
        for (int bit = 0; bit < 8; ++bit) {
            ivec3 off = ivec3(bit & 1, (bit >> 1) & 1, (bit >> 2) & 1);
            ivec3 pos = gc + off;
            if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, ivec3(ubo.volumeDim.xyz)))) { bdry = true; break; }
            uint v = imageLoad(volumeTexture, pos).r;
            if (float(v) >= ubo.isovalue) mc_case |= (1 << bit);
        }
        if (!bdry) shared_cell_estimates[idx] = estimateGeometryForCell(mc_case);
    }
    barrier();

    // --- Region Analysis: hierarchical, parallel via subgroup ---
    const uvec3 dims[3] = { uvec3(8u), uvec3(4u), uvec3(2u) };
    for (int level = 0; level < 3; ++level) {
        uvec3 subDim = dims[level];
        uint numSubX = BLOCK_DIM_X / subDim.x;
        uint numSubY = BLOCK_DIM_Y / subDim.y;
        uint numSubZ = BLOCK_DIM_Z / subDim.z;
        uint totalSubs = numSubX * numSubY * numSubZ;

        for (uint sb = localID; sb < totalSubs; sb += SUBGROUP_SIZE) {
            uint x = sb % numSubX;
            uint y = (sb / numSubX) % numSubY;
            uint z = sb / (numSubX * numSubY);
            uvec3 offset = uvec3(x, y, z) * subDim;
            BlockEstimate est = estimateGeometryForSubBlock(offset, subDim);
            bool nonEmpty = (est.estVertexCount > 0 || est.estPrimCount > 0);
            bool fits     = nonEmpty && est.estVertexCount <= MAX_MESHLET_VERTICES && est.estPrimCount <= MAX_MESHLET_PRIMITIVES;
            uvec4 mask    = subgroupBallot(fits);
            uint count    = subgroupBallotBitCount(mask);
            uint laneIdx  = gl_SubgroupInvocationID;
            uint before   = subgroupBallotBitCount(mask & ((~0u) >> (SUBGROUP_SIZE - laneIdx)));
            if (fits) {
                uint outIdx = atomicAdd(shared_num_final_sub_blocks, 1);
                if (outIdx < MAX_MESH_TASKS_PER_TASK_SHADER) {
                    shared_final_sub_blocks[outIdx].blockOrigin        = blockOrigin;
                    shared_final_sub_blocks[outIdx].subBlockOffset     = offset;
                    shared_final_sub_blocks[outIdx].subBlockDim        = subDim;
                }
            }
        }
        barrier(); // wait all levels before next
    }

    // --- Allocate & Dispatch ---
    barrier();
    uint finalCount = shared_num_final_sub_blocks;
    if (finalCount > 0) {
        // Compute total allocations
        uint totalV = 0, totalP = 0;
        for (uint i = localID; i < finalCount; i += gl_WorkGroupSize.x) {
            BlockEstimate e = estimateGeometryForSubBlock(shared_final_sub_blocks[i].subBlockOffset,
                                                          shared_final_sub_blocks[i].subBlockDim);
            totalV += min(e.estVertexCount, MAX_MESHLET_VERTICES);
            totalP += min(e.estPrimCount,   MAX_MESHLET_PRIMITIVES);
        }
        // reduce across workgroup
        uint groupV = subgroupAdd(totalV);
        uint groupP = subgroupAdd(totalP);
        if (subgroupElect()) { // one per subgroup
                               uint baseDesc = atomicAdd(meshletCounter, finalCount);
                               uint baseVert = atomicAdd(vertexCounter, groupV);
                               uint baseIdx  = atomicAdd(indexCounter, groupP * 3);
                               // store bases into shared for dispatch
                               shared_final_sub_blocks[0].baseDescriptorOffset = baseDesc;
                               shared_final_sub_blocks[0].baseVertexOffset     = baseVert;
                               shared_final_sub_blocks[0].baseIndexOffset      = baseIdx;
        }
        barrier();

        uint baseDesc = shared_final_sub_blocks[0].baseDescriptorOffset;
        uint baseVert = shared_final_sub_blocks[0].baseVertexOffset;
        uint baseIdx  = shared_final_sub_blocks[0].baseIndexOffset;

        // emit each task
        for (uint i = localID; i < finalCount; i += gl_WorkGroupSize.x) {
            SubBlockInfo sb = shared_final_sub_blocks[i];
            sb.baseDescriptorOffset = baseDesc + i;
            sb.baseVertexOffset     = baseVert + i * MAX_MESHLET_VERTICES;
            sb.baseIndexOffset      = baseIdx + i * MAX_MESHLET_PRIMITIVES * 3;
            taskPayload[i] = sb;
        }
        barrier();
        if (localID == 0) {
            EmitMeshTasksEXT(finalCount, 1, 1);
        }
    }
}