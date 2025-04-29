#version 460 core
#extension GL_EXT_mesh_shader : require // Enable Mesh Shader extension
#extension GL_KHR_shader_subgroup_arithmetic : require // For subgroup operations
#extension GL_KHR_shader_subgroup_ballot : require // For subgroup operations (optional here, but useful)
#extension GL_KHR_shader_subgroup_shuffle : require // For subgroup operations (optional here, but useful)
#extension GL_EXT_shader_atomic_int64 : enable // If using 64-bit offsets/counters
#extension GL_EXT_buffer_reference : require // If using buffer device addresses
#extension GL_EXT_scalar_block_layout : enable // If using scalar layout for UBO/SSBO


// --- Workgroup Layout ---
// One task workgroup per active block ID passed from the filtering stage.
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// --- Constants ---
#define MAX_MESHLET_VERTICES 128 // Target from paper/config
#define MAX_MESHLET_PRIMITIVES 256 // Target from paper/config (adjust based on HW limits if needed)
#define BLOCK_DIM_X 8 // Must match C++ PushConstants/UBO
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 8
#define SUBGROUP_SIZE 32 // Assumed warp/subgroup size

// --- Descriptor Set Bindings (ASSUMES C++ SIDE IS UPDATED) ---
layout(set = 0, binding = 0, scalar) uniform ExtractionConstantsUBO {
    uvec4 volumeDim;     // x, y, z contain dimensions
    uvec4 blockGridDim;  // x, y, z contain grid dimensions
    float isovalue;
// Add other constants if needed (e.g. blockDim if not hardcoded)
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeTexture; // Volume Data (uint8)

layout(set = 0, binding = 2, scalar) buffer CompactedBlockIDs { uint blockIDs[]; }; // Input block IDs

layout(set = 0, binding = 3, scalar) buffer MarchingCubesTable { int triTable[]; }; // MC triTable[256][16]

// Output Buffers (Binding points must match C++ and Mesh Shader)
// Using atomic counters at the start of each buffer for allocation.
// Alternatively, pass separate atomic counter buffers.
// Ensure C++ allocates space for the counter (e.g. + sizeof(uint/uint64))
// and initializes counter to 0.
layout(set = 0, binding = 4, scalar) buffer VertexBuffer { uint vertexCounter; /* vec3 pos, vec3 norm */ }; // Output Vertices
layout(set = 0, binding = 5, scalar) buffer IndexBuffer { uint indexCounter; /* uint indices[] */ };       // Output Indices (local to meshlet)
layout(set = 0, binding = 6, scalar) buffer MeshletDescriptorBuffer { uint meshletCounter; /* MeshletDescriptor descriptors[] */ }; // Output Descriptors

// --- Task Shader -> Mesh Shader Payload ---
// Defines data passed per dispatched mesh shader workgroup
struct SubBlockInfo {
    uvec3 blockOrigin;       // Origin of the parent block in voxel coords
    uvec3 subBlockOffset;    // Offset of this sub-block within the parent (e.g., 0 or 4)
    uvec3 subBlockDim;       // Dimensions of this sub-block (e.g., 8x8x8 or 4x4x4)
    uint baseVertexOffset;   // Start offset in global VertexBuffer
    uint baseIndexOffset;    // Start offset in global IndexBuffer
    uint baseDescriptorOffset;// Index in global MeshletDescriptorBuffer
    uint activeCellCount;    // Number of active cells in this sub-block
// uint activeCellIndices[...]; // OPTIONAL: Pass list of active cell indices within sub-block
};
// Maximum number of mesh tasks launched by one task shader task (e.g., 8 for split)
#define MAX_SUB_BLOCKS 8
taskPayloadSharedEXT SubBlockInfo taskPayload[MAX_SUB_BLOCKS];

// --- Shared Memory for Analysis ---
shared uint shared_activeCellCount[SUBGROUP_SIZE];
shared uint shared_estVertexCount[SUBGROUP_SIZE];
shared uint shared_estPrimitiveCount[SUBGROUP_SIZE];
// Shared memory for storing active cell indices within the block (if not passed in payload)
#define MAX_CELLS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)
// shared uint shared_activeCellIndices[MAX_CELLS_PER_BLOCK];
// shared uint shared_activeCellListCounter;


// --- Helper Functions ---
uint MCHash(ivec3 cellCoord, uvec3 blockDim) {
    // Simple hash, assumes coords are within block dims
    return cellCoord.z * blockDim.x * blockDim.y + cellCoord.y * blockDim.x + cellCoord.x;
}

uint estimateGeometry(uint mc_case) {
    // Simple estimation based on triTable entries until -1 is found
    // Returns packed count: (primitiveCount << 16) | vertexCount (approx)
    // This is a basic estimate; real count depends on vertex sharing.
    uint primCount = 0;
    uint vertCount = 0; // Very rough vertex count estimate
    int idx = 0;
    while (idx < 15 && triTable[mc_case * 16 + idx] != -1) {
        primCount++;
        idx += 3;
    }
    primCount /= 3;
    // Vertex count estimate is harder without topology check, use primitive count as proxy
    if (primCount > 0) {
        vertCount = primCount + 2; // Heuristic: roughly V = P + 2 for MC patches
    }
    return (primCount << 16) | vertCount;
}


void main() {
    uint taskID = gl_WorkGroupID.x; // Each task processes one active block ID
    uint localID = gl_LocalInvocationIndex;

    // Initialize shared memory counters if needed (only thread 0)
    if (localID == 0) {
        // shared_activeCellListCounter = 0;
    }
    // Clear per-thread shared accumulators
    shared_activeCellCount[localID] = 0;
    shared_estVertexCount[localID] = 0;
    shared_estPrimitiveCount[localID] = 0;
    barrier(); // Ensure initialization is done

    // Get the 1D block index this task is responsible for
    uint blockIndex1D = blockIDs[taskID];

    // Convert 1D block index to 3D block coordinates
    uvec3 blockCoord;
    blockCoord.x = blockIndex1D % ubo.blockGridDim.x;
    blockCoord.y = (blockIndex1D / ubo.blockGridDim.x) % ubo.blockGridDim.y;
    blockCoord.z = blockIndex1D / (ubo.blockGridDim.x * ubo.blockGridDim.y);

    uvec3 blockOrigin = blockCoord * uvec3(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);

    // --- Analyze Cells within the Block (Parallel over threads) ---
    // Divide cells among threads in the workgroup
    uint cellsPerThread = (MAX_CELLS_PER_BLOCK + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    uint startCell = localID * cellsPerThread;
    uint endCell = min(startCell + cellsPerThread, MAX_CELLS_PER_BLOCK);

    uint localActiveCellCount = 0;
    uint localEstVertexCount = 0;
    uint localEstPrimCount = 0;

    for (uint cellIdx1D = startCell; cellIdx1D < endCell; ++cellIdx1D) {
        // Convert 1D cell index (within block) to 3D local coordinates
        ivec3 localCellCoord;
        localCellCoord.x = int(cellIdx1D % BLOCK_DIM_X);
        localCellCoord.y = int((cellIdx1D / BLOCK_DIM_X) % BLOCK_DIM_Y);
        localCellCoord.z = int(cellIdx1D / (BLOCK_DIM_X * BLOCK_DIM_Y));

        ivec3 globalCellCoord = ivec3(blockOrigin) + localCellCoord;

        // Marching Cubes Case Calculation
        uint mc_case = 0;
        float cornerValues[8];
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerCoord = globalCellCoord + cornerOffset;

            // Clamp coordinates to volume bounds
            cornerCoord = clamp(cornerCoord, ivec3(0), ivec3(ubo.volumeDim.xyz - 1u));

            // Sample volume (assuming R8 format normalized to float 0-1)
            float val = imageLoad(volumeTexture, cornerCoord).r / 255.0f; // Adjust scaling if format differs
            cornerValues[i] = val; // Store for potential later use (gradient calc)

            if (val >= ubo.isovalue / 255.0f) { // Compare normalized isovalue
                mc_case |= (1 << i);
            }
        }

        // Check if cell is active (generates geometry)
        if (mc_case != 0 && mc_case != 255) {
            localActiveCellCount++;
            // Estimate geometry generated by this cell
            uint counts = estimateGeometry(mc_case);
            localEstPrimCount += (counts >> 16);
            localEstVertexCount += (counts & 0xFFFF);

            // OPTIONAL: Store active cell index if passing list via payload
            // uint listIdx = atomicAdd(shared_activeCellListCounter, 1);
            // if (listIdx < MAX_CELLS_PER_BLOCK) {
            //     shared_activeCellIndices[listIdx] = cellIdx1D;
            // }
        }
    }

    // Store local results in shared memory
    shared_activeCellCount[localID] = localActiveCellCount;
    shared_estVertexCount[localID] = localEstVertexCount;
    shared_estPrimitiveCount[localID] = localEstPrimCount;

    // --- Aggregate results across workgroup ---
    barrier(); // Wait for all threads to finish cell analysis

    // Use subgroupAdd or a shared memory reduction (if workgroup > subgroup)
    uint totalActiveCells = subgroupAdd(localActiveCellCount);
    uint totalEstVertices = subgroupAdd(localEstVertexCount);
    uint totalEstPrims = subgroupAdd(localEstPrimCount);

    // Only one thread (e.g., localID 0) performs the final steps
    if (subgroupElect()) { // Use subgroupElect for clarity, leader works
        uint numMeshTasksToLaunch = 0;
        uint totalAllocatedVertices = 0;
        uint totalAllocatedIndices = 0;
        uint totalAllocatedMeshlets = 0;

        // --- Partitioning Decision (Simplified) ---
        if (totalActiveCells > 0 && (totalEstVertices > MAX_MESHLET_VERTICES || totalEstPrims > MAX_MESHLET_PRIMITIVES)) {
            // Estimated geometry exceeds limits for a single meshlet. Split into 8 sub-blocks.
            // NOTE: This is a *very* basic split. A better approach would analyze
            // the geometry distribution more carefully or use the paper's divisive method.
            // For this simple split, we don't recalculate precise geometry per sub-block,
            // we just assume each sub-block needs space and dispatch a task.
            // The mesh shader will handle empty sub-blocks if needed.

            numMeshTasksToLaunch = 8;
            uvec3 subBlockDim = uvec3(BLOCK_DIM_X / 2, BLOCK_DIM_Y / 2, BLOCK_DIM_Z / 2);

            // Estimate total space needed for potentially 8 meshlets (over-estimate)
            totalAllocatedVertices = totalEstVertices; // Use total estimate
            totalAllocatedIndices = totalEstPrims * 3; // Use total estimate
            totalAllocatedMeshlets = numMeshTasksToLaunch;

            // Atomically allocate space from global counters
            // Add guards if atomic functions are not supported or if offsets might exceed 32-bit
            uint baseDescOffset = atomicAdd(meshletCounter, numMeshTasksToLaunch);
            uint baseVertOffset = atomicAdd(vertexCounter, totalAllocatedVertices);
            uint baseIdxOffset = atomicAdd(indexCounter, totalAllocatedIndices);

            // Prepare payload for each of the 8 sub-blocks
            for (uint i = 0; i < 8; ++i) {
                taskPayload[i].blockOrigin = blockOrigin;
                taskPayload[i].subBlockOffset = uvec3((i & 1) * subBlockDim.x, ((i >> 1) & 1) * subBlockDim.y, ((i >> 2) & 1) * subBlockDim.z);
                taskPayload[i].subBlockDim = subBlockDim;
                // Assign offsets (mesh shader needs to know its portion, but allocation was total)
                // For simplicity, pass the base offset. Mesh shader needs more info or atomics.
                // Passing base offsets and letting mesh shader use atomics *within* the sub-block is complex.
                // Passing calculated slice offsets is better but requires precise count per sub-block.
                // Let's pass the base offsets and the mesh shader index.
                taskPayload[i].baseVertexOffset = baseVertOffset; // Mesh shader needs to further offset based on its index/atomics
                taskPayload[i].baseIndexOffset = baseIdxOffset;   // Mesh shader needs to further offset based on its index/atomics
                taskPayload[i].baseDescriptorOffset = baseDescOffset + i; // Each gets one descriptor slot
                taskPayload[i].activeCellCount = 0; // Task shader doesn't have precise per-sub-block count here
                // OPTIONAL: Populate active cell list slice for this sub-block if generated earlier
            }

        } else if (totalActiveCells > 0) {
            // Fits within limits, create one meshlet for the whole block.
            numMeshTasksToLaunch = 1;
            totalAllocatedVertices = totalEstVertices;
            totalAllocatedIndices = totalEstPrims * 3;
            totalAllocatedMeshlets = numMeshTasksToLaunch;

            // Atomically allocate space
            uint baseDescOffset = atomicAdd(meshletCounter, numMeshTasksToLaunch);
            uint baseVertOffset = atomicAdd(vertexCounter, totalAllocatedVertices);
            uint baseIdxOffset = atomicAdd(indexCounter, totalAllocatedIndices);

            // Prepare payload for the single meshlet
            taskPayload[0].blockOrigin = blockOrigin;
            taskPayload[0].subBlockOffset = uvec3(0);
            taskPayload[0].subBlockDim = uvec3(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
            taskPayload[0].baseVertexOffset = baseVertOffset;
            taskPayload[0].baseIndexOffset = baseIdxOffset;
            taskPayload[0].baseDescriptorOffset = baseDescOffset;
            taskPayload[0].activeCellCount = totalActiveCells;
            // OPTIONAL: Pass full active cell list if generated earlier
        }

        // --- Emit Mesh Tasks ---
        if (numMeshTasksToLaunch > 0) {
            EmitMeshTasksEXT(numMeshTasksToLaunch, 1, 1);
        }
    }
    // Task shader execution completes here.
}