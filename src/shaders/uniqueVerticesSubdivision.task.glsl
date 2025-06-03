#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require // For subgroupBroadcastFirst
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
#define MAX_VERTS_PER_CELL 12 // Max vertices a single cell can produce before sharing
#define MAX_PRIMS_PER_CELL 5  // Max triangles a single cell can produce
#define MAX_MESHLET_VERTS 256u // Max unique vertices a meshlet aims to produce
#define MAX_MESHLET_PRIMS 256u // Max primitives a meshlet aims to produce
#define MAX_TRI_INDICES 16    // Assumed padding for triTable entries per cube case
#define WORKGROUP_SIZE 32
#define EFFECTIVE_MAX_MESHLET_VERTS (MAX_MESHLET_VERTS / 8) // Example: 128 if target is 256
#define EFFECTIVE_MAX_MESHLET_PRIMS (MAX_MESHLET_PRIMS / 8) // Example: 128

// --- Structures ---
struct TaskPayload {
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletDescOffset;
    uvec3 blockOrigin;      // Absolute origin in volume coordinates OF THE SUB-BLOCK
    uvec3 subBlockDim;      // Dimensions of the sub-block for the mesh shader
    uint taskId;            // Original compactedBlockID for debugging
};

taskPayloadSharedEXT TaskPayload taskPayloadOut;

// --- Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;         // Default dimensions of blocks (e.g., 8,8,8 or 16,16,16)
    uvec4 blockGridDim;     // Grid dimensions in terms of ubo.blockDim blocks
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;

layout(set = 0, binding = 2, std430) readonly buffer ActiveBlockCount_SSBO {
    uint count;
} activeBlockCountBuffer;

layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs_SSBO {
    uint compactedBlkArray[];
} blockIds;

// Using classic Marching Cubes triTable for triangle definitions
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO {
    int triTable[]; // Classic MC: 256 cases * 16 ints (max 5 tris * 3 verts + terminator)
} mc;

// Binding 5 would be for edgeTable, but it's used by the Mesh Shader, not Task Shader for this logic
// layout(set=0, binding=5, std430) readonly buffer EdgeTable_SSBO { uint edgeTable[]; } et;

// Global counters for allocating space in vertex/index buffers
layout(set = 0, binding = 11, std430) buffer VertexCount_SSBO { uint vertexCounter; } vCount;
layout(set = 0, binding = 13, std430) buffer IndexCount_SSBO { uint indexCounter; } iCount;
// Global counter for allocating space for meshlet descriptors (task shader allocates, mesh shader fills)
layout(set = 0, binding = 12, std430) buffer MeshletDescCount_SSBO { uint meshletCounter; } meshletDescCount;


// --- Helper Functions ---
// Samples a Marching Cubes case for a cell at its global origin
uint sampleMcCase(uvec3 cell_global_origin) {
    uint cubeCase = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 corner_offset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
        ivec3 voxel_coord = ivec3(cell_global_origin) + corner_offset;
        uint val = 0;
        // Boundary checks for sampling volume image
        if (all(greaterThanEqual(voxel_coord, ivec3(0))) && all(lessThan(voxel_coord, ivec3(ubo.volumeDim.xyz)))) {
            val = imageLoad(volumeImage, voxel_coord).r;
        }
        if (float(val) >= ubo.isovalue) {
            cubeCase |= (1 << i);
        }
    }
    return cubeCase;
}

// Estimates vertices for a single cell by counting unique edges from its triTable entry.
// This is a pre-sharing estimate for subdivision purposes.
uint countCellVertices(uint cubeCase) {
    uint flags = 0u; // Bitmask to track unique edges
    uint vertCount = 0u;
    int base = int(cubeCase) * MAX_TRI_INDICES; // Assumes triTable is padded
    for (int i = 0; i < MAX_TRI_INDICES; ++i) {
        int edgeID = mc.triTable[base + i];
        if (edgeID < 0) break; // Terminator for the list of edges for this case
        uint mask = 1u << uint(edgeID); // Create a mask for this edge
        if ((flags & mask) == 0u) { // If this edge hasn't been counted yet
            flags |= mask;
            vertCount++;
        }
    }
    return vertCount;
}

// Counts triangles for a single cell from its triTable entry.
uint countCellPrimitives(uint cubeCase) {
    uint primCount = 0;
    int base = int(cubeCase) * MAX_TRI_INDICES; // Assumes triTable is padded
    for (int i = 0; i < MAX_TRI_INDICES; i += 3) { // Each triangle uses 3 edge indices
        if (mc.triTable[base + i] < 0) break; // Terminator or start of padding
        primCount++;
    }
    return primCount;
}

// Estimates total vertices in a given sub-block (run by one thread)
uint estimateVertsInSubBlock(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) {
    uint sum = 0;
    for (uint z = 0; z < subBlockDimensions.z; ++z) {
        for (uint y = 0; y < subBlockDimensions.y; ++y) {
            for (uint x = 0; x < subBlockDimensions.x; ++x) {
                uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                uint cubeCase = sampleMcCase(cellGlobalOrigin);
                if (cubeCase != 0 && cubeCase != 255) {
                    sum += countCellVertices(cubeCase);
                    if (sum > EFFECTIVE_MAX_MESHLET_VERTS + WORKGROUP_SIZE) { // Add a little buffer for parallel estimation inaccuracy
                        // debugPrintfEXT("TS EST_VERTS_EARLY_EXIT: Sum %u exceeded %u for block %u,%u,%u dim %u,%u,%u\n",
                        //    sum, EFFECTIVE_MAX_MESHLET_VERTS, subBlockGlobalOrigin.x,subBlockGlobalOrigin.y,subBlockGlobalOrigin.z,
                        //    subBlockDimensions.x,subBlockDimensions.y,subBlockDimensions.z);
                        return sum;
                    }
                }
            }
        }
    }
    return sum;
}
uint estimatePrimsInSubBlock(uvec3 subBlockGlobalOrigin, uvec3 subBlockDimensions) { /* Uses EFFECTIVE_MAX_MESHLET_PRIMS for early exit */
    uint sum = 0;
    for (uint z = 0; z < subBlockDimensions.z; ++z) {
        for (uint y = 0; y < subBlockDimensions.y; ++y) {
            for (uint x = 0; x < subBlockDimensions.x; ++x) {
                uvec3 cellGlobalOrigin = subBlockGlobalOrigin + uvec3(x, y, z);
                uint cubeCase = sampleMcCase(cellGlobalOrigin);
                if (cubeCase != 0 && cubeCase != 255) {
                    sum += countCellPrimitives(cubeCase);
                     if (sum > EFFECTIVE_MAX_MESHLET_PRIMS + WORKGROUP_SIZE) return sum;
                }
            }
        }
    }
    return sum;
}

// --- Task Shader Main ---
layout(local_size_x = WORKGROUP_SIZE) in;
void main() {
    if (gl_LocalInvocationIndex != 0u) {
        return;
    }

    uint compactedBlockID = gl_WorkGroupID.x;
    if (compactedBlockID >= activeBlockCountBuffer.count) {
        return;
    }
    uint originalBlockIndex = blockIds.compactedBlkArray[compactedBlockID];

    uvec3 fullBlockGlobalOrigin;
    fullBlockGlobalOrigin.z = originalBlockIndex / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = originalBlockIndex % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    fullBlockGlobalOrigin.y = sliceIndex / ubo.blockGridDim.x;
    fullBlockGlobalOrigin.x = sliceIndex % ubo.blockGridDim.x;
    fullBlockGlobalOrigin *= ubo.blockDim.xyz;

    uvec3 initialBlockDim = ubo.blockDim.xyz;

    // Use a simple array as a stack for pending blocks to process
    // struct SubBlock { uvec3 origin; uvec3 dim; };
    // SubBlock blockStack[8*8]; // Max 8 quarters from 8 halves = 64. Max 1 initial.
    // uint stackPtr = 0;
    // blockStack[stackPtr++] = SubBlock(fullBlockGlobalOrigin, initialBlockDim);
    // We'll simplify to avoid an explicit stack and just recurse one/two levels for clarity.

    // Process full block
    uint estVerts = estimateVertsInSubBlock(fullBlockGlobalOrigin, initialBlockDim);
    uint estPrims = estimatePrimsInSubBlock(fullBlockGlobalOrigin, initialBlockDim);

    if (estVerts == 0 && estPrims == 0) {
        return;
    }

    // if (estVerts <= EFFECTIVE_MAX_MESHLET_VERTS && estPrims <= EFFECTIVE_MAX_MESHLET_PRIMS) {
        uint cells = initialBlockDim.x * initialBlockDim.y * initialBlockDim.z;
        uint reservedVerts = cells * MAX_VERTS_PER_CELL;
        uint reservedIndices = cells * MAX_PRIMS_PER_CELL * 3u;
        uint vOff = atomicAdd(vCount.vertexCounter, reservedVerts);
        uint iOff = atomicAdd(iCount.indexCounter, reservedIndices);
        uint dOff = atomicAdd(meshletDescCount.meshletCounter, 1u);
        taskPayloadOut = TaskPayload(vOff, iOff, dOff, fullBlockGlobalOrigin, initialBlockDim, compactedBlockID);
        EmitMeshTasksEXT(1, 1, 1);

        return;
    // }

    // Subdivide full block into 8 half-blocks
    uvec3 baseHalfDim = max(uvec3(1), initialBlockDim / 2u);
    for (uint oz = 0u; oz < 2u; ++oz) {
        for (uint oy = 0u; oy < 2u; ++oy) {
            for (uint ox = 0u; ox < 2u; ++ox) {
                uvec3 currentSubBlockOffset = uvec3(ox * baseHalfDim.x, oy * baseHalfDim.y, oz * baseHalfDim.z);
                uvec3 subBlockOrigin = fullBlockGlobalOrigin + currentSubBlockOffset;
                uvec3 subBlockDim;
                subBlockDim.x = (ox == 0) ? baseHalfDim.x : initialBlockDim.x - baseHalfDim.x;
                subBlockDim.y = (oy == 0) ? baseHalfDim.y : initialBlockDim.y - baseHalfDim.y;
                subBlockDim.z = (oz == 0) ? baseHalfDim.z : initialBlockDim.z - baseHalfDim.z;
                subBlockDim = max(uvec3(1), subBlockDim);

                estVerts = estimateVertsInSubBlock(subBlockOrigin, subBlockDim);
                estPrims = estimatePrimsInSubBlock(subBlockOrigin, subBlockDim);

                if (estVerts == 0 && estPrims == 0) continue;

                if (estVerts <= EFFECTIVE_MAX_MESHLET_VERTS && estPrims <= EFFECTIVE_MAX_MESHLET_PRIMS) {
                    uint cells = subBlockDim.x * subBlockDim.y * subBlockDim.z;
                    uint reservedVerts = cells * MAX_VERTS_PER_CELL;
                    uint reservedIndices = cells * MAX_PRIMS_PER_CELL * 3u;
                    uint vOff = atomicAdd(vCount.vertexCounter, reservedVerts);
                    uint iOff = atomicAdd(iCount.indexCounter, reservedIndices);
                    uint dOff = atomicAdd(meshletDescCount.meshletCounter, 1u);
                    taskPayloadOut = TaskPayload(vOff, iOff, dOff, subBlockOrigin, subBlockDim, compactedBlockID);
                    // debugPrintfEXT("TS EMIT HALF");
                    EmitMeshTasksEXT(1, 1, 1);
                } else {
                    // Subdivide this half-block into 8 quarter-blocks
                    uvec3 baseQuarterDim = max(uvec3(1), subBlockDim / 2u);
                    bool emittedAnyQuarter = false;
                    for (uint qz = 0u; qz < 2u; ++qz) {
                        for (uint qy = 0u; qy < 2u; ++qy) {
                            for (uint qx = 0u; qx < 2u; ++qx) {
                                if ((qx == 1 && subBlockDim.x < 2) || (qy == 1 && subBlockDim.y < 2) || (qz == 1 && subBlockDim.z < 2)) {
                                    if ((qx == 1 && subBlockDim.x == 1) || (qy == 1 && subBlockDim.y == 1) || (qz == 1 && subBlockDim.z == 1)) continue;
                                }
                                uvec3 quarterSubBlockOffset = uvec3(qx * baseQuarterDim.x, qy * baseQuarterDim.y, qz * baseQuarterDim.z);
                                uvec3 quarterOrigin = subBlockOrigin + quarterSubBlockOffset;
                                uvec3 quarterDim;
                                quarterDim.x = (qx == 0) ? baseQuarterDim.x : subBlockDim.x - baseQuarterDim.x;
                                quarterDim.y = (qy == 0) ? baseQuarterDim.y : subBlockDim.y - baseQuarterDim.y;
                                quarterDim.z = (qz == 0) ? baseQuarterDim.z : subBlockDim.z - baseQuarterDim.z;
                                quarterDim = max(uvec3(1), quarterDim);

                                uint estVertsQ = estimateVertsInSubBlock(quarterOrigin, quarterDim);
                                uint estPrimsQ = estimatePrimsInSubBlock(quarterOrigin, quarterDim);

                                if (estVertsQ == 0 && estPrimsQ == 0) continue;

                                // For quarter blocks, we emit them even if they might slightly exceed EFFECTIVE limits,
                                // as this is the finest practical subdivision. The mesh shader will clamp.
                                // The main goal is that these quarter blocks are small enough for actual MAX_MESHLET_VERTS.
                                // If estVertsQ still exceeds MAX_MESHLET_VERTS_TARGET, then the scene is extremely dense.
                                // if (estVertsQ > MAX_MESHLET_VERTS_TARGET) {
                                //    debugPrintfEXT("TS WARN_DENSE_QUARTER: Task %u, Quarter EstV %u > Target %u\n", compactedBlockID, estVertsQ, MAX_MESHLET_VERTS_TARGET);
                                // }

                                uint cells = quarterDim.x * quarterDim.y * quarterDim.z;
                                uint reservedVerts = cells * MAX_VERTS_PER_CELL;
                                uint reservedIndices = cells * MAX_PRIMS_PER_CELL * 3u;
                                uint vOff = atomicAdd(vCount.vertexCounter, reservedVerts);
                                uint iOff = atomicAdd(iCount.indexCounter, reservedIndices);
                                uint dOff = atomicAdd(meshletDescCount.meshletCounter, 1u);
                                taskPayloadOut = TaskPayload(vOff, iOff, dOff, quarterOrigin, quarterDim, compactedBlockID);
                                EmitMeshTasksEXT(1, 1, 1);
                                emittedAnyQuarter = true;
                                debugPrintfEXT("TS EMIT QUARTER");
                            }
                        }
                    }
                    if (!emittedAnyQuarter && (estVerts > 0 || estPrims > 0)) { // Fallback if half-block was too small to quarter or all quarters were empty
                        uint cells = subBlockDim.x * subBlockDim.y * subBlockDim.z;
                        uint reservedVerts = cells * MAX_VERTS_PER_CELL;
                        uint reservedIndices = cells * MAX_PRIMS_PER_CELL * 3u;
                        uint vOff = atomicAdd(vCount.vertexCounter, reservedVerts);
                        uint iOff = atomicAdd(iCount.indexCounter, reservedIndices);
                        uint dOff = atomicAdd(meshletDescCount.meshletCounter, 1u);
                        taskPayloadOut = TaskPayload(vOff, iOff, dOff, subBlockOrigin, subBlockDim, compactedBlockID);
                        EmitMeshTasksEXT(1, 1, 1);
                    }
                }
            }
        }
    }
}