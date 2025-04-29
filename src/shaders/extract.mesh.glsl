#version 460 core
#extension GL_EXT_mesh_shader : require // Enable Mesh Shader extension
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_atomic_int64 : enable // If needed
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable


// --- Meshlet Descriptor Structure (matches C++) ---
struct MeshletDescriptor {
    uint vertexOffset;    // Offset into the global vertexBuffer
    uint indexOffset;     // Offset into the global indexBuffer
    uint vertexCount;     // Number of vertices IN THIS MESHLET
    uint primitiveCount;  // Number of triangles IN THIS MESHLET
// Optional Bounds etc.
};

// --- Workgroup Layout ---
// One mesh workgroup per sub-block dispatched by the Task shader.
// Use a fixed size, e.g., 32 threads. Must match C++ pipeline setup if specified there.
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// --- Output Vertex/Primitive Counts ---
// These MUST be <= device limits (e.g., 256 verts, 256 prims for basic mesh shader)
// We aim for the paper's target values.
layout(max_vertices = 128) out;
layout(max_primitives = 256) out; // Adjust based on C++ limits if different
layout(triangles) out; // Output triangles

// --- Constants ---
#define MAX_VERTS_PER_SUBBLOCK 128 // Should match max_vertices
#define MAX_PRIMS_PER_SUBBLOCK 256 // Should match max_primitives
#define SUBGROUP_SIZE 32
#define UNIQUE_VERTEX_HASH_SIZE 1024 // Size of shared memory hash table for vertex reuse

// --- Descriptor Set Bindings (Must match C++ and Task Shader) ---
layout(set = 0, binding = 0, scalar) uniform ExtractionConstantsUBO {
    uvec4 volumeDim;
    uvec4 blockGridDim;
    float isovalue;
// Add other constants if needed
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeTexture;

layout(set = 0, binding = 3, scalar) buffer MarchingCubesTable { int triTable[]; }; // MC triTable[256][16]

// Output Buffers (Write Access)
// Matches Task Shader bindings
layout(set = 0, binding = 4, scalar) buffer VertexBuffer { uint vertexCounter; vec3 positions[]; /* vec3 normals[]; */ }; // Output Vertices (Example layout)
layout(set = 0, binding = 5, scalar) buffer IndexBuffer { uint indexCounter; uint indices[]; };          // Output Indices (local to meshlet)
layout(set = 0, binding = 6, scalar) buffer MeshletDescriptorBuffer { uint meshletCounter; MeshletDescriptor descriptors[]; }; // Output Descriptors

// --- Input Payload from Task Shader ---
struct SubBlockInfo {
    uvec3 blockOrigin;
    uvec3 subBlockOffset;
    uvec3 subBlockDim;
    uint baseVertexOffset;
    uint baseIndexOffset;
    uint baseDescriptorOffset;
    uint activeCellCount; // Note: Might be 0 if task shader couldn't calculate precisely
// uint activeCellIndices[...]; // Optional list
};
taskPayloadSharedEXT SubBlockInfo taskPayload;
// --- Output Vertex Structure (matches buffer layout and Primitive Assembly) ---
struct OutputVertex {
    vec3 position;
    vec3 normal;
};
layout(location = 0) out OutputVertex meshVertices[]; // Corresponds to gl_MeshVerticesEXT

// --- Shared Memory ---
// For PMB-style unique vertex generation within the sub-block
shared uint uniqueVertexIndices[UNIQUE_VERTEX_HASH_SIZE]; // Hash table for vertex indices on edges
shared uint numVerticesWritten; // Atomic counter for local vertices
shared uint numPrimitivesWritten; // Atomic counter for local primitives

// --- Helper Functions ---
// Function to calculate vertex position on an edge
vec3 calculateVertexPos(ivec3 p1_coord, ivec3 p2_coord, float val1, float val2) {
    // Linear interpolation
    float t = (ubo.isovalue / 255.0f - val1) / (val2 - val1);
    return mix(vec3(p1_coord), vec3(p2_coord), t);
}

// Function to calculate vertex normal (using central differences on volume)
vec3 calculateVertexNormal(vec3 pos) {
    // Sample volume texture around pos to calculate gradient
    float dx = (imageLoad(volumeTexture, ivec3(pos + vec3(1, 0, 0))).r - imageLoad(volumeTexture, ivec3(pos - vec3(1, 0, 0))).r) / 255.0f;
    float dy = (imageLoad(volumeTexture, ivec3(pos + vec3(0, 1, 0))).r - imageLoad(volumeTexture, ivec3(pos - vec3(0, 1, 0))).r) / 255.0f;
    float dz = (imageLoad(volumeTexture, ivec3(pos + vec3(0, 0, 1))).r - imageLoad(volumeTexture, ivec3(pos - vec3(0, 0, 1))).r) / 255.0f;
    return normalize(vec3(dx, dy, dz));
}

// Hash function for edge ID within the sub-block (Needs careful definition)
// Example: Hash based on the lower coordinate and the edge index (0-2 for x,y,z)
uint hashEdge(ivec3 cellCoordLocal, uint edgeIndex) {
    // A more robust hash function might be needed depending on block size
    uint h = cellCoordLocal.x + cellCoordLocal.y * taskPayload.subBlockDim.x + cellCoordLocal.z * taskPayload.subBlockDim.x * taskPayload.subBlockDim.y;
    h = (h << 3) | (edgeIndex & 7); // Combine with edge index
    return h % UNIQUE_VERTEX_HASH_SIZE;
}

// Function to get/create unique vertex on an edge (PMB style)
// Returns the LOCAL index of the vertex for this meshlet.
uint getOrCreateVertex(ivec3 localCellCoord, uint edgeIdx, float cornerValues[8]) {
    // 1. Determine edge endpoints (p1, p2) based on localCellCoord and edgeIdx
    ivec3 p1_offset = ivec3(0);
    ivec3 p2_offset = ivec3(0);
    int corner1 = 0;
    int corner2 = 0;
    // Based on standard MC edge indexing:
    // 0: 0-1 (x)  1: 1-2 (y)  2: 2-3 (x)  3: 3-0 (y)
    // 4: 4-5 (x)  5: 5-6 (y)  6: 6-7 (x)  7: 7-4 (y)
    // 8: 0-4 (z)  9: 1-5 (z) 10: 2-6 (z) 11: 3-7 (z)
    // TODO: Define this mapping robustly
    // Example for edge 0 (between corner 0 and 1):
    if (edgeIdx == 0) { p1_offset = ivec3(0,0,0); p2_offset = ivec3(1,0,0); corner1=0; corner2=1;}
    // ... implement for all 12 edges ...
    else { return ~0u; } // Invalid edge index

    ivec3 p1_coord_global = ivec3(taskPayload.blockOrigin + taskPayload.subBlockOffset) + localCellCoord + p1_offset;
    ivec3 p2_coord_global = ivec3(taskPayload.blockOrigin + taskPayload.subBlockOffset) + localCellCoord + p2_offset;


    // 2. Hash the edge (use the edge's canonical representation, e.g., based on lower coord)
    // TODO: Define canonical edge representation and hashing
    uint edgeHash = hashEdge(localCellCoord, edgeIdx); // Simplistic hash

    // 3. Check shared memory hash table (atomic compare and swap)
    uint invalidIndex = ~0u;
    uint existingLocalIndex = atomicCompSwap(uniqueVertexIndices[edgeHash], invalidIndex, MAX_VERTS_PER_SUBBLOCK + 1); // Use temp value > max

    if (existingLocalIndex == invalidIndex) {
        // This thread is the first to claim this edge vertex
        // Calculate vertex position and normal
        vec3 pos = calculateVertexPos(p1_coord_global, p2_coord_global, cornerValues[corner1], cornerValues[corner2]);
        vec3 norm = calculateVertexNormal(pos); // Simplified normal calc

        // Atomically allocate a LOCAL index for this new vertex
        uint newLocalIndex = atomicAdd(numVerticesWritten, 1);

        if (newLocalIndex < MAX_VERTS_PER_SUBBLOCK) {
            // Write vertex data to output vertex array for the rasterizer
            meshVertices[newLocalIndex].position = pos;
            meshVertices[newLocalIndex].normal = norm;

            // Write vertex data to the global buffer
            uint globalVertexIndex = taskPayload.baseVertexOffset + newLocalIndex;
            // TODO: Write pos and norm correctly packed into the buffer
            positions[globalVertexIndex] = pos;
            // normals[globalVertexIndex] = norm; // Assuming separate normal array or interleaved

            // Store the *local* index back into the shared memory hash table
            atomicExchange(uniqueVertexIndices[edgeHash], newLocalIndex);
            return newLocalIndex;
        } else {
            // Exceeded vertex limit for this meshlet! Handle error/clamp.
            atomicCompSwap(uniqueVertexIndices[edgeHash], MAX_VERTS_PER_SUBBLOCK + 1, invalidIndex); // Revert claim
            return invalidIndex; // Indicate failure
        }
    } else if (existingLocalIndex > MAX_VERTS_PER_SUBBLOCK) {
        // Another thread is currently creating it, spin wait briefly (or use more robust sync)
        // Spin waiting is generally bad, consider alternative sync if this happens often.
        // For simplicity, just return invalid for now.
        memoryBarrierShared(); // Ensure visibility
        return uniqueVertexIndices[edgeHash]; // Try reading again after barrier
        // return invalidIndex;
    } else {
        // Vertex already created by another thread, return its local index
        return existingLocalIndex;
    }
}

int getMCTriangleVertices(uint caseIndex, uint vertexNum) {
    // Access packed triTable buffer
    return triTable[caseIndex * 16 + vertexNum];
}

void main() {
    uint globalThreadID = gl_GlobalInvocationID.x; // Unique ID across the dispatch
    uint localID = gl_LocalInvocationIndex;
    uint workgroupID = gl_WorkGroupID.x; // Index corresponding to the launched mesh task

    // --- Initialization ---
    if (localID == 0) {
        numVerticesWritten = 0;
        numPrimitivesWritten = 0;
    }
    // Initialize shared vertex hash table
    for (uint i = localID; i < UNIQUE_VERTEX_HASH_SIZE; i += gl_WorkGroupSize.x) {
        uniqueVertexIndices[i] = ~0u; // Initialize with invalid index
    }
    barrier(); // Ensure initialization complete


    // --- Determine Cells to Process ---
    // This mesh task handles one sub-block defined in the payload.
    // Divide the cells *within the sub-block* among threads.
    uvec3 subBlockDim = taskPayload.subBlockDim;
    uint totalCellsInSubBlock = subBlockDim.x * subBlockDim.y * subBlockDim.z;

    // NOTE: This assumes the task shader passed the *total* active cell count for the *original* block,
    // not the count specific to this sub-block. A precise count requires more complex task shader logic.
    // We proceed by iterating all cells in the sub-block and checking them again.

    uint cellsPerThread = (totalCellsInSubBlock + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    uint startCell = localID * cellsPerThread;
    uint endCell = min(startCell + cellsPerThread, totalCellsInSubBlock);

    // Local storage for generated triangle indices for this thread
    uint threadLocalIndices[15]; // Max 5 triangles * 3 indices
    uint threadPrimitiveCount = 0;

    for (uint cellIdx1D_local = startCell; cellIdx1D_local < endCell; ++cellIdx1D_local) {
        // Convert 1D cell index (within sub-block) to 3D local coordinates
        ivec3 localCellCoord;
        localCellCoord.x = int(cellIdx1D_local % subBlockDim.x);
        localCellCoord.y = int((cellIdx1D_local / subBlockDim.x) % subBlockDim.y);
        localCellCoord.z = int(cellIdx1D_local / (subBlockDim.x * subBlockDim.y));

        // Calculate global coordinate for sampling volume
        ivec3 globalCellCoord = ivec3(taskPayload.blockOrigin + taskPayload.subBlockOffset) + localCellCoord;

        // Calculate Marching Cubes case
        uint mc_case = 0;
        float cornerValues[8]; // Store corner values for vertex interpolation
        bool onBoundary = false; // Check if cell touches volume boundary
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerCoord = globalCellCoord + cornerOffset;

            // Clamp coordinates
            if (any(lessThan(cornerCoord, ivec3(0))) || any(greaterThanEqual(cornerCoord, ivec3(ubo.volumeDim.xyz)))) {
                onBoundary = true;
                break; // Skip cells partially outside volume
            }
            // Sample volume
            float val = imageLoad(volumeTexture, cornerCoord).r / 255.0f;
            cornerValues[i] = val;

            if (val >= ubo.isovalue / 255.0f) {
                mc_case |= (1 << i);
            }
        }

        if (onBoundary || mc_case == 0 || mc_case == 255) {
            continue; // Skip non-active or boundary cells
        }

        // Generate triangles for this case using triTable
        for (int v = 0; v < 15; v += 3) {
            int edgeIndex0 = getMCTriangleVertices(mc_case, v);
            if (edgeIndex0 == -1) break; // End of triangle list for this case
            int edgeIndex1 = getMCTriangleVertices(mc_case, v + 1);
            int edgeIndex2 = getMCTriangleVertices(mc_case, v + 2);

            // Get/create unique LOCAL vertex indices for the edges
            uint localVertIndex0 = getOrCreateVertex(localCellCoord, uint(edgeIndex0), cornerValues);
            uint localVertIndex1 = getOrCreateVertex(localCellCoord, uint(edgeIndex1), cornerValues);
            uint localVertIndex2 = getOrCreateVertex(localCellCoord, uint(edgeIndex2), cornerValues);

            // Check if vertex creation was successful and within limits
            if (localVertIndex0 != ~0u && localVertIndex1 != ~0u && localVertIndex2 != ~0u &&
            threadPrimitiveCount < MAX_PRIMS_PER_SUBBLOCK)
            {
                // Store the LOCAL vertex indices for this triangle
                threadLocalIndices[threadPrimitiveCount * 3 + 0] = localVertIndex0;
                threadLocalIndices[threadPrimitiveCount * 3 + 1] = localVertIndex1;
                threadLocalIndices[threadPrimitiveCount * 3 + 2] = localVertIndex2;
                threadPrimitiveCount++;
            } else {
                // Handle vertex/primitive limit exceeded for this thread/meshlet
                // Break or log error. For now, just stop adding primitives.
                break;
            }
        }
    } // End loop over cells for this thread

    // --- Aggregate Primitives and Write Output ---
    barrier(); // Ensure all threads have finished vertex/primitive generation

    // Atomically allocate space in the global index buffer for this thread's primitives
    uint localPrimitiveOffset = atomicAdd(numPrimitivesWritten, threadPrimitiveCount);

    // Check if total primitives exceed meshlet limit
    if (localPrimitiveOffset + threadPrimitiveCount <= MAX_PRIMS_PER_SUBBLOCK) {
        // Write this thread's triangle indices to the global index buffer
        uint globalIndexOffset = taskPayload.baseIndexOffset + localPrimitiveOffset * 3;
        for (uint i = 0; i < threadPrimitiveCount * 3; ++i) {
            // Check buffer bounds if precise allocation wasn't possible
            // if (globalIndexOffset + i < MAX_INDICES_IN_BUFFER)
            indices[globalIndexOffset + i] = threadLocalIndices[i];
        }
    } else {
        // Handle primitive limit overflow for the whole meshlet
        // This thread's primitives cannot be added. Might need rollback logic
        // for vertices if vertices were added speculatively.
        // For simplicity, we just don't write the indices.
    }

    barrier(); // Ensure all threads have written indices and updated counters

    // --- Finalize (Thread 0 / Elected Leader) ---
    if (subgroupElect()) {
        uint finalVertexCount = min(numVerticesWritten, MAX_VERTS_PER_SUBBLOCK);
        uint finalPrimitiveCount = min(numPrimitivesWritten, MAX_PRIMS_PER_SUBBLOCK);

        // Set the actual output counts for the mesh shader pipeline
        SetMeshOutputsEXT(finalVertexCount, finalPrimitiveCount);

        // Write the meshlet descriptor
        if (finalPrimitiveCount > 0 && finalVertexCount > 0) {
            uint descriptorIndex = taskPayload.baseDescriptorOffset;
            // Write descriptor data
            // Check buffer bounds
            // if (descriptorIndex < MAX_DESCRIPTORS_IN_BUFFER)
            descriptors[descriptorIndex].vertexOffset = taskPayload.baseVertexOffset;
            descriptors[descriptorIndex].indexOffset = taskPayload.baseIndexOffset;
            descriptors[descriptorIndex].vertexCount = finalVertexCount;
            descriptors[descriptorIndex].primitiveCount = finalPrimitiveCount;
        }
    }

    // Primitive Assembly: Output the local indices for the generated primitives
    // This uses the local indices generated by the threads.
    // The rasterizer will use gl_VertexCount and gl_PrimitiveCount set by SetMeshOutputsEXT.
    // Output up to `finalPrimitiveCount` primitives.
    // This part needs careful mapping from the distributed threadLocalIndices to the linear gl_PrimitiveIndicesEXT.
    // Simplification: Assume `numPrimitivesWritten` is the final count and output indices based on that.
    // A more robust way might involve another shared memory array.

    // This output step is primarily for the fixed-function primitive assembly,
    // which we are discarding anyway since rasterizerDiscardEnable = true.
    // So, the content here might not strictly matter as long as SetMeshOutputsEXT is called.
    // However, validation layers might require valid indices within the declared vertex count.
    // For now, we can potentially skip writing gl_PrimitiveIndicesEXT if discard is enabled.

    // Example (if needed for validation, might be complex to implement correctly):
    /*
    uint totalPrims = numPrimitivesWritten; // Use the final count
    uint primsPerThread = (totalPrims + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    uint startPrim = localID * primsPerThread;
    uint endPrim = min(startPrim + primsPerThread, totalPrims);

    for(uint p = startPrim; p < endPrim; ++p) {
        // Need a way to map 'p' back to the correct thread's indices or a shared index array
        // This part is non-trivial with the distributed index generation.
        // gl_PrimitiveIndicesEXT[p * 3 + 0] = ...
        // gl_PrimitiveIndicesEXT[p * 3 + 1] = ...
        // gl_PrimitiveIndicesEXT[p * 3 + 2] = ...
    }
    */

}