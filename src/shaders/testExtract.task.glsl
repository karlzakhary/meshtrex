#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require // Or int32 if offsets fit
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require // For subgroupBroadcastFirst
#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
// Dimensions of the blocks processed by ONE Task/Mesh invocation
// This should match the block size used in your filtering stage
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 8 // Example: Adjust to your actual block size
#define CELLS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)

// Maximum possible output per cell in classic MC
#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5 // Max triangles per MC case

// Workgroup size for this Task Shader
#define WORKGROUP_SIZE 32

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

struct TaskPayload {
    uint globalVertexOffset;     // Offset reserved in global Vertex Buffer
    uint globalIndexOffset;      // Offset reserved in global Index Buffer
    uint globalMeshletDescOffset;// Offset for this block's descriptor
    uvec3 blockOrigin;           // Absolute origin in volume coordinates
    uint taskId;
    // blockDimensions are implicit (BLOCK_DIM_X/Y/Z)
};

// Payload sent to Mesh Shader
taskPayloadSharedEXT TaskPayload taskPayloadOut;
// --- End Structures ---

// Binding 0: UBO (Assuming layout matches C++ PushConstants struct)
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    // Use PushConstants name if that's the struct in C++ UBO
    uvec4 volumeDim;
    uvec4 blockDim; // BlockDim from original PushConstants might be unused here
    uvec4 blockGridDim; // Grid dim of initial blocks
    float isovalue;
    // Add other members if they exist in C++ PushConstants
} ubo;

// Binding 1: Volume Image - Not needed in this simplified Task Shader

// Binding 2: Compacted Block IDs (Input)
layout(set = 0, binding = 3, std430) readonly buffer CompactedBlockIDs_SSBO { uint compactedBlkArray[]; } blockIds;

// Binding 3: Active Block Count Buffer - Not needed in this simplified Task Shader

// Binding 4: MC Triangle Table - Not needed in this Task Shader

// Binding 5: MC NumVertices Table - Not needed in this Task Shader

// Binding 6: Vertex Buffer Data - Not directly accessed by Task Shader

// Binding 7: Vertex Count Buffer (Output Counter)
layout(set = 0, binding = 7, std430) buffer VertexCount_SSBO { uint vertexCounter; } vCount;

// Binding 8: Index Buffer Data - Not directly accessed by Task Shader

// Binding 9: Index Count Buffer (Output Counter)
layout(set = 0, binding = 9, std430) buffer IndexCount_SSBO { uint indexCounter; } iCount;

// Binding 10: Meshlet Descriptor Buffer Data - Not directly accessed by Task Shader

// Binding 11: Meshlet Descriptor Count Buffer (Output Counter)
layout(set = 0, binding = 11, std430) buffer MeshletDescCount_SSBO { uint meshletCounter; } meshletCount;
// --- End Descriptor Set Bindings ---

// --- Task Shader Main ---
layout(local_size_x = WORKGROUP_SIZE) in;
void main() {
    uint compactedBlockID = gl_WorkGroupID.x;
    // TODO: Add bounds check for compactedBlockID using Active Block Count buffer (Binding 3) if needed

    // Get original block index from Binding 2
    uint originalBlockIndex = blockIds.compactedBlkArray[compactedBlockID];

    // Calculate block origin
    uvec3 blockOrigin;
    blockOrigin.z = originalBlockIndex / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = originalBlockIndex % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    blockOrigin.y = sliceIndex / ubo.blockGridDim.x;
    blockOrigin.x = sliceIndex % ubo.blockGridDim.x;
    blockOrigin *= ubo.blockDim.xyz;

    // Reserve max space
    uint maxPossibleVerts = CELLS_PER_BLOCK * MAX_VERTS_PER_CELL;
    uint maxPossibleIndices = CELLS_PER_BLOCK * MAX_PRIMS_PER_CELL * 3;

    uint reservedGlobalVtxOffset = 0;
    uint reservedGlobalIdxOffset = 0;
    uint reservedGlobalDescOffset = 0;

    if (gl_LocalInvocationIndex == 0) {
        // Access counters using the correct binding points
        reservedGlobalVtxOffset = atomicAdd(vCount.vertexCounter, maxPossibleVerts);     // Binding 7 counter
        reservedGlobalIdxOffset = atomicAdd(iCount.indexCounter, maxPossibleIndices);       // Binding 9 counter
        reservedGlobalDescOffset = atomicAdd(meshletCount.meshletCounter, 1u);            // Binding 11 counter

//        if (gl_LocalInvocationIndex == 0 && blockOrigin.x != 0 && blockOrigin.y != 0) {
//            debugPrintfEXT(
//                "  Task[%u] Values Before Payload Assign: VtxOff=%u, IdxOff=%u, DescOff=%u, BlockOrigin=(%u,%u,%u) from ubo.blockDim=(%u,%u,%u)\n",
//                gl_WorkGroupID.x,
//                reservedGlobalVtxOffset,
//                reservedGlobalIdxOffset,
//                reservedGlobalDescOffset,
//                blockOrigin.x, blockOrigin.y, blockOrigin.z,
//                ubo.blockDim.x, ubo.blockDim.y, ubo.blockDim.z
//            );
//        }
    }

    // Broadcast offsets
    reservedGlobalVtxOffset = subgroupBroadcastFirst(reservedGlobalVtxOffset);
    reservedGlobalIdxOffset = subgroupBroadcastFirst(reservedGlobalIdxOffset);
    reservedGlobalDescOffset = subgroupBroadcastFirst(reservedGlobalDescOffset);

//    if (gl_LocalInvocationIndex == 0) {
//        debugPrintfEXT("  Task[%u] Values Before Payload Assign: VtxOff=%u, IdxOff=%u, DescOff=%u\n",
//                       gl_WorkGroupID.x,
//                       reservedGlobalVtxOffset,
//                       reservedGlobalIdxOffset,
//                       reservedGlobalDescOffset);
//    }


    // Populate Payload
    taskPayloadOut.globalVertexOffset = reservedGlobalVtxOffset;
    taskPayloadOut.globalIndexOffset = reservedGlobalIdxOffset;
    taskPayloadOut.globalMeshletDescOffset = reservedGlobalDescOffset;
    taskPayloadOut.blockOrigin = blockOrigin;
    taskPayloadOut.taskId = gl_WorkGroupID.x;

//    if (gl_LocalInvocationIndex == 0) {
//        for(int i =0 ; i < 1000 ; i++) {
//            if (blockIds.compactedBlkArray[i] != 0) {
//                debugPrintfEXT("active blockid: %u", blockIds.compactedBlkArray[i]);
//            }
//
//        }
//        debugPrintfEXT(
//            "TS[%u] → blockIds[%u]=%u → blockOrigin=(%u,%u,%u)\n",
//            gl_WorkGroupID.x,
//            gl_WorkGroupID.x,
//            originalBlockIndex,
//            blockOrigin.x, blockOrigin.y, blockOrigin.z
//        );
//    }

    // Emit Task
    EmitMeshTasksEXT(1, 1, 1);
}