#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_debug_printf : enable

// Block dimensions matching PMB paper
#define CORE_BLOCK_DIM_X 4
#define CORE_BLOCK_DIM_Y 4  
#define CORE_BLOCK_DIM_Z 4

// Mesh shader limits - following NVIDIA's recommendations
#define MAX_MESHLET_VERTS 64u   // NVIDIA recommended for optimal performance
#define MAX_MESHLET_PRIMS 126u  // NVIDIA recommended (unchanged)

// MC constants
#define MAX_TRI_INDICES 16
#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5

// Workgroup sizes
#define TASK_WORKGROUP_SIZE 32

// Structures shared with C++
struct VertexData {
    vec4 position;  // xyz = position, w = 1.0
    vec4 normal;    // xyz = normal, w = 0.0
};

struct TaskPayload {
    uvec3 blockOrigin;      // Global origin of the block in voxel coordinates
    uvec3 blockDim;         // Dimensions of the block to process
    uint originalBlockId;   // Original block index from compacted array
    uint level;             // Subdivision level (0 = full block, 1 = half, 2 = quarter)
};

// Binding points
#define BINDING_PUSH_CONSTANTS     0
#define BINDING_VOLUME_IMAGE       1
#define BINDING_ACTIVE_BLOCK_COUNT 2
#define BINDING_COMPACTED_BLOCKS   3
#define BINDING_MC_TRI_TABLE       4

taskPayloadSharedEXT TaskPayload taskPayloadOut;

// Bindings
layout(set = 0, binding = BINDING_PUSH_CONSTANTS, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = BINDING_VOLUME_IMAGE, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = BINDING_ACTIVE_BLOCK_COUNT, std430) readonly buffer ActiveBlockCount_SSBO { 
    uint count; 
} activeBlockCountBuffer;
layout(set = 0, binding = BINDING_COMPACTED_BLOCKS, std430) readonly buffer CompactedBlockIDs_SSBO { 
    uint compactedBlkArray[]; 
} blockIds;
layout(set = 0, binding = BINDING_MC_TRI_TABLE, std430) readonly buffer MarchingCubesTriTable_SSBO { 
    int triTable[]; 
} mc;

// Helper function to sample MC case
uint sampleMcCase(uvec3 cellOrigin) {
    uint cubeCase = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 offset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
        ivec3 coord = ivec3(cellOrigin) + offset;
        
        uint val = 0;
        if (all(greaterThanEqual(coord, ivec3(0))) && 
            all(lessThan(coord, ivec3(ubo.volumeDim.xyz)))) {
            val = imageLoad(volumeImage, coord).r;
        }
        
        if (float(val) >= ubo.isovalue) {
            cubeCase |= (1u << i);
        }
    }
    return cubeCase;
}

// Count vertices for a cell
uint countCellVertices(uint cubeCase) {
    uint vertexFlags = 0u;
    uint vertCount = 0u;
    int base = int(cubeCase) * MAX_TRI_INDICES;
    
    for (int i = 0; i < MAX_TRI_INDICES; ++i) {
        int edgeID = mc.triTable[base + i];
        if (edgeID < 0) break;
        
        uint mask = 1u << uint(edgeID);
        if ((vertexFlags & mask) == 0u) {
            vertexFlags |= mask;
            vertCount++;
        }
    }
    return vertCount;
}

// Count primitives for a cell
uint countCellPrimitives(uint cubeCase) {
    uint primCount = 0;
    int base = int(cubeCase) * MAX_TRI_INDICES;
    
    for (int i = 0; i < MAX_TRI_INDICES; i += 3) {
        if (mc.triTable[base + i] < 0) break;
        primCount++;
    }
    return primCount;
}

// Estimate vertices/primitives in a sub-block
void estimateSubBlock(uvec3 origin, uvec3 dimensions, out uint verts, out uint prims) {
    verts = 0;
    prims = 0;
    
    for (uint z = 0; z < dimensions.z && verts <= MAX_MESHLET_VERTS; ++z) {
        for (uint y = 0; y < dimensions.y && verts <= MAX_MESHLET_VERTS; ++y) {
            for (uint x = 0; x < dimensions.x && verts <= MAX_MESHLET_VERTS; ++x) {
                uvec3 cellOrigin = origin + uvec3(x, y, z);
                uint cubeCase = sampleMcCase(cellOrigin);
                
                verts += countCellVertices(cubeCase);
                prims += countCellPrimitives(cubeCase);
            }
        }
    }
}
layout(local_size_x = TASK_WORKGROUP_SIZE) in;
void main() {
    if (gl_LocalInvocationIndex != 0u) return;
    
    uint compactedBlockID = gl_WorkGroupID.x;
    if (compactedBlockID >= activeBlockCountBuffer.count) return;
    
    uint originalBlockIndex = blockIds.compactedBlkArray[compactedBlockID];
    
    // Calculate block origin from original index
    uvec3 blockCoord;
    blockCoord.z = originalBlockIndex / (ubo.blockGridDim.x * ubo.blockGridDim.y);
    uint sliceIndex = originalBlockIndex % (ubo.blockGridDim.x * ubo.blockGridDim.y);
    blockCoord.y = sliceIndex / ubo.blockGridDim.x;
    blockCoord.x = sliceIndex % ubo.blockGridDim.x;
    
    uvec3 blockOrigin = blockCoord * (ubo.blockDim.xyz - 1);
    uvec3 blockDim = ubo.blockDim.xyz;
    
    // Clamp block dimensions to volume bounds
    blockDim = min(blockDim, ubo.volumeDim.xyz - blockOrigin);
    
    // Estimate vertices and primitives for the full block
    uint estVerts, estPrims;
    estimateSubBlock(blockOrigin, blockDim, estVerts, estPrims);
    
    if (estVerts == 0 && estPrims == 0) return;
    
    // If block fits in one meshlet, dispatch it
    // Using NVIDIA's recommended limits: 64 vertices, 126 primitives
    // if (estVerts <= MAX_MESHLET_VERTS && estPrims <= MAX_MESHLET_PRIMS) {
        taskPayloadOut.blockOrigin = blockOrigin;
        taskPayloadOut.blockDim = blockDim;
        taskPayloadOut.originalBlockId = originalBlockIndex;
        taskPayloadOut.level = 0;
        EmitMeshTasksEXT(1, 1, 1);
        return;
    // }
    
    // Otherwise, subdivide the block
    uvec3 halfDim = max(uvec3(1), blockDim / 2u);
    
    for (uint oz = 0; oz < 2; ++oz) {
        for (uint oy = 0; oy < 2; ++oy) {
            for (uint ox = 0; ox < 2; ++ox) {
                uvec3 subOrigin = blockOrigin + uvec3(ox, oy, oz) * halfDim;
                uvec3 subDim = uvec3(
                    (ox == 0) ? halfDim.x : blockDim.x - halfDim.x,
                    (oy == 0) ? halfDim.y : blockDim.y - halfDim.y,
                    (oz == 0) ? halfDim.z : blockDim.z - halfDim.z
                );
                
                // Clamp to volume bounds
                subDim = min(subDim, ubo.volumeDim.xyz - subOrigin);
                if (any(equal(subDim, uvec3(0)))) continue;
                
                uint subVerts, subPrims;
                estimateSubBlock(subOrigin, subDim, subVerts, subPrims);
                
                if (subVerts > 0 || subPrims > 0) {
                    taskPayloadOut.blockOrigin = subOrigin;
                    taskPayloadOut.blockDim = subDim;
                    taskPayloadOut.originalBlockId = originalBlockIndex;
                    taskPayloadOut.level = 1;
                    EmitMeshTasksEXT(1, 1, 1);
                }
            }
        }
    }
}