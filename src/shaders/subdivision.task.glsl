#version 460 core
#extension GL_EXT_mesh_shader               : require
#extension GL_EXT_shader_atomic_int64       : require
#extension GL_EXT_scalar_block_layout      : enable
#extension GL_KHR_shader_subgroup_ballot    : require
#extension GL_EXT_debug_printf              : enable

// --- Configurable Parameters ---
#define BLOCK_DIM_X      8
#define BLOCK_DIM_Y      8
#define BLOCK_DIM_Z      8
#define CELLS_PER_BLOCK  (BLOCK_DIM_X*BLOCK_DIM_Y*BLOCK_DIM_Z)
#define MAX_MESHLET_VERTS 256u

// --- Structures ---
struct TaskPayload {
    uint    globalVertexOffset;
    uint    globalIndexOffset;
    uint    globalMeshletDescOffset;
    uvec3   blockOrigin;
    uvec3   subBlockDim;
    uint    taskId;
};
taskPayloadSharedEXT TaskPayload taskPayloadOut;
// --- End Structures ---

// --- UBO + SSBO Bindings ---
layout(set=0, binding=0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set=0, binding=1, r8ui) uniform readonly uimage3D    volumeImage;
layout(set=0, binding=3, std430) readonly buffer CompactedBlockIDs { uint compactedBlkArray[]; } blockIds;
layout(set=0, binding=5, std430) readonly buffer NumVertsTable   { uint numVertsTable[]; } nvt;

// output counters
layout(set=0, binding=7, std430) buffer VCountSSBO { uint vCount; } vCount;
layout(set=0, binding=9, std430) buffer ICountSSBO { uint iCount; } iCount;
layout(set=0, binding=11,std430) buffer DCountSSBO { uint dCount; } dCount;

// helper: estimate # verts for an arbitrary sub‐block
uint estimateVerts(uvec3 origin, uvec3 dim) {
    uint sum = 0;
    // only thread 0 does the work
    if (gl_LocalInvocationIndex == 0) {
        for (uint z = 0; z < dim.z; ++z)
        for (uint y = 0; y < dim.y; ++y)
        for (uint x = 0; x < dim.x; ++x) {
            uvec3 cell = origin + uvec3(x,y,z);
            // sample MC case
            uint cubeCase = 0;
            for (int i=0; i<8; ++i) {
                ivec3 off = ivec3(i&1, (i&2)>>1, (i&4)>>2);
                ivec3 c = ivec3(cell) + off;
                uint val = 0;
                if (all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, ivec3(ubo.volumeDim.xyz))))
                val = imageLoad(volumeImage, c).r;
                if (float(val) >= ubo.isovalue) cubeCase |= (1<<i);
            }
            sum += nvt.numVertsTable[cubeCase];
            if (sum > MAX_MESHLET_VERTS) {
                break;
            }
        }
    }
    // broadcast to whole subgroup
    return subgroupBroadcastFirst(sum);
}

layout(local_size_x=32) in;
void main() {
    uint bid = gl_WorkGroupID.x;
    uint originalBlockIndex = blockIds.compactedBlkArray[bid];

    // compute absolute origin of the 8³ block
    uvec3 grid = ubo.blockGridDim.xyz;
    uint zslice = originalBlockIndex / (grid.x * grid.y);
    uint slice  = originalBlockIndex % (grid.x * grid.y);
    uvec3 origin = uvec3(slice % grid.x, slice / grid.x, zslice) * ubo.blockDim.xyz;

    // first try the full 8×8×8
    uint est = estimateVerts(origin, ubo.blockDim.xyz);
    uvec3 dim = ubo.blockDim.xyz;
    uint subCells     = dim.x * dim.y * dim.z;
    uint maxVerts     = subCells * 12;    // = #cells ×12
    uint maxIndices   = subCells * 5 * 3u;
    if (est <= MAX_MESHLET_VERTS) {
        // reserve space & emit one meshlet
        if (gl_LocalInvocationIndex == 0) {
            uint vOff = atomicAdd(vCount.vCount, maxVerts);
            uint iOff = atomicAdd(iCount.iCount, maxIndices);
            uint dOff = atomicAdd(dCount.dCount, 1u);
            taskPayloadOut.globalVertexOffset    = vOff;
            taskPayloadOut.globalIndexOffset     = iOff;
            taskPayloadOut.globalMeshletDescOffset = dOff;
            taskPayloadOut.blockOrigin           = origin;
            taskPayloadOut.subBlockDim           = ubo.blockDim.xyz;
            taskPayloadOut.taskId                = bid;
            EmitMeshTasksEXT(1,1,1);
            return;
        }
    }

    // else subdivide into eight 4×4×4
    uvec3 halfDim = ubo.blockDim.xyz / 2u;
    for (uint oz=0; oz<2; ++oz) {
        for (uint oy=0; oy<2; ++oy) {
            for (uint ox=0; ox<2; ++ox) {
                uvec3 subOrigin = origin + uvec3(ox,oy,oz)* halfDim;
                uint subCells2     = halfDim.x * halfDim.y * halfDim.z;
                uint maxVerts2     = subCells2 * 12;    // = #cells ×12
                uint maxIndices2   = subCells2 * 5 * 3u;
                uint est2 = estimateVerts(subOrigin, halfDim);
                if (est2 <= MAX_MESHLET_VERTS) {
                    if (gl_LocalInvocationIndex == 0) {
                        uint vOff = atomicAdd(vCount.vCount, maxVerts2);
                        uint iOff = atomicAdd(iCount.iCount, maxIndices2);
                        uint dOff = atomicAdd(dCount.dCount, 1u);
                        taskPayloadOut.globalVertexOffset     = vOff;
                        taskPayloadOut.globalIndexOffset      = iOff;
                        taskPayloadOut.globalMeshletDescOffset = dOff;
                        taskPayloadOut.blockOrigin            = subOrigin;
                        taskPayloadOut.subBlockDim            = halfDim;
                        taskPayloadOut.taskId                 = bid;
                        EmitMeshTasksEXT(1,1,1);
                    }
                } else {
                    if (gl_LocalInvocationIndex == 0) {
                        debugPrintfEXT("OVERFLOW subblock at (%u,%u,%u) size=(%u,%u,%u)\n",
                                       subOrigin.x, subOrigin.y, subOrigin.z,
                                       halfDim.x, halfDim.y, halfDim.z);

                        uint vOff = atomicAdd(vCount.vCount, maxVerts2);
                        uint iOff = atomicAdd(iCount.iCount, maxIndices2);
                        uint dOff = atomicAdd(dCount.dCount, 1u);
                        taskPayloadOut.globalVertexOffset     = vOff;
                        taskPayloadOut.globalIndexOffset      = iOff;
                        taskPayloadOut.globalMeshletDescOffset = dOff;
                        taskPayloadOut.blockOrigin            = subOrigin;
                        taskPayloadOut.subBlockDim            = halfDim;
                        taskPayloadOut.taskId                 = bid;
                        EmitMeshTasksEXT(1,1,1);
                    }
                }
            }
        }
    }
}