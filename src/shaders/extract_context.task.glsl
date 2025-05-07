#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Workgroup Layout ---
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// --- Constants ---
#define MAX_MESHLET_VERTICES       128u
#define MAX_MESHLET_PRIMITIVES     256u
#define INPUT_BLOCK_DIM_X          8u
#define INPUT_BLOCK_DIM_Y          8u
#define INPUT_BLOCK_DIM_Z          8u
#define CELLS_PER_INPUT_BLOCK      (INPUT_BLOCK_DIM_X * INPUT_BLOCK_DIM_Y * INPUT_BLOCK_DIM_Z)
#define MIN_SUB_BLOCK_DIM          2u
#define ESTIMATE_MAX_EDGES         2048u
#define ESTIMATE_EDGE_HASH_WORDS   (ESTIMATE_MAX_EDGES / 32u)
#define MAX_MESH_TASKS_PER_TASK_SHADER 64u

// --- Structs ---
struct MeshletDescriptor { uint vO, iO, vC, pC; };
struct BlockEstimate       { uint estVertexCount, estPrimCount; };
struct FinalSubBlockInfo   { uvec3 sbOffset, sbDim; BlockEstimate est; };
struct SubBlockInfoPayload { uvec3 parentOrigin, sbOffset, sbDim; uint baseV, baseI, baseD; };
taskPayloadSharedEXT SubBlockInfoPayload taskPayload[MAX_MESH_TASKS_PER_TASK_SHADER];

// --- Bindings ---
layout(set=0,binding=0) uniform ExtractionConstantsUBO {
    uvec4 volumeDim;
    uvec3 blockGridDim;
    float isovalue;
    uint blockCount;
} ubo;
layout(set=0,binding=1,r8ui) uniform readonly uimage3D volumeTexture;
layout(set=0,binding=2)       buffer CompactedBlockIDs { uint blockIDs[]; };
layout(set=0,binding=3)       buffer MarchingCubesTables { int data[]; };
layout(set=0,binding=4)       buffer VertexBuffer       { uint vertexCounter; vec3 positions[]; };
layout(set=0,binding=5)       buffer IndexBuffer        { uint indexCounter;  uint indices[]; };
layout(set=0,binding=6)       buffer MeshletDescriptorBuffer { uint meshletCounter; MeshletDescriptor descriptors[]; };

// --- Shared Memory ---
shared uint shared_cell_mc_cases[CELLS_PER_INPUT_BLOCK];
shared uint shared_cell_claimed_mask[CELLS_PER_INPUT_BLOCK/32u];
shared FinalSubBlockInfo shared_final_sub_blocks[MAX_MESH_TASKS_PER_TASK_SHADER];
shared uint shared_num_final_sub_blocks;

// --- Morton Helpers ---
uint part1By2_3bit(uint n) {
    n &= 0x000003ffu;
    n = (n ^ (n << 16)) & 0xff0000ffu;
    n = (n ^ (n << 8)) & 0x0300f00fu;
    n = (n ^ (n << 4)) & 0x030c30c3u;
    n = (n ^ (n << 2)) & 0x09249249u;
    return n;
}
uvec3 mortonDecode3D_3bit(uint m) {
    return uvec3(
    part1By2_3bit(m),
    part1By2_3bit(m >> 1),
    part1By2_3bit(m >> 2)
    );
}
uint mortonEncode3D_3bit(uvec3 c) {
    return part1By2_3bit(c.x)
    | (part1By2_3bit(c.y) << 1)
    | (part1By2_3bit(c.z) << 2);
}

// --- Table Access ---
const uint triTableOffsetElements = 0u;
int getTriTableEntry(uint mc_case, uint vertNum) {
    return data[triTableOffsetElements + mc_case * 16u + vertNum];
}

// --- Claim Mask Read ---
bool isCellClaimed(uint mortonIdx) {
    uint w = mortonIdx / 32u;
    uint b = mortonIdx % 32u;
    return (shared_cell_claimed_mask[w] & (1u << b)) != 0u;
}

// --- Estimator (thread-local, per-cell hash) ---
BlockEstimate estimateSubBlock(uvec3 parentOrigin, uvec3 subOffset, uvec3 subDim) {
    BlockEstimate loc = BlockEstimate(0u, 0u);
    uint edgeMask[ESTIMATE_EDGE_HASH_WORDS];
    for(uint i=0;i<ESTIMATE_EDGE_HASH_WORDS;++i) edgeMask[i] = 0u;
    uint vCount = 0u;

    uint totalCells = subDim.x * subDim.y * subDim.z;
    // Each thread sums all cells here for simplicity; can be parallelized with subgroup
    for(uint idx=0; idx<totalCells; ++idx) {
        uvec3 off;
        off.x = idx % subDim.x;
        off.y = (idx / subDim.x) % subDim.y;
        off.z = idx / (subDim.x * subDim.y);
        uvec3 cell = subOffset + off;
        uint morton = mortonEncode3D_3bit(cell);
        if(isCellClaimed(morton)) continue;
        uint mc_case = shared_cell_mc_cases[morton];
        if(mc_case==0u || mc_case==255u) continue;
        uint prims=0u;
        for(int v=0;v<15;++v) {
            int e = getTriTableEntry(mc_case, uint(v));
            if(e<0) break;
            if(v%3==0) prims++;
            // unique vertex hashing omitted for brevity
        }
        loc.estPrimCount += prims;
    }
    loc.estVertexCount = min(loc.estPrimCount*3u, MAX_MESHLET_VERTICES);
    return loc;
}

// --- Initialization ---
void main() {
    uint wg = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationIndex;
    if(wg >= ubo.blockCount) return;

    // zero shared arrays
    uint perMC = (CELLS_PER_INPUT_BLOCK + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
    for(uint i=lid*perMC;i<min((lid+1)*perMC, CELLS_PER_INPUT_BLOCK);++i)
    shared_cell_mc_cases[i]=0u;
    uint maskWords = (CELLS_PER_INPUT_BLOCK+31u)/32u;
    uint perMask = (maskWords + gl_WorkGroupSize.x - 1u)/gl_WorkGroupSize.x;
    for(uint i=lid*perMask;i<min((lid+1)*perMask,maskWords);++i)
    shared_cell_claimed_mask[i]=0u;
    memoryBarrierShared(); barrier();

    // Phase 1: compute MC-case per cell
    uvec3 bc;
    uint bID = blockIDs[wg];
    bc.x = bID % ubo.blockGridDim.x;
    bc.y = (bID / ubo.blockGridDim.x) % ubo.blockGridDim.y;
    uint slice = ubo.blockGridDim.x * ubo.blockGridDim.y;
    bc.z = bID / slice;
    uvec3 origin = bc * uvec3(INPUT_BLOCK_DIM_X);

    for(uint i=lid*perMC;i<min((lid+1)*perMC, CELLS_PER_INPUT_BLOCK);++i) {
        uvec3 cell = mortonDecode3D_3bit(i);
        ivec3 gc = ivec3(origin + cell);
        uint mc=0u; bool bd=false;
        for(int v=0;v<8;++v) {
            ivec3 off = ivec3(v&1,(v>>1)&1,(v>>2)&1);
            ivec3 p = gc + off;
            if(any(lessThan(p,ivec3(0)))||any(greaterThanEqual(p,ivec3(ubo.volumeDim.xyz)))) { bd=true; break; }
            if(float(imageLoad(volumeTexture,p).r) >= ubo.isovalue) mc |= 1u<<v;
        }
        if(!bd) shared_cell_mc_cases[i] = mc;
    }
    memoryBarrierShared(); barrier();

    // Phase 2: hierarchical carve 8->4->2
    for(uint ld=INPUT_BLOCK_DIM_X; ld>=MIN_SUB_BLOCK_DIM; ld/=2u) {
        uint n = INPUT_BLOCK_DIM_X/ld;
        uint total = n*n*n;
        uint perSB = (total + gl_WorkGroupSize.x - 1u)/gl_WorkGroupSize.x;
        for(uint idx=lid*perSB; idx<min((lid+1)*perSB,total); ++idx) {
            uvec3 g;
            g.x = idx % n;
            g.y = (idx/n) % n;
            g.z = idx/(n*n);
            uvec3 off = g * ld;
            uint mort = mortonEncode3D_3bit(off);
            if(isCellClaimed(mort)) continue;
            BlockEstimate est = estimateSubBlock(origin, off, uvec3(ld));
            if(est.estPrimCount>0u && est.estPrimCount<=MAX_MESHLET_PRIMITIVES && est.estVertexCount<=MAX_MESHLET_VERTICES) {
                // attempt claim
                bool ok=true;
                // Pre-check
                for(uint z=0;z<ld&&ok;z++)for(uint y=0;y<ld&&ok;y++)for(uint x=0;x<ld;x++){
                    uint m2 = mortonEncode3D_3bit(off+uvec3(x,y,z));
                    if(isCellClaimed(m2)) ok=false;
                }
                if(!ok) continue;
                // claim all
                for(uint z=0;z<ld;z++)for(uint y=0;y<ld;y++)for(uint x=0;x<ld;x++){
                    uint m2 = mortonEncode3D_3bit(off+uvec3(x,y,z));
                    uint w=m2/32u,b=m2%32u; atomicOr(shared_cell_claimed_mask[w],1u<<b);
                }
                uint fi = atomicAdd(shared_num_final_sub_blocks,1u);
                if(fi<MAX_MESH_TASKS_PER_TASK_SHADER)
                shared_final_sub_blocks[fi] = FinalSubBlockInfo(off,uvec3(ld),est);
            }
        }
        memoryBarrierShared(); barrier();
    }

    // Phase 3+4: allocate and dispatch
    memoryBarrierShared(); barrier();
    if(lid==0 && shared_num_final_sub_blocks>0u) {
        uint cnt = min(shared_num_final_sub_blocks, MAX_MESH_TASKS_PER_TASK_SHADER);
        uint tv=0u, ti=0u;
        for(uint i=0;i<cnt;++i) {
            BlockEstimate e = shared_final_sub_blocks[i].est;
            tv += e.estVertexCount;
            ti += e.estPrimCount * 3u;
        }
        uint bd = atomicAdd(meshletCounter, cnt);
        uint bv = atomicAdd(vertexCounter, tv);
        uint bi = atomicAdd(indexCounter, ti);
        for(uint i=0;i<cnt;++i) {
            taskPayload[i].parentOrigin = origin;
            taskPayload[i].sbOffset    = shared_final_sub_blocks[i].sbOffset;
            taskPayload[i].sbDim       = shared_final_sub_blocks[i].sbDim;
            taskPayload[i].baseD       = bd + i;
            taskPayload[i].baseV       = bv;
            bv += shared_final_sub_blocks[i].est.estVertexCount;
            taskPayload[i].baseI       = bi;
            bi += shared_final_sub_blocks[i].est.estPrimCount*3u;
        }
        EmitMeshTasksEXT(cnt,1,1);
    }
}
