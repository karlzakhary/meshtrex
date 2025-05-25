#version 460 core
#extension GL_EXT_mesh_shader               : require
#extension GL_EXT_shader_atomic_int64       : require
#extension GL_EXT_scalar_block_layout      : enable
#extension GL_EXT_debug_printf              : enable

#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5

// --- Structures ---
struct TaskPayload {
    uint    globalVertexOffset;
    uint    globalIndexOffset;
    uint    globalMeshletDescOffset;
    uvec3   blockOrigin;
    uvec3   subBlockDim;
    uint    taskId;
};
taskPayloadSharedEXT TaskPayload taskPayloadIn;

struct VertexData {
    vec4 position;
    vec4 normal;
};
struct MeshletDesc {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

// --- UBO + SSBO Bindings ---
layout(set=0, binding=0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;
layout(set=0, binding=1, r8ui) uniform readonly uimage3D volumeImage;
layout(set=0, binding=4, std430) readonly buffer TriTable { int triTable[]; } mc;
layout(set=0, binding=6, std430) buffer VertSSBO { VertexData vertex_data[]; } vertices;
layout(set=0, binding=8, std430) buffer IdxSSBO { uint indices[]; } indices;
layout(set=0, binding=10,std430) buffer DescSSBO{ MeshletDesc meshletDesc[]; } descs;

// dynamic shared limits
shared VertexData sharedVerts[256];
shared uvec3    sharedIdxs [256*MAX_PRIMS_PER_CELL];
shared uint     vCount;
shared uint     pCount;

layout(local_size_x=32) in;
layout(max_vertices=256, max_primitives=256) out;
layout(triangles) out;
layout(location=0) out PerV { vec3 normal; } pv[];


// Maps a Marching Cubes edge index (0-11) to its defining corner vertices and axis.
// Corner indices (0-7) follow the standard MC pattern:
//   4----5
//  /|   /|
// 7----6 |    Y
// | 0--|-1    | / Z
// |/   |/     |/
// 3----2      o----X
//
void getEdgeInfo(int edgeIndex, out ivec3 p1_offset, out ivec3 p2_offset, out int corner1_idx, out int corner2_idx, out int axis) {
    switch (edgeIndex) {
        case 0:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(1,0,0); corner1_idx=0; corner2_idx=1; axis=0; return; // Edge 0-1 (X)
        case 1:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,1,0); corner1_idx=1; corner2_idx=2; axis=1; return; // Edge 1-2 (Y)
        case 2:  p1_offset=ivec3(0,1,0); p2_offset=ivec3(1,1,0); corner1_idx=3; corner2_idx=2; axis=0; return; // Edge 3-2 (X)
        case 3:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,1,0); corner1_idx=0; corner2_idx=3; axis=1; return; // Edge 0-3 (Y)
        case 4:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(1,0,1); corner1_idx=4; corner2_idx=5; axis=0; return; // Edge 4-5 (X)
        case 5:  p1_offset=ivec3(1,0,1); p2_offset=ivec3(1,1,1); corner1_idx=5; corner2_idx=6; axis=1; return; // Edge 5-6 (Y)
        case 6:  p1_offset=ivec3(0,1,1); p2_offset=ivec3(1,1,1); corner1_idx=7; corner2_idx=6; axis=0; return; // Edge 7-6 (X)
        case 7:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(0,1,1); corner1_idx=4; corner2_idx=7; axis=1; return; // Edge 4-7 (Y)
        case 8:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,0,1); corner1_idx=0; corner2_idx=4; axis=2; return; // Edge 0-4 (Z)
        case 9:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,0,1); corner1_idx=1; corner2_idx=5; axis=2; return; // Edge 1-5 (Z)
        case 10: p1_offset=ivec3(1,1,0); p2_offset=ivec3(1,1,1); corner1_idx=2; corner2_idx=6; axis=2; return; // Edge 2-6 (Z)
        case 11: p1_offset=ivec3(0,1,0); p2_offset=ivec3(0,1,1); corner1_idx=3; corner2_idx=7; axis=2; return; // Edge 3-7 (Z)
        default: // Should not happen
                 p1_offset=ivec3(0); p2_offset=ivec3(0); corner1_idx=-1; corner2_idx=-1; axis=-1; return;
    }
}

vec3 interpolateVertex(ivec3 p1_global, ivec3 p2_global, float val1, float val2) {

    if (abs(val1 - val2) < 1e-6f) { return vec3(p1_global); }
    float t = clamp((ubo.isovalue - val1) / (val2 - val1), 0.0f, 1.0f);
    return mix(vec3(p1_global), vec3(p2_global), t);
}

vec3 calculateNormal(ivec3 globalCellCoord, vec3 vertexPos) {
    // Use imageLoad directly, or load into shared mem first and use that
    ivec3 Npos = ivec3(round(vertexPos));
    float s1_x = float(imageLoad(volumeImage, clamp(Npos + ivec3(1,0,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s2_x = float(imageLoad(volumeImage, clamp(Npos - ivec3(1,0,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s1_y = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,1,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s2_y = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,1,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s1_z = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,0,1), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s2_z = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,0,1), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    vec3 grad = vec3(s2_x - s1_x, s2_y - s1_y, s2_z - s1_z);
    if (length(grad) < 1e-5f) return vec3(0, 1, 0);
    return -normalize(grad);
}

void main() {
    // zero
    if (gl_LocalInvocationIndex==0){
        vCount = 0;
        pCount = 0;
    }
    barrier();

    uvec3 origin = taskPayloadIn.blockOrigin;
    uvec3 dim    = taskPayloadIn.subBlockDim;
    uint cellCount = dim.x*dim.y*dim.z;

    // march each cell
    for (uint idx = gl_LocalInvocationIndex; idx < cellCount; idx += gl_WorkGroupSize.x) {
        uvec3 lc = uvec3( idx % dim.x,
        (idx/dim.x) % dim.y,
        idx/(dim.x*dim.y) );
        ivec3 cell = ivec3(origin + lc);

        // compute case
        uint cubeCase=0; float cv[8];
        for(int i=0;i<8;i++){
            ivec3 off = ivec3(i&1,(i&2)>>1,(i&4)>>2);
            ivec3 c = cell+off;
            uint v=0;
            if(all(greaterThanEqual(c,ivec3(0))) && all(lessThan(c,ivec3(ubo.volumeDim.xyz))))
            v = imageLoad(volumeImage,c).r;
            cv[i]=float(v);
            if(cv[i]>=ubo.isovalue) cubeCase|=(1u<<i);
        }
        if(cubeCase==0||cubeCase==255) continue;

        // build triangles
        int base = int(cubeCase)*16;
        uint localV[12]; for(int k=0;k<12;k++) localV[k]=0xFFFFFFFFu;
        for(int t=0;t<15;t+=3){
            int e0=mc.triTable[base+t+0];
            if(e0<0) break;
            int e1=mc.triTable[base+t+1],
            e2=mc.triTable[base+t+2];
            uvec3 tri;
            int es[3]={e0,e1,e2};
            // for each edge...
            for(int k=0;k<3;k++){
                int edge=es[k];
                if(localV[edge]==0xFFFFFFFFu){
                    // compute vertex
                    ivec3 p1,p2;int c1,c2,ax;
                    getEdgeInfo(edge,p1,p2,c1,c2,ax);
                    vec3 pos=interpolateVertex(cell+p1,cell+p2,cv[c1],cv[c2]);
                    vec3 nrm=calculateNormal(cell,pos);
                    uint vi = atomicAdd(vCount,1u);
                    sharedVerts[vi]=VertexData(vec4(pos,1),vec4(nrm,0));
                    localV[edge]=vi;
                }
                tri[k]=localV[edge];
            }
            uint pi = atomicAdd(pCount,1u);
            sharedIdxs[pi]=tri;
        }
    }

    barrier();

    uint finalV = min(vCount,256u);
    uint finalP = min(pCount,512u);

    SetMeshOutputsEXT(finalV, finalP);

    // write out
    uint gV = taskPayloadIn.globalVertexOffset;
    for(uint i=gl_LocalInvocationIndex;i<finalV;i+=gl_WorkGroupSize.x){
        VertexData vd = sharedVerts[i];
        gl_MeshVerticesEXT[i].gl_Position = vd.position;
        pv[i].normal = vd.normal.xyz;
        vertices.vertex_data[gV+i]=vd;
    }
    uint gI = taskPayloadIn.globalIndexOffset;
    for(uint i=gl_LocalInvocationIndex;i<finalP;i+=gl_WorkGroupSize.x){
        uvec3 tri = sharedIdxs[i];
        gl_PrimitiveTriangleIndicesEXT[i]=tri;
        indices.indices[gI+i*3+0] = tri.x + gV;
        indices.indices[gI+i*3+1] = tri.y + gV;
        indices.indices[gI+i*3+2] = tri.z + gV;
    }

    // write descriptor
    barrier();
    if(gl_LocalInvocationIndex==0){
        uint d = taskPayloadIn.globalMeshletDescOffset;
        descs.meshletDesc[d] = MeshletDesc(gV, gI, finalV, finalP);
    }
}