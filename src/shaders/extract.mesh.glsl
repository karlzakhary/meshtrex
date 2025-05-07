#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// one invocation per meshlet
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
// --- Output Vertex/Primitive Counts ---
// These MUST be <= device limits (e.g., 256 verts, 256 prims for basic mesh shader)
// We aim for the paper's target values.
layout(max_vertices = 128) out;
layout(max_primitives = 256) out; // Adjust based on C++ limits if different
layout(triangles) out; // Output triangles

#define MAX_MESH_TASKS_PER_TASK_SHADER 64u // Max 2x2x2 sub-blocks in an 8x8x8

// match your task‐shader SubBlockInfoPayload
struct SubBlockInfoPayload {
    uvec3 parentBlockOrigin;
    uvec3 subBlockOffset;
    uvec3 subBlockDim;
    uint baseVertexOffset;
    uint baseIndexOffset;
    uint baseDescriptorOffset;
};

struct MeshletDescriptorLayout { uint vO, iO, vC, pC; };

// pulled in from the task‐shader
taskPayloadSharedEXT SubBlockInfoPayload taskPayload[MAX_MESH_TASKS_PER_TASK_SHADER];

// same UBO/bindings as your task shader
layout(set = 0, binding = 0, scalar) uniform ExtractionConstantsUBO {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeTexture;
layout(set = 0, binding = 4, scalar) buffer MarchingCubesTriangleTable { int triTable[]; };
layout(set = 0, binding = 5, scalar) buffer MarchingCubesNumberVertices { int numVertices[]; };

// where we write actual geometry
layout(set = 0, binding = 6, scalar) buffer VertexBuffer {
    uint vertexCounter;    // not used here
    vec3 positions[];
};
layout(set = 0, binding = 7, scalar) buffer IndexBuffer {
    uint indexCounter;     // not used here
    uint indices[];
};
layout(set = 0, binding = 8, scalar) buffer MeshletDescriptorBuffer {
    uint meshletCounter;   // not used here
    MeshletDescriptorLayout descriptors[];
};

// helper to unpack Morton 3-bit
uint Morton_SpreadBits_3bit(uint x) {
    x &= 7u;
    x = (x | (x << 8))  & 0x0000F00Fu;
    x = (x | (x << 4))  & 0x000C30C3u;
    x = (x | (x << 2))  & 0x00249249u;
    return x;
}
uvec3 mortonDecode3D_3bit(uint m) {
    uint x = m;
    uint y = m >> 1;
    uint z = m >> 2;
    x = (x & 0x00249249u);
    y = (y & 0x00249249u);
    z = (z & 0x00249249u);
    x = (x | (x >> 2)) & 0x000C30C3u; x = (x | (x >> 4)) & 0x0000F00Fu; x = (x | (x >> 8)) & 0x0000007Fu;
    y = (y | (y >> 2)) & 0x000C30C3u; y = (y | (y >> 4)) & 0x0000F00Fu; y = (y | (y >> 8)) & 0x0000007Fu;
    z = (z | (z >> 2)) & 0x000C30C3u; z = (z | (z >> 4)) & 0x0000F00Fu; z = (z | (z >> 8)) & 0x0000007Fu;
    return uvec3(x,y,z);
}
int getTriEntry(uint mc_case, uint vertNum) {
    return int(triTable[mc_case * 16 + vertNum]);
}

// interpolate along the given edge of the cell
vec3 edgeInterp(ivec3 p0, ivec3 p1) {
    float v0 = float(imageLoad(volumeTexture, p0).r);
    float v1 = float(imageLoad(volumeTexture, p1).r);
    float t = (ubo.isovalue - v0) / (v1 - v0);
    return mix(vec3(p0), vec3(p1), t);
}

void main() {
    // only one invocation
    SubBlockInfoPayload info = taskPayload[gl_WorkGroupID.x];

    uvec3 origin = info.parentBlockOrigin + info.subBlockOffset;
    uvec3 dim    = info.subBlockDim;

    uint vCount = 0;
    uint pCount = 0;

    // loop cells
    for (uint z = 0; z < dim.z; ++z) {
        for (uint y = 0; y < dim.y; ++y) {
            for (uint x = 0; x < dim.x; ++x) {
                ivec3 cell = ivec3(origin + uvec3(x,y,z));
                // build MC case
                uint mc = 0;
                bool bdry = false;
                for (int b = 0; b < 8; ++b) {
                    ivec3 off = ivec3(b & 1, (b>>1)&1, (b>>2)&1);
                    ivec3 p   = cell + off;
                    if (any(lessThan(p, ivec3(0))) ||
                    any(greaterThanEqual(p, ivec3(ubo.volumeDim.xyz)))) {
                        bdry = true;
                        break;
                    }
                    uint val = imageLoad(volumeTexture, p).r;
                    if (float(val) >= ubo.isovalue) mc |= 1u << b;
                }
                if (bdry || mc == 0u || mc == 255u) continue;

                // emit triangles
                for (uint e = 0; e < 12; ++e) {
                    int edge = int(getTriEntry(mc, e));
                    if (edge < 0) break;

                    if (e % 3 == 0) {
                        // start a new triangle
                        ivec3 offs0, offs1;
                        // map edge -> (p0,p1)
                        switch (edge) {
                            case 0:  offs0=ivec3(0,0,0); offs1=ivec3(1,0,0); break;
                            case 1:  offs0=ivec3(1,0,0); offs1=ivec3(1,1,0); break;
                            case 2:  offs0=ivec3(1,1,0); offs1=ivec3(0,1,0); break;
                            case 3:  offs0=ivec3(0,1,0); offs1=ivec3(0,0,0); break;
                            case 4:  offs0=ivec3(0,0,1); offs1=ivec3(1,0,1); break;
                            case 5:  offs0=ivec3(1,0,1); offs1=ivec3(1,1,1); break;
                            case 6:  offs0=ivec3(1,1,1); offs1=ivec3(0,1,1); break;
                            case 7:  offs0=ivec3(0,1,1); offs1=ivec3(0,0,1); break;
                            case 8:  offs0=ivec3(0,0,0); offs1=ivec3(0,0,1); break;
                            case 9:  offs0=ivec3(1,0,0); offs1=ivec3(1,0,1); break;
                            case 10: offs0=ivec3(1,1,0); offs1=ivec3(1,1,1); break;
                            case 11: offs0=ivec3(0,1,0); offs1=ivec3(0,1,1); break;
                            default: continue;
                        }
                        // three vertices per triangle
                        vec3 pA = edgeInterp(cell + offs0, cell + offs1);
                        // next two edges will reuse same switch above
                        int edgeB = getTriEntry(mc, e+1);
                        int edgeC = getTriEntry(mc, e+2);
                        ivec3 b0,b1,c0,c1;
                        // (repeat switch for B)
                        switch(edgeB){
                            case 0:  offs0=ivec3(0,0,0); offs1=ivec3(1,0,0); break;
                            case 1:  offs0=ivec3(1,0,0); offs1=ivec3(1,1,0); break;
                            case 2:  offs0=ivec3(1,1,0); offs1=ivec3(0,1,0); break;
                            case 3:  offs0=ivec3(0,1,0); offs1=ivec3(0,0,0); break;
                            case 4:  offs0=ivec3(0,0,1); offs1=ivec3(1,0,1); break;
                            case 5:  offs0=ivec3(1,0,1); offs1=ivec3(1,1,1); break;
                            case 6:  offs0=ivec3(1,1,1); offs1=ivec3(0,1,1); break;
                            case 7:  offs0=ivec3(0,1,1); offs1=ivec3(0,0,1); break;
                            case 8:  offs0=ivec3(0,0,0); offs1=ivec3(0,0,1); break;
                            case 9:  offs0=ivec3(1,0,0); offs1=ivec3(1,0,1); break;
                            case 10: offs0=ivec3(1,1,0); offs1=ivec3(1,1,1); break;
                            case 11: offs0=ivec3(0,1,0); offs1=ivec3(0,1,1); break;
                            default: continue;
                        }
                        vec3 pB = edgeInterp(cell + b0, cell + b1);
                        // (and C)
                        switch(edgeC){
                            case 0:  offs0=ivec3(0,0,0); offs1=ivec3(1,0,0); break;
                            case 1:  offs0=ivec3(1,0,0); offs1=ivec3(1,1,0); break;
                            case 2:  offs0=ivec3(1,1,0); offs1=ivec3(0,1,0); break;
                            case 3:  offs0=ivec3(0,1,0); offs1=ivec3(0,0,0); break;
                            case 4:  offs0=ivec3(0,0,1); offs1=ivec3(1,0,1); break;
                            case 5:  offs0=ivec3(1,0,1); offs1=ivec3(1,1,1); break;
                            case 6:  offs0=ivec3(1,1,1); offs1=ivec3(0,1,1); break;
                            case 7:  offs0=ivec3(0,1,1); offs1=ivec3(0,0,1); break;
                            case 8:  offs0=ivec3(0,0,0); offs1=ivec3(0,0,1); break;
                            case 9:  offs0=ivec3(1,0,0); offs1=ivec3(1,0,1); break;
                            case 10: offs0=ivec3(1,1,0); offs1=ivec3(1,1,1); break;
                            case 11: offs0=ivec3(0,1,0); offs1=ivec3(0,1,1); break;
                            default: continue;
                        }
                        vec3 pC = edgeInterp(cell + c0, cell + c1);

                        // write
                        uint i0 = info.baseVertexOffset + (vCount++);
                        positions[i0] = pA;
                        uint i1 = info.baseVertexOffset + (vCount++);
                        positions[i1] = pB;
                        uint i2 = info.baseVertexOffset + (vCount++);
                        positions[i2] = pC;

                        uint iOff = info.baseIndexOffset + (pCount*3);
                        indices[iOff + 0] = i0;
                        indices[iOff + 1] = i1;
                        indices[iOff + 2] = i2;
                        ++pCount;
                    }
                }
            }
        }
    }

    // store descriptor for this meshlet
    descriptors[info.baseDescriptorOffset].vO = info.baseVertexOffset;
    descriptors[info.baseDescriptorOffset].iO = info.baseIndexOffset;
    descriptors[info.baseDescriptorOffset].vC = vCount;
    descriptors[info.baseDescriptorOffset].pC = pCount;
}