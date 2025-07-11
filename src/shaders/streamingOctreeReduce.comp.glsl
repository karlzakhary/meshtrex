#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_debug_printf : enable

/* ── Push constants ────────────────────────────────────────────────
   Octree reduction for streaming min-max computation
   Note: This is identical to the non-streaming version because
   min-max images are NOT sparse - only the volume atlas is sparse */
layout(push_constant) uniform PushConstants {
    uvec3 srcDim;         // Source dimensions for this level
    uvec3 dstDim;         // Destination dimensions for next level
} pc;

/* ── Descriptor Set Bindings ──────────────────────────────────────── */
layout(set = 0, binding = 0, rg32ui) uniform readonly uimage3D srcLevel;   // Level L (input)
layout(set = 0, binding = 1, rg32ui) uniform writeonly uimage3D dstLevel; // Level L+1 (output)

/* One work-group reduces ONE dst texel  →  use 8 threads (1 per child). */
layout(local_size_x = 2, local_size_y = 2, local_size_z = 2) in;

void main()
{
    /* which (x,y,z) texel of the *destination* image this WG writes */
    uvec3 dst = gl_WorkGroupID.xyz;              /* 0 … dstDim-1 */
    /* which texel (= child) THIS invocation processes inside the 2×2×2 block */
    uvec3 child = gl_LocalInvocationID.xyz;      /* 0 or 1 in each axis */

    /* --------------------------------------------
       1 · load min,max of our child texel
       -------------------------------------------- */
    uvec3 src = dst * 2u + child;                /* child's coords in level L */

    /* out-of-range test: happens only at the           *
     * border when srcDim is not a power of two.        */
    bool valid = all(lessThan(src, pc.srcDim));
    
    uint vMin = 0xFFFFFFFFu;
    uint vMax = 0u;
    
    if (valid) {
        uvec2 minMax = imageLoad(srcLevel, ivec3(src)).xy;
        vMin = minMax.x;
        vMax = minMax.y;
    }

    /* --------------------------------------------
       2 · subgroup reduction (8-lane subgroup)
       -------------------------------------------- */
    for (uint off = 1u; off < 8u; off <<= 1u) {
        vMin = min(vMin, subgroupShuffleXor(vMin, off));
        vMax = max(vMax, subgroupShuffleXor(vMax, off));
    }

    /* --------------------------------------------
       3 · Write the reduced value
       -------------------------------------------- */
    if (subgroupElect()) { /* one TRUE per subgroup */
        // Just ensure we're within bounds
        if (all(lessThan(dst, pc.dstDim))) {
            imageStore(dstLevel, ivec3(dst), uvec4(vMin, vMax, 0u, 0u));
        }
    }
}