#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle   : require
#extension GL_EXT_debug_printf : enable


/* ── Push constants ────────────────────────────────────────────────
   srcDim = resolution of the CURRENT level      (e.g. 128³ … 2³)
   dstDim = resolution of the NEXT  level        (always = srcDim / 2)
   They are set by the CPU before every dispatch.                    */
layout(push_constant) uniform PC {
    uvec3 srcDim;      /* width,height,depth of level L   */
    uvec3 dstDim;      /* width,height,depth of level L+1 */
} pc;

/* ── Bindings ────────────────────────────────────────────────────── */
layout(binding = 0, rg32ui)  uniform readonly  uimage3D srcLevel;  /* level L   */
layout(binding = 1, rg32ui)  uniform writeonly uimage3D dstLevel;  /* level L+1 */

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
    uvec3 src = dst * 2u + child;                /* child’s coords in level L */

    /* out-of-range test: happens only at the           *
     * border when srcDim is not a power of two.        */
    bool valid =
        all(lessThan(src, pc.srcDim));

    uint vMin = 0xFFFFFFFFu;
    uint vMax = 0u;
    if (valid) {
        uvec2 mm = imageLoad(srcLevel, ivec3(src)).xy;
        vMin = mm.x;
        vMax = mm.y;
    }

    /* --------------------------------------------
       2 · subgroup reduction (8-lane subgroup)
       -------------------------------------------- */
    for (uint off = 1u; off < 8u; off <<= 1u) {
        vMin = min(vMin, subgroupShuffleXor(vMin, off));
        vMax = max(vMax, subgroupShuffleXor(vMax, off));
    }

    /* lane 0 writes the result */
    if (subgroupElect())                /* one TRUE per subgroup      */
        imageStore(dstLevel, ivec3(dst), uvec4(vMin, vMax, 0u, 0u));
}