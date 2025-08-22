#version 460 core
#extension GL_EXT_scalar_block_layout : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Input: Previous level of the pyramid (as a storage image)
layout(set = 0, binding = 0, r32f) uniform readonly image2D prevLevelInput;

// Output: Current level of the pyramid
layout(set = 0, binding = 1, r32f) uniform writeonly image2D currentLevelOutput;

void main() {
    ivec2 outputCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outputSize = imageSize(currentLevelOutput);
    
    if (any(greaterThanEqual(outputCoord, outputSize))) {
        return;
    }
    
    ivec2 inputBase = outputCoord * 2;
    
    // Sample 2x2 region from the previous, higher-resolution level
    float d00 = imageLoad(prevLevelInput, inputBase + ivec2(0, 0)).r;
    float d10 = imageLoad(prevLevelInput, inputBase + ivec2(1, 0)).r;
    float d01 = imageLoad(prevLevelInput, inputBase + ivec2(0, 1)).r;
    float d11 = imageLoad(prevLevelInput, inputBase + ivec2(1, 1)).r;

    // Reversed-Z: Use MAX depth for conservative test
    // With reversed-Z, closer objects have HIGHER depth values (near 1.0)
    // An object is occluded if its depth is less than the maximum (closest) occluder
    float maxDepth = max(max(d00, d10), max(d01, d11));
    
    imageStore(currentLevelOutput, outputCoord, vec4(maxDepth));
}