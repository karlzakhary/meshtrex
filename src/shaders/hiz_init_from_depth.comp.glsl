#version 460 core
#extension GL_EXT_scalar_block_layout : enable

// Copies depth buffer to first level of Hi-Z pyramid
// Reads from depth texture and writes to storage image

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Input: Depth buffer as sampler
layout(set = 0, binding = 0) uniform sampler2D depthTexture;

// Output: First level of Hi-Z pyramid
layout(set = 0, binding = 1, r32f) uniform writeonly image2D hiZLevel0;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(hiZLevel0);
    
    // Check bounds
    if (coord.x >= imageSize.x || coord.y >= imageSize.y) {
        return;
    }
    
    // Sample depth value from depth buffer
    vec2 uv = (vec2(coord) + 0.5) / vec2(imageSize);
    float depth = texelFetch(depthTexture, coord, 0).r;
    
    // Reversed-Z: depth buffer is cleared to 0.0 (far plane)
    // No need to check for uninitialized values since 0.0 is already correct
    
    // Write to Hi-Z pyramid level 0
    imageStore(hiZLevel0, coord, vec4(depth));
}