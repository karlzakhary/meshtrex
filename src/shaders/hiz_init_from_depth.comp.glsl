#version 460 core
#extension GL_EXT_scalar_block_layout : enable

// Copies depth buffer to first level of Hi-Z pyramid
// Reads from depth texture and writes to storage image
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;


layout(set = 0, binding = 0) uniform sampler2D depthTexture;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D hiZLevel0;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(hiZLevel0);
    
    if (coord.x >= imageSize.x || coord.y >= imageSize.y) {
        return;
    }
    
    // Sample depth value from depth buffer
    vec2 uv = (vec2(coord) + 0.5) / vec2(imageSize);
    vec4 val = texelFetch(depthTexture, coord, 0);
    
    imageStore(hiZLevel0, coord, val);
}