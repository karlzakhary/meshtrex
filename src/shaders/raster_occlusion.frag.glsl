#version 460 core
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_debug_printf : enable

// Enable early fragment tests for depth testing
layout(early_fragment_tests) in;

// Visibility buffer SSBO - sparse array, one uint per block
layout(set = 0, binding = 2) restrict writeonly coherent buffer VisibilityBuffer {
    uint visibles[];
} visibility;

// Per-primitive input from mesh shader
perprimitiveEXT layout(location = 0) in Interpolants {
    flat uint blockID;
} inPrimitive;

void main() {
    // This fragment shader only executes for fragments that pass the depth test
    // Write 1 to indicate this block is visible
    visibility.visibles[inPrimitive.blockID] = 1;
}