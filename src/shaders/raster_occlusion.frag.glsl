#version 460 core
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_mesh_shader : require

#define DEBUG_OCCLUSION 0

// Enable early fragment tests for depth testing
layout(early_fragment_tests) in;

// Visibility buffer SSBO - sparse array, one uint per block
layout(set = 0, binding = 2) restrict coherent buffer VisibilityBuffer {
    uint visibles[];
} visibility;

// Debug statistics buffer (optional)
layout(set = 0, binding = 3) coherent buffer DebugStats {
    uint blocksQueried;
    uint blocksVisible;
    uint atomicOps;
} debugStats;

// Per-primitive input from mesh shader
perprimitiveEXT layout(location = 0) in Interpolants {
    flat uint blockID;
} inPrimitive;

void main() {
    // This fragment shader only executes for fragments that pass the depth test
    // Write 1 to indicate this block is visible

    #ifdef DEBUG_OCCLUSION
    uint oldValue = atomicOr(visibility.visibles[inPrimitive.blockID], 1);
    
    atomicAdd(debugStats.atomicOps, 1);
    if (oldValue == 0) {
        atomicAdd(debugStats.blocksVisible, 1);
    }
    #else
    visibility.visibles[inPrimitive.blockID] = 1;
    #endif
    
}