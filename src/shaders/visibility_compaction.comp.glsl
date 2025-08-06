#version 460 core
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_debug_printf : enable

// Large workgroup for efficient compaction
#define WORKGROUP_SIZE 1024

layout(local_size_x = WORKGROUP_SIZE) in;

// Input: Sparse visibility flags (1 uint per block) from fragment shader
layout(set = 0, binding = 0) restrict readonly buffer VisibilityBuffer {
    uint visibles[];
} visibility;

// Output: Compacted bitfield (1 bit per block, 32 blocks per uint)
layout(set = 0, binding = 1) restrict writeonly buffer CompactedVisibility {
    uint bitfield[];
} compactedVis;

// Push constants
layout(push_constant) uniform PushConstants {
    uint totalBlocks;
} pushConstants;

void main() {
    uint globalID = gl_GlobalInvocationID.x;
    
    // Each thread processes one block
    if (globalID < pushConstants.totalBlocks) {
        // Check if this block is visible
        bool isVisible = (visibility.visibles[globalID] != 0);
        
        // Use subgroup ballot to collect visibility bits from 32 consecutive threads
        uvec4 ballot = subgroupBallot(isVisible);
        
        // Thread 31 of each subgroup writes the compacted result
        if ((gl_LocalInvocationID.x % 32) == 31) {
            uint bitfieldIndex = globalID / 32;
            compactedVis.bitfield[bitfieldIndex] = ballot.x;
            
            // Debug output for first few bitfield entries
            if (bitfieldIndex < 3 && ballot.x != 0) {
                debugPrintfEXT("VisCompaction: bitfield[%d] = 0x%08x (%d visible blocks)", 
                              bitfieldIndex, ballot.x, bitCount(ballot.x));
            }
        }
    }
}