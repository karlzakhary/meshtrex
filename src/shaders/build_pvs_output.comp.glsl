#version 460 core
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_debug_printf : enable

// Warp size for efficient reduction
#define WORKGROUP_SIZE 32

layout(local_size_x = WORKGROUP_SIZE) in;

// Input: Current frame visibility bitfield
layout(set = 0, binding = 0) restrict readonly buffer CurrentVisibilityBits {
    uint bitfield[];
} currentVis;

// Input: Previous frame visibility bitfield
layout(set = 0, binding = 1) restrict readonly buffer PreviousVisibilityBits {
    uint bitfield[];
} previousVis;

// Output: PVS current frame (dense array of block indices)
layout(set = 0, binding = 2) coherent buffer PVSCurrentBuffer {
    uint count;
    uint indices[];
} pvsCurrent;

// Output: PVS difference (current but not last frame)
layout(set = 0, binding = 3) coherent buffer PVSDifferenceBuffer {
    uint count;
    uint indices[];
} pvsDifference;

// Push constants
layout(push_constant) uniform PushConstants {
    uint numBitfieldEntries;  // Total number of uint32s in bitfield
} pushConstants;

// Shared memory for warp-level reductions
shared uint s_warpOffsetCurrent;
shared uint s_warpOffsetDifference;

void main() {
    uint threadID = gl_LocalInvocationID.x;
    uint bitfieldIndex = gl_GlobalInvocationID.x;
    
    // Initialize shared memory
    if (threadID == 0) {
        s_warpOffsetCurrent = 0;
        s_warpOffsetDifference = 0;
    }
    barrier();
    
    uint visibleBlocksCurrent = 0;
    uint visibleBlocksDifference = 0;
    uint threadVisibleCountCurrent = 0;
    uint threadVisibleCountDifference = 0;
    
    if (bitfieldIndex < pushConstants.numBitfieldEntries) {
        // Load visibility bits for this thread
        uint currentBits = currentVis.bitfield[bitfieldIndex];
        uint previousBits = previousVis.bitfield[bitfieldIndex];
        
        // Compute difference: visible now but not in previous frame
        uint differenceBits = currentBits & (~previousBits);
        
        // Count visible blocks for this thread
        threadVisibleCountCurrent = bitCount(currentBits);
        threadVisibleCountDifference = bitCount(differenceBits);
        
        // Store for later processing
        visibleBlocksCurrent = currentBits;
        visibleBlocksDifference = differenceBits;
    }
    
    // Perform warp-level prefix sum using subgroup operations
    uint prefixSumCurrent = subgroupExclusiveAdd(threadVisibleCountCurrent);
    uint prefixSumDifference = subgroupExclusiveAdd(threadVisibleCountDifference);
    
    // Last thread in warp gets total count
    uint warpTotalCurrent = subgroupAdd(threadVisibleCountCurrent);
    uint warpTotalDifference = subgroupAdd(threadVisibleCountDifference);
    
    // Last thread allocates space in output arrays
    uint globalOffsetCurrent = 0;
    uint globalOffsetDifference = 0;
    
    if (threadID == 31) {
        if (warpTotalCurrent > 0) {
            globalOffsetCurrent = atomicAdd(pvsCurrent.count, warpTotalCurrent);
        }
        if (warpTotalDifference > 0) {
            globalOffsetDifference = atomicAdd(pvsDifference.count, warpTotalDifference);
        }
        s_warpOffsetCurrent = globalOffsetCurrent;
        s_warpOffsetDifference = globalOffsetDifference;
        
        // Debug output for first few workgroups
        // if (gl_WorkGroupID.x < 3) {
        //     debugPrintfEXT("BuildPVS WG %d: found %d current, %d difference blocks", 
        //                   gl_WorkGroupID.x, warpTotalCurrent, warpTotalDifference);
        // }
    }
    
    barrier();
    
    // Each thread writes its visible block indices
    if (bitfieldIndex < pushConstants.numBitfieldEntries && threadVisibleCountCurrent > 0) {
        uint writeOffsetCurrent = s_warpOffsetCurrent + prefixSumCurrent;
        uint writeOffsetDifference = s_warpOffsetDifference + prefixSumDifference;
        
        uint baseBlockIndex = bitfieldIndex * 32;
        
        // Write indices for current frame visible blocks
        uint numWrittenCurrent = 0;
        uint numWrittenDifference = 0;
        
        for (uint bit = 0; bit < 32; bit++) {
            uint mask = 1u << bit;
            
            if ((visibleBlocksCurrent & mask) != 0) {
                pvsCurrent.indices[writeOffsetCurrent + numWrittenCurrent] = baseBlockIndex + bit;
                numWrittenCurrent++;
            }
            
            if ((visibleBlocksDifference & mask) != 0) {
                pvsDifference.indices[writeOffsetDifference + numWrittenDifference] = baseBlockIndex + bit;
                numWrittenDifference++;
            }
        }
    }
}