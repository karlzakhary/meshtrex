#version 460
#extension GL_EXT_scalar_block_layout : require

// Compute shader to update indirect draw buffer from PVS counts
// This enables fully GPU-driven rendering without CPU readback

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Structure matching VkDrawMeshTasksIndirectCommandEXT
struct IndirectCommand {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

// Input: PVS count buffers from occlusion culling
layout(binding = 0, std430) readonly buffer PrevCountBuffer {
    uint count;
} pvsPrevCount;

layout(binding = 1, std430) readonly buffer DiffCountBuffer {
    uint count;
} pvsDiffCount;

// Output: Indirect draw buffer
layout(binding = 2, std430) writeonly buffer IndirectDrawBuffer {
    IndirectCommand commands[2];  // [0] = Pass 1 (previous), [1] = Pass 2 (difference)
} indirectDraw;

// Push constants for configuration
layout(push_constant) uniform PushConstants {
    uint bypassPVS;        // 1 = bypass PVS, use all blocks
    uint totalBlockCount;  // Total number of blocks (for bypass mode)
} pushConstants;

void main() {
    // Only one thread needed
    if (gl_GlobalInvocationID.x != 0) return;
    
    // Pass 1: Previous visible blocks
    // Each block requires 2 workgroups (for split processing)
    if (pushConstants.bypassPVS == 1) {
        // Bypass mode: process all blocks
        indirectDraw.commands[0].groupCountX = pushConstants.totalBlockCount * 2;
    } else {
        // Normal mode: use PVS count
        indirectDraw.commands[0].groupCountX = pvsPrevCount.count * 2;
    }
    indirectDraw.commands[0].groupCountY = 1;
    indirectDraw.commands[0].groupCountZ = 1;
    
    // Pass 2: Newly visible blocks (difference)
    if (pushConstants.bypassPVS == 1) {
        // In bypass mode, Pass 2 should not run (no temporal difference)
        indirectDraw.commands[1].groupCountX = 0;
    } else {
        // Normal mode: use difference count
        indirectDraw.commands[1].groupCountX = pvsDiffCount.count * 2;
    }
    indirectDraw.commands[1].groupCountY = 1;
    indirectDraw.commands[1].groupCountZ = 1;
}