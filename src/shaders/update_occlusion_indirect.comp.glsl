#version 460
#extension GL_EXT_scalar_block_layout : require

// Compute shader to update indirect draw buffer for occlusion culling
// Converts block count to workgroup count for task/mesh shaders

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Structure matching VkDrawMeshTasksIndirectCommandEXT
struct IndirectCommand {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

// Output: Indirect draw buffer
layout(binding = 0, std430) writeonly buffer IndirectDrawBuffer {
    IndirectCommand command;
} indirectDraw;

// Push constants for configuration
layout(push_constant) uniform PushConstants {
    uint totalBlockCount;  // Total number of blocks to process
} pushConstants;

void main() {
    // Only one thread needed
    if (gl_GlobalInvocationID.x != 0) return;
    
    // Calculate workgroup count for occlusion culling
    // Blocks are processed in 8x8x8 groups (512 blocks), split into two 8x8x4 halves
    // So we need 2 workgroups per 8x8x8 = 512 blocks
    uint blocksPerGroup = 8u * 8u * 8u; // 512
    uint numGroups = (pushConstants.totalBlockCount + blocksPerGroup - 1u) / blocksPerGroup;
    uint numWorkgroups = numGroups * 2u; // 2 workgroups per 8x8x8 group
    
    // Write indirect draw command
    indirectDraw.command.groupCountX = numWorkgroups;
    indirectDraw.command.groupCountY = 1;
    indirectDraw.command.groupCountZ = 1;
}