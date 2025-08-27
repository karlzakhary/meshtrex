#version 460
#extension GL_EXT_scalar_block_layout : require

// Compute shader to update indirect dispatch buffer for compute occlusion culling
// Converts block count to workgroup count for compute dispatch

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Structure matching VkDispatchIndirectCommand
struct IndirectCommand {
    uint x;
    uint y;
    uint z;
};

// Output: Indirect dispatch buffer
layout(binding = 0, std430) writeonly buffer IndirectDispatchBuffer {
    IndirectCommand command;
} indirectDispatch;

// Push constants for configuration
layout(push_constant) uniform PushConstants {
    uint totalBlockCount;  // Total number of blocks to process
} pushConstants;

void main() {
    // Only one thread needed
    if (gl_GlobalInvocationID.x != 0) return;
    
    // Calculate workgroup count for compute occlusion culling
    // Matching the shader's workgroup size of 256 threads
    uint workgroupSize = 256u;
    uint numWorkgroups = (pushConstants.totalBlockCount + workgroupSize - 1u) / workgroupSize;
    
    // Write indirect dispatch command
    indirectDispatch.command.x = numWorkgroups;
    indirectDispatch.command.y = 1;
    indirectDispatch.command.z = 1;
}