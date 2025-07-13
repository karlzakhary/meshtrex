#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_debug_printf: require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Input: Active block count from filtering pass
layout(binding = 0) readonly buffer ActiveBlockCountBuffer {
    uint activeBlockCount;
};

// Output: Indirect draw command for mesh tasks
struct DrawMeshTasksIndirectCommand {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

layout(binding = 1) writeonly buffer IndirectDrawBuffer {
    DrawMeshTasksIndirectCommand indirectCommand;
};

void main() {
    // Read the active block count from the filtering pass
    uint taskCount = activeBlockCount;
    debugPrintfEXT("Task count: %u", taskCount);
    
    // Write the indirect draw command for mesh task dispatch
    // If taskCount is 0, the mesh shader dispatch will be skipped automatically
    indirectCommand.groupCountX = taskCount;
    indirectCommand.groupCountY = (taskCount > 0) ? 1 : 0;
    indirectCommand.groupCountZ = (taskCount > 0) ? 1 : 0;
}