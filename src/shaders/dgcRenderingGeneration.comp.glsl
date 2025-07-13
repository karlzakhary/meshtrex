#version 460
#extension GL_EXT_debug_printf : enable

// DGC compute shader for rendering pipeline
// Reads meshlet descriptor count from extraction output and generates indirect draw command

// Input: Meshlet descriptor count from extraction
layout(set = 0, binding = 0, std430) readonly buffer MeshletDescriptorCount {
    uint meshletCount;
};

// Output: Indirect draw command for rendering task/mesh shaders
layout(set = 0, binding = 1, std430) writeonly buffer IndirectCommands {
    // VkDrawMeshTasksIndirectCommandEXT structure
    uint groupCountX;
    uint groupCountY; 
    uint groupCountZ;
} indirectCommand;

// Push constants for configuration
layout(push_constant) uniform PushConstants {
    uint taskWorkgroupSize;  // How many meshlets each task workgroup processes
} pc;

void main() {
    // Only thread 0 writes the command
    if (gl_GlobalInvocationID.x != 0) return;
    
    // Read the meshlet count from extraction output
    uint totalMeshlets = meshletCount;
    
    debugPrintfEXT("DGC Rendering: Total meshlets from extraction: %u", totalMeshlets);
    
    // Calculate number of task workgroups needed
    // Each task workgroup will process 'taskWorkgroupSize' meshlets
    uint taskGroups = (totalMeshlets + pc.taskWorkgroupSize - 1) / pc.taskWorkgroupSize;
    
    debugPrintfEXT("DGC Rendering: Dispatching %u task groups (workgroup size: %u)", 
                   taskGroups, pc.taskWorkgroupSize);
    
    // Write the indirect draw command
    indirectCommand.groupCountX = taskGroups;
    indirectCommand.groupCountY = 1;
    indirectCommand.groupCountZ = 1;
}