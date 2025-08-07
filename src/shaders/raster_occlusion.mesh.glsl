#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

// Maximum vertices and primitives per mesh shader
#define WORKGROUP_SIZE 32
#define MAX_VERTICES 96  // 32 blocks * 4 vertices per quad
#define MAX_PRIMITIVES 126 // 32 blocks * 2 triangles per quad

layout(local_size_x = WORKGROUP_SIZE) in;
layout(triangles, max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;

// UBO bindings
layout(set = 0, binding = 0) uniform ViewUniforms {
    mat4 viewProj;
    uvec4 volumeDim;      // Volume dimensions in voxels
    uvec4 blockDim;       // Block dimensions (8, 8, 4, 1)
    uvec4 blockGridDim;   // Number of blocks in each dimension
    float isovalue;
} view;

// For compatibility, params is just an alias to view
#define params view

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pushConstants;

// Input from task shader - matches memory interface
taskPayloadSharedEXT struct Task {
    uint baseID;              // Workgroup ID from task shader
    uint numOccupiedBlocks;   // Number of occupied blocks in this group (up to 256 for 8x8x4)
    uint8_t denseOccupancyIndex[256]; // Dense array of local block indices for 8x8x4 blocks
} taskInput;

// Per-primitive output - block ID for fragment shader
perprimitiveEXT layout(location = 0) out Interpolants {
    flat uint blockID;
} outPrimitive[];

// Unpack linear index to 3D coordinates within group
uvec3 unflattenIndex(uint index, uvec2 extents) {
    uvec3 result;
    result.z = index / extents.y;
    index -= result.z * extents.y;
    result.y = index / extents.x;
    result.x = index % extents.x;
    return result;
}

// Convert group-local block to global block ID
uint getGlobalBlockID(uint workgroupID, uint localBlockIndex) {
    //    58 -      // Reconstruct which block group this is (processing 8x8x4 groups)
    //    58 +      // Reconstruct which 8x8x4 half this workgroup processes
    //    59 +      // We dispatch 2 workgroups per 8x8x8 region
    //    60        uint groupsPerRow = (params.blockGridDim.x + 7) / 8;
    //    61        uint groupsPerSlice = groupsPerRow * ((params.blockGridDim.y + 7) / 8);
    //    62        
    //    63 -      uint groupID = workgroupID;
    //    63 +      // Each 8x8x8 region gets 2 workgroups (lower and upper half)
    //    64 +      uint regionID = workgroupID / 2;
    //    65 +      uint halfIndex = workgroupID % 2; // 0 = lower half, 1 = upper half
    //    66        
    //    67 -      // Unpack group coordinates
    //    68 -      uvec3 groupCoord;
    //    69 -      groupCoord.z = groupID / groupsPerSlice;
    //    70 -      uint temp = groupID % groupsPerSlice;
    //    71 -      groupCoord.y = temp / groupsPerRow;
    //    72 -      groupCoord.x = temp % groupsPerRow;
    //    67 +      // Unpack region coordinates
    //    68 +      uvec3 regionCoord;
    //    69 +      regionCoord.z = regionID / groupsPerSlice;
    //    70 +      uint temp = regionID % groupsPerSlice;
    //    71 +      regionCoord.y = temp / groupsPerRow;
    //    72 +      regionCoord.x = temp % groupsPerRow;
    //    73        
    //    74 -      // Starting block for this group (8x8x4)
    //    75 -      uvec3 groupStartBlock = groupCoord * uvec3(8, 8, 4);
    //    74 +      // Starting block for this 8x8x4 half
    //    75 +      uvec3 groupStartBlock = regionCoord * uvec3(8, 8, 8) + uvec3(0, 0, halfIndex * 4);
    // Reconstruct which 8x8x4 half this workgroup processes
    // We dispatch 2 workgroups per 8x8x8 region
    uint groupsPerRow = (params.blockGridDim.x + 7) / 8;
    uint groupsPerSlice = groupsPerRow * ((params.blockGridDim.y + 7) / 8);
    
    // Each 8x8x8 region gets 2 workgroups (lower and upper half)
    uint regionID = workgroupID / 2;
    uint halfIndex = workgroupID % 2; // 0 = lower half, 1 = upper half
    
    // Unpack region coordinates
    uvec3 regionCoord;
    regionCoord.z = regionID / groupsPerSlice;
    uint temp = regionID % groupsPerSlice;
    regionCoord.y = temp / groupsPerRow;
    regionCoord.x = temp % groupsPerRow;
    
    // Starting block for this 8x8x4 half
    uvec3 groupStartBlock = regionCoord * uvec3(8, 8, 8) + uvec3(0, 0, halfIndex * 4);
    
    // Convert local index to 3D offset within the 8x8x4 group
    uvec3 localBlockOffset = unflattenIndex(localBlockIndex, uvec2(8, 64)); // 8x8 = 64
    
    // Global block coordinates
    uvec3 globalBlockCoord = groupStartBlock + localBlockOffset;
    
    // Convert to linear block ID
    return globalBlockCoord.x + 
           globalBlockCoord.y * params.blockGridDim.x + 
           globalBlockCoord.z * params.blockGridDim.x * params.blockGridDim.y;
}

// Project a point to screen space and return vec3(x, y, z)
vec3 projectToScreen(vec3 worldPos) {
    vec4 clipPos = pushConstants.viewProj * vec4(worldPos, 1.0);
    
    if (clipPos.w > 0.0) {
        vec3 ndc = clipPos.xyz / clipPos.w;
        return ndc;
    }
    return vec3(0.0);
}

void main() {
    uint threadID = gl_LocalInvocationID.x;
    uint meshWorkgroupID = gl_WorkGroupID.x;
    
    // Calculate which block this thread processes
    uint blockOffset = meshWorkgroupID * 32 + threadID;
    
    bool hasValidBlock = false;
    if (blockOffset < taskInput.numOccupiedBlocks) {
        // Get the local block index from dense array
        uint localBlockIndex = taskInput.denseOccupancyIndex[blockOffset];
        
        // Convert to global block ID using workgroup ID
        uint blockID = getGlobalBlockID(taskInput.baseID, localBlockIndex);
        
        // Get 3D block coordinates
        uvec3 blockCoord = uvec3(
            blockID % params.blockGridDim.x,
            (blockID / params.blockGridDim.x) % params.blockGridDim.y,
            blockID / (params.blockGridDim.x * params.blockGridDim.y)
        );
        
        // Calculate world-space bounding box of the block
        vec3 blockMin = vec3(blockCoord * params.blockDim.xyz);
        vec3 blockMax = blockMin + vec3(params.blockDim.xyz);
        
        // Generate 8 corner points of the 3D bounding box
        vec3 corners[8] = {
            vec3(blockMin.x, blockMin.y, blockMin.z),
            vec3(blockMax.x, blockMin.y, blockMin.z),
            vec3(blockMin.x, blockMax.y, blockMin.z),
            vec3(blockMax.x, blockMax.y, blockMin.z),
            vec3(blockMin.x, blockMin.y, blockMax.z),
            vec3(blockMax.x, blockMin.y, blockMax.z),
            vec3(blockMin.x, blockMax.y, blockMax.z),
            vec3(blockMax.x, blockMax.y, blockMax.z)
        };
        
        // Project all corners to screen space
        vec3 projectedCorners[8];
        vec2 screenMin = vec2(1.0);
        vec2 screenMax = vec2(-1.0);
        float minZ = 1.0;
        float maxZ = 0.0;
        bool anyValidProjection = false;
        
        for (uint i = 0; i < 8; i++) {
            vec4 clipPos = pushConstants.viewProj * vec4(corners[i], 1.0);
            if (clipPos.w > 0.0) {
                projectedCorners[i] = clipPos.xyz / clipPos.w;
                screenMin = min(screenMin, projectedCorners[i].xy);
                screenMax = max(screenMax, projectedCorners[i].xy);
                minZ = min(minZ, projectedCorners[i].z);
                maxZ = max(maxZ, projectedCorners[i].z);
                anyValidProjection = true;
            }
        }
        
        // Skip if no valid projections (block is behind camera)
        if (!anyValidProjection) {
            // Don't generate geometry for this block
            hasValidBlock = false;
        } else {
            hasValidBlock = true;
            
            // Generate a screen-space quad with conservative depth
            uint vertexBase = threadID * 4;
            uint primitiveBase = threadID * 2;
            
            // Use MORE conservative depth and expanded bounds (like Kreskowski)
            // This ensures blocks at boundaries aren't incorrectly culled
            float conservativeZ = max(0.0, minZ - 0.01); // More aggressive depth offset
            
            // Expand screen-space bounds slightly to be more conservative
            // This helps prevent missing blocks at exact boundaries
            vec2 expansion = vec2(0.002); // Small expansion in NDC space
            vec2 expandedMin = screenMin - expansion;
            vec2 expandedMax = screenMax + expansion;
            
            // Output 4 vertices for the expanded quad
            gl_MeshVerticesEXT[vertexBase + 0].gl_Position = vec4(expandedMin.x, expandedMin.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 1].gl_Position = vec4(expandedMax.x, expandedMin.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 2].gl_Position = vec4(expandedMin.x, expandedMax.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 3].gl_Position = vec4(expandedMax.x, expandedMax.y, conservativeZ, 1.0);
            
            // Output 2 triangles (6 indices)
            gl_PrimitiveTriangleIndicesEXT[primitiveBase + 0] = uvec3(vertexBase + 0, vertexBase + 1, vertexBase + 2);
            gl_PrimitiveTriangleIndicesEXT[primitiveBase + 1] = uvec3(vertexBase + 1, vertexBase + 3, vertexBase + 2);
            
            // Pass block ID to fragment shader
            outPrimitive[primitiveBase + 0].blockID = blockID;
            outPrimitive[primitiveBase + 1].blockID = blockID;
            
            // Debug first few quads
            // if (blockOffset < 3) {
            //     debugPrintfEXT("OcclusionMesh: block %d -> quad NDC[%.2f,%.2f to %.2f,%.2f] z=%.4f (conservative=%.4f)",
            //                   blockID, screenMin.x, screenMin.y, screenMax.x, screenMax.y, minZ, conservativeZ);
            // }
        }
    }
    
    // Simply count how many valid blocks we have
    // In Kreskowski's approach, the task shader already provides a compacted list
    // so ALL threads that receive a block will generate geometry
    uint validBlockCount = min(taskInput.numOccupiedBlocks - meshWorkgroupID * 32, 32u);
    
    // Only thread 0 sets the output counts (like Kreskowski)
    if (threadID == 0) {
        uint vertexCount = validBlockCount * 4;
        uint primitiveCount = validBlockCount * 2;
        
        // if (meshWorkgroupID == 0) {
        //     uint blocksInThisWG = min(taskInput.numOccupiedBlocks - meshWorkgroupID * 32, 32);
        //     debugPrintfEXT("OcclusionMesh WG 0: SetMeshOutputsEXT(%d vertices, %d primitives) for %d valid blocks (of %d total)\n",
        //                   vertexCount, primitiveCount, validBlockCount, blocksInThisWG);
        // }
        
        SetMeshOutputsEXT(vertexCount, primitiveCount);
    }
}