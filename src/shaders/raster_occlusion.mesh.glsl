#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_debug_printf : enable

// Maximum vertices and primitives per mesh shader
#define WORKGROUP_SIZE 32
#define MAX_VERTICES 128  // 32 blocks * 4 vertices per quad
#define MAX_PRIMITIVES 64 // 32 blocks * 2 triangles per quad

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
    uint numOccupiedBlocks;   // Number of occupied blocks in this 8x8x4 group
    uint8_t denseOccupancyIndex[256]; // Dense array of local block indices
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
    // Reconstruct which 8x8x4 group this is
    uint groupsPerRow = (params.blockGridDim.x + 7) / 8;
    uint groupsPerSlice = groupsPerRow * ((params.blockGridDim.y + 7) / 8);
    
    uint group888ID = workgroupID / 2;
    uint halfIndex = workgroupID % 2;
    
    // Unpack 8x8x8 group coordinates
    uvec3 group888Coord;
    group888Coord.z = group888ID / groupsPerSlice;
    uint temp = group888ID % groupsPerSlice;
    group888Coord.y = temp / groupsPerRow;
    group888Coord.x = temp % groupsPerRow;
    
    // Starting block for this 8x8x4 half
    uvec3 groupStartBlock = group888Coord * 8;
    groupStartBlock.z += halfIndex * 4;
    
    // Convert local index to 3D offset
    uvec3 localBlockOffset = unflattenIndex(localBlockIndex, uvec2(8, 64));
    
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
    
    // Debug projection for center of volume (should be visible)
    if (worldPos.x > 31.0 && worldPos.x < 33.0 && 
        worldPos.y > 31.0 && worldPos.y < 33.0 && 
        worldPos.z > 31.0 && worldPos.z < 33.0) {
        debugPrintfEXT("Projecting center: (%.1f,%.1f,%.1f) -> clip(%.2f,%.2f,%.2f,%.2f)",
                      worldPos.x, worldPos.y, worldPos.z,
                      clipPos.x, clipPos.y, clipPos.z, clipPos.w);
    }
    
    if (clipPos.w > 0.0) {
        vec3 ndc = clipPos.xyz / clipPos.w;
        return ndc;
    }
    return vec3(0.0);
}
shared uint validBlockCount;

void main() {
    uint threadID = gl_LocalInvocationID.x;
    uint meshWorkgroupID = gl_WorkGroupID.x;
    
    // Shared counter for valid blocks
    if (threadID == 0) {
        validBlockCount = 0;
    }
    barrier();
    
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
        bool anyValidProjection = false;
        
        for (uint i = 0; i < 8; i++) {
            vec4 clipPos = pushConstants.viewProj * vec4(corners[i], 1.0);
            if (clipPos.w > 0.0) {
                projectedCorners[i] = clipPos.xyz / clipPos.w;
                screenMin = min(screenMin, projectedCorners[i].xy);
                screenMax = max(screenMax, projectedCorners[i].xy);
                minZ = min(minZ, projectedCorners[i].z);
                anyValidProjection = true;
            }
        }
        
        // Skip if no valid projections (block is behind camera)
        if (!anyValidProjection) {
            // Don't generate geometry for this block
            hasValidBlock = false;
        } else {
            hasValidBlock = true;
            atomicAdd(validBlockCount, 1);
            
            // Generate a screen-space quad with conservative depth
            uint vertexBase = threadID * 4;
            uint primitiveBase = threadID * 2;
            
            // Use conservative depth - slightly in front of the nearest corner
            // This ensures the quad will be tested against existing geometry
            float conservativeZ = max(0.0, minZ - 0.001);
            
            // Output 4 vertices for the quad
            gl_MeshVerticesEXT[vertexBase + 0].gl_Position = vec4(screenMin.x, screenMin.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 1].gl_Position = vec4(screenMax.x, screenMin.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 2].gl_Position = vec4(screenMin.x, screenMax.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 3].gl_Position = vec4(screenMax.x, screenMax.y, conservativeZ, 1.0);
            
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
    
    // Synchronize threads
    barrier();
    
    // Last thread sets the output counts based on valid blocks
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