#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

#define WORKGROUP_SIZE 32
#define MAX_VERTICES 128 // 32 threads * 4 vertices
#define MAX_PRIMITIVES 64  // 32 threads * 2 triangles

layout(local_size_x = WORKGROUP_SIZE) in;
layout(triangles, max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;

// UBO bindings
layout(set = 0, binding = 0) uniform ViewUniforms {
    mat4 viewProj;
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} view;

taskPayloadSharedEXT struct Task {
    uint baseID;
    uint numOccupiedBlocks;
    uint8_t denseOccupancyIndex[256];
} taskInput;

perprimitiveEXT layout(location = 0) out Interpolants {
    flat uint blockID;
} outPrimitive[];

uvec3 unflattenIndex(uint index, uvec2 extents) {
    uvec3 result;
    result.z = index / extents.y;
    index -= result.z * extents.y;
    result.y = index / extents.x;
    result.x = index % extents.x;
    return result;
}

uint getGlobalBlockID(uint taskWorkgroupID, uint localBlockIndex) {
    uvec3 groupGridDim = (view.blockGridDim.xyz + uvec3(7, 7, 3)) / uvec3(8, 8, 4);
    uvec3 groupCoord = unflattenIndex(taskWorkgroupID, uvec2(groupGridDim.x, groupGridDim.x * groupGridDim.y));
    uvec3 groupStartBlock = groupCoord * uvec3(8, 8, 4);
    uvec3 localBlockOffset = unflattenIndex(localBlockIndex, uvec2(8, 64));
    uvec3 globalBlockCoord = groupStartBlock + localBlockOffset;
    return globalBlockCoord.x + globalBlockCoord.y * view.blockGridDim.x + globalBlockCoord.z * view.blockGridDim.x * view.blockGridDim.y;
}

void main() {
    uint threadID = gl_LocalInvocationID.x;
    uint meshWorkgroupID = gl_WorkGroupID.x;
    
    uint blockOffsetInDenseList = meshWorkgroupID * WORKGROUP_SIZE + threadID;
    
    uint numActiveThreads = min(taskInput.numOccupiedBlocks - meshWorkgroupID * WORKGROUP_SIZE, WORKGROUP_SIZE);

    if (threadID < numActiveThreads) {
        uint localBlockIndex = uint(taskInput.denseOccupancyIndex[blockOffsetInDenseList]);
        uint globalBlockID = getGlobalBlockID(taskInput.baseID, localBlockIndex);
        
        uvec3 blockCoord = uvec3(
            globalBlockID % view.blockGridDim.x,
            (globalBlockID / view.blockGridDim.x) % view.blockGridDim.y,
            globalBlockID / (view.blockGridDim.x * view.blockGridDim.y)
        );
        
        vec3 blockMin = vec3(blockCoord * view.blockDim.xyz);
        vec3 blockMax = blockMin + vec3(view.blockDim.xyz);
        
        vec3 corners[8] = {
            vec3(blockMin.x, blockMin.y, blockMin.z), vec3(blockMax.x, blockMin.y, blockMin.z),
            vec3(blockMin.x, blockMax.y, blockMin.z), vec3(blockMax.x, blockMax.y, blockMin.z),
            vec3(blockMin.x, blockMin.y, blockMax.z), vec3(blockMax.x, blockMin.y, blockMax.z),
            vec3(blockMin.x, blockMax.y, blockMax.z), vec3(blockMax.x, blockMax.y, blockMax.z)
        };
        
        vec2 screenMin = vec2(2.0);
        vec2 screenMax = vec2(-2.0);
        float minZ = 1.0;
        
        for (uint i = 0; i < 8; i++) {
            vec4 clipPos = view.viewProj * vec4(corners[i], 1.0);
            if (clipPos.w > 0.0) {
                vec3 ndc = clipPos.xyz / clipPos.w;
                screenMin = min(screenMin, ndc.xy);
                screenMax = max(screenMax, ndc.xy);
                minZ = min(minZ, ndc.z);
            }
        }
        
        if (screenMin.x <= screenMax.x) { // Check if block is on screen
            uint vertexBase = threadID * 4;
            uint primitiveBase = threadID * 2;
            
            float conservativeZ = minZ - 0.001;
            
            gl_MeshVerticesEXT[vertexBase + 0].gl_Position = vec4(screenMin.x, screenMin.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 1].gl_Position = vec4(screenMax.x, screenMin.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 2].gl_Position = vec4(screenMin.x, screenMax.y, conservativeZ, 1.0);
            gl_MeshVerticesEXT[vertexBase + 3].gl_Position = vec4(screenMax.x, screenMax.y, conservativeZ, 1.0);
            
            gl_PrimitiveTriangleIndicesEXT[primitiveBase + 0] = uvec3(vertexBase + 0, vertexBase + 1, vertexBase + 2);
            gl_PrimitiveTriangleIndicesEXT[primitiveBase + 1] = uvec3(vertexBase + 1, vertexBase + 3, vertexBase + 2);
            
            outPrimitive[primitiveBase + 0].blockID = globalBlockID;
            outPrimitive[primitiveBase + 1].blockID = globalBlockID;
        }
    }
    
    if (threadID == 0) {
        SetMeshOutputsEXT(numActiveThreads * 4, numActiveThreads * 2);
    }
}