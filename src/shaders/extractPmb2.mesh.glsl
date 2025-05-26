#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

// --- Parameters ---
#define MAX_VERTICES 256
#define MAX_PRIMITIVES 256
#define WORKGROUP_SIZE 32

// CRITICAL: Use fixed-point arithmetic for perfect precision
#define FIXED_POINT_SCALE 1000000

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

struct VertexData {
    vec4 position;
    vec4 normal;
};

struct TaskPayload {
    uint globalVertexOffset;
    uint globalIndexOffset;
    uint globalMeshletDescOffset;
    uvec3 blockOrigin;
    uvec3 subBlockDim;
    uint taskId;
    uint estimatedVertices;
    uint estimatedPrimitives;
};

taskPayloadSharedEXT TaskPayload taskPayloadIn;

// --- Descriptor Set Bindings ---
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;
layout(set = 0, binding = 6, std430) buffer Vertex_SSBO { VertexData vertex_data[]; } vertices;
layout(set = 0, binding = 8, std430) buffer Index_SSBO { uint indices[]; } indexBuffer;
layout(set = 0, binding = 10, std430) buffer MeshletDesc_SSBO { MeshletDescriptor meshletDescriptors[]; } meshletDesc;

layout(local_size_x = WORKGROUP_SIZE) in;
layout(max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;
layout(triangles) out;

layout(location = 0) out PerVertexData { vec3 normal; } vertexData[];

// --- Shared Memory ---
shared VertexData sharedMeshletVertices[MAX_VERTICES];
shared uvec3 sharedMeshletIndices[MAX_PRIMITIVES];
shared uint sharedMeshletVertexCount;
shared uint sharedMeshletPrimitiveCount;

// Simple vertex deduplication
struct VertexKey {
    int x, y, z; // Fixed-point coordinates
};

shared VertexKey vertexKeys[MAX_VERTICES];
shared uint vertexKeyCount;

// --- PERFECT DETERMINISTIC FUNCTIONS ---

// Fixed-point arithmetic for perfect precision
VertexKey createVertexKey(ivec3 p1, ivec3 p2, float val1, float val2) {
    // Convert to signed for safety
    int isovalue_fp = int(ubo.isovalue * float(FIXED_POINT_SCALE));
    int val1_fp = int(val1 * float(FIXED_POINT_SCALE));
    int val2_fp = int(val2 * float(FIXED_POINT_SCALE));
    
    // Force consistent ordering
    ivec3 minP = min(p1, p2);
    ivec3 maxP = max(p1, p2);
    
    bool p1_is_min = all(equal(p1, minP));
    int minVal_fp = p1_is_min ? val1_fp : val2_fp;
    int maxVal_fp = p1_is_min ? val2_fp : val1_fp;
    
    // Fixed-point interpolation
    int denominator = maxVal_fp - minVal_fp;
    
    VertexKey key;
    if (abs(denominator) < 10) { // Threshold in fixed-point
        key.x = minP.x * FIXED_POINT_SCALE;
        key.y = minP.y * FIXED_POINT_SCALE;
        key.z = minP.z * FIXED_POINT_SCALE;
    } else {
        int numerator = isovalue_fp - minVal_fp;
        
        // Clamp t to [0, FIXED_POINT_SCALE]
        int t_fp = clamp((numerator * FIXED_POINT_SCALE) / denominator, 0, FIXED_POINT_SCALE);
        
        // Fixed-point interpolation
        key.x = minP.x * FIXED_POINT_SCALE + t_fp * (maxP.x - minP.x);
        key.y = minP.y * FIXED_POINT_SCALE + t_fp * (maxP.y - minP.y);
        key.z = minP.z * FIXED_POINT_SCALE + t_fp * (maxP.z - minP.z);
    }
    
    return key;
}

// Convert key back to position
vec3 keyToPosition(VertexKey key) {
    return vec3(float(key.x), float(key.y), float(key.z)) / float(FIXED_POINT_SCALE);
}

// Perfect volume sampling
float sampleVolume(ivec3 coord) {
    ivec3 clampedCoord = clamp(coord, ivec3(0), ivec3(ubo.volumeDim.xyz) - ivec3(1));
    return float(imageLoad(volumeImage, clampedCoord).r);
}

// Compare vertex keys for deduplication
bool keysEqual(VertexKey a, VertexKey b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Find existing vertex or create new one
uint findOrCreateVertex(VertexKey key) {
    // Linear search through existing vertices
    for (uint i = 0; i < vertexKeyCount; ++i) {
        if (keysEqual(vertexKeys[i], key)) {
            return i; // Found existing vertex
        }
    }
    
    // Create new vertex
    uint newIndex = atomicAdd(vertexKeyCount, 1u);
    if (newIndex < MAX_VERTICES) {
        vertexKeys[newIndex] = key;
        
        vec3 pos = keyToPosition(key);
        vec3 norm = normalize(vec3(1)); // Simple normal for now
        
        sharedMeshletVertices[newIndex] = VertexData(vec4(pos, 1.0), vec4(norm, 0.0));
        return newIndex;
    }
    
    return 0xFFFFFFFF; // Failed
}

vec3 calculateNormal(vec3 pos) {
    ivec3 ipos = ivec3(round(pos));
    float s1_x = sampleVolume(ipos + ivec3(1,0,0));
    float s2_x = sampleVolume(ipos - ivec3(1,0,0));
    float s1_y = sampleVolume(ipos + ivec3(0,1,0));
    float s2_y = sampleVolume(ipos - ivec3(0,1,0));
    float s1_z = sampleVolume(ipos + ivec3(0,0,1));
    float s2_z = sampleVolume(ipos - ivec3(0,0,1));
    
    vec3 grad = vec3(s2_x - s1_x, s2_y - s1_y, s2_z - s1_z);
    if (length(grad) < 1e-5f) return vec3(0, 1, 0);
    return -normalize(grad);
}

// Helper functions
uint linearIndex(uvec3 coord, uvec3 dim) {
    return coord.z * dim.x * dim.y + coord.y * dim.x + coord.x;
}

uvec3 from3DIndex(uint linear, uvec3 dim) {
    uvec3 result;
    result.z = linear / (dim.x * dim.y);
    uint remainder = linear % (dim.x * dim.y);
    result.y = remainder / dim.x;
    result.x = remainder % dim.x;
    return result;
}

void main() {
    uint threadID = gl_LocalInvocationIndex;
    uvec3 blockOrigin = taskPayloadIn.blockOrigin;
    uvec3 subBlockDim = taskPayloadIn.subBlockDim;
    
    // Initialize shared memory
    if (threadID == 0) {
        sharedMeshletVertexCount = 0;
        sharedMeshletPrimitiveCount = 0;
        vertexKeyCount = 0;
    }
    
    // Initialize vertex keys
    for (uint i = threadID; i < MAX_VERTICES; i += WORKGROUP_SIZE) {
        vertexKeys[i].x = 0;
        vertexKeys[i].y = 0;
        vertexKeys[i].z = 0;
    }
    
    barrier();

    // SIMPLE MARCHING CUBES - No PMB complexity, just perfect vertices
    uint totalCells = subBlockDim.x * subBlockDim.y * subBlockDim.z;
    
    for (uint cellIdx = threadID; cellIdx < totalCells; cellIdx += WORKGROUP_SIZE) {
        uvec3 cellCoord = from3DIndex(cellIdx, subBlockDim);
        ivec3 globalCellCoord = ivec3(blockOrigin) + ivec3(cellCoord);
        
        // Bounds check
        if (any(greaterThanEqual(globalCellCoord + ivec3(1), ivec3(ubo.volumeDim.xyz)))) {
            continue;
        }
        
        // Sample corner values
        uint cubeCase = 0;
        float cornerValues[8];
        
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerCoord = globalCellCoord + cornerOffset;
            cornerValues[i] = sampleVolume(cornerCoord);
            
            if (cornerValues[i] >= ubo.isovalue) {
                cubeCase |= (1u << i);
            }
        }
        
        if (cubeCase != 0 && cubeCase != 255) {
            // Count triangles
            uint numTris = 0;
            int baseTriTableIdx = int(cubeCase * 16);
            for (int i = 0; i < 16; i += 3) {
                if (mc.triTable[baseTriTableIdx + i] == -1) break;
                numTris++;
            }
            
            if (numTris > 0) {
                uint localTriOffset = atomicAdd(sharedMeshletPrimitiveCount, numTris);
                
                if (localTriOffset + numTris <= MAX_PRIMITIVES) {
                    // Create triangles with perfect vertex deduplication
                    for (uint tri = 0; tri < numTris; ++tri) {
                        uvec3 triangleIndices = uvec3(0xFFFFFFFF);
                        bool validTriangle = true;
                        
                        for (int v = 0; v < 3; ++v) {
                            int edgeID = mc.triTable[baseTriTableIdx + tri * 3 + v];
                            if (edgeID < 0 || edgeID >= 12) {
                                validTriangle = false;
                                break;
                            }
                            
                            // Get edge endpoints
                            ivec3 edgeP1, edgeP2;
                            float edgeVal1, edgeVal2;
                            
                            if (edgeID == 0) { edgeP1 = globalCellCoord; edgeP2 = globalCellCoord + ivec3(1,0,0); edgeVal1 = cornerValues[0]; edgeVal2 = cornerValues[1]; }
                            else if (edgeID == 1) { edgeP1 = globalCellCoord + ivec3(1,0,0); edgeP2 = globalCellCoord + ivec3(1,1,0); edgeVal1 = cornerValues[1]; edgeVal2 = cornerValues[2]; }
                            else if (edgeID == 2) { edgeP1 = globalCellCoord + ivec3(0,1,0); edgeP2 = globalCellCoord + ivec3(1,1,0); edgeVal1 = cornerValues[3]; edgeVal2 = cornerValues[2]; }
                            else if (edgeID == 3) { edgeP1 = globalCellCoord; edgeP2 = globalCellCoord + ivec3(0,1,0); edgeVal1 = cornerValues[0]; edgeVal2 = cornerValues[3]; }
                            else if (edgeID == 4) { edgeP1 = globalCellCoord + ivec3(0,0,1); edgeP2 = globalCellCoord + ivec3(1,0,1); edgeVal1 = cornerValues[4]; edgeVal2 = cornerValues[5]; }
                            else if (edgeID == 5) { edgeP1 = globalCellCoord + ivec3(1,0,1); edgeP2 = globalCellCoord + ivec3(1,1,1); edgeVal1 = cornerValues[5]; edgeVal2 = cornerValues[6]; }
                            else if (edgeID == 6) { edgeP1 = globalCellCoord + ivec3(0,1,1); edgeP2 = globalCellCoord + ivec3(1,1,1); edgeVal1 = cornerValues[7]; edgeVal2 = cornerValues[6]; }
                            else if (edgeID == 7) { edgeP1 = globalCellCoord + ivec3(0,0,1); edgeP2 = globalCellCoord + ivec3(0,1,1); edgeVal1 = cornerValues[4]; edgeVal2 = cornerValues[7]; }
                            else if (edgeID == 8) { edgeP1 = globalCellCoord; edgeP2 = globalCellCoord + ivec3(0,0,1); edgeVal1 = cornerValues[0]; edgeVal2 = cornerValues[4]; }
                            else if (edgeID == 9) { edgeP1 = globalCellCoord + ivec3(1,0,0); edgeP2 = globalCellCoord + ivec3(1,0,1); edgeVal1 = cornerValues[1]; edgeVal2 = cornerValues[5]; }
                            else if (edgeID == 10) { edgeP1 = globalCellCoord + ivec3(1,1,0); edgeP2 = globalCellCoord + ivec3(1,1,1); edgeVal1 = cornerValues[2]; edgeVal2 = cornerValues[6]; }
                            else if (edgeID == 11) { edgeP1 = globalCellCoord + ivec3(0,1,0); edgeP2 = globalCellCoord + ivec3(0,1,1); edgeVal1 = cornerValues[3]; edgeVal2 = cornerValues[7]; }
                            else { validTriangle = false; break; }
                            
                            // Create PERFECT vertex key
                            VertexKey key = createVertexKey(edgeP1, edgeP2, edgeVal1, edgeVal2);
                            
                            // Find or create vertex
                            uint vertexIndex = findOrCreateVertex(key);
                            if (vertexIndex == 0xFFFFFFFF) {
                                validTriangle = false;
                                break;
                            }
                            
                            triangleIndices[v] = vertexIndex;
                        }
                        
                        if (validTriangle) {
                            sharedMeshletIndices[localTriOffset + tri] = triangleIndices;
                        } else {
                            sharedMeshletIndices[localTriOffset + tri] = uvec3(0);
                        }
                    }
                }
            }
        }
    }
    
    barrier();
    
    // Update vertex count and calculate normals
    sharedMeshletVertexCount = vertexKeyCount;
    
    // Calculate proper normals
    for (uint i = threadID; i < vertexKeyCount; i += WORKGROUP_SIZE) {
        vec3 pos = sharedMeshletVertices[i].position.xyz;
        vec3 norm = calculateNormal(pos);
        sharedMeshletVertices[i].normal = vec4(norm, 0.0);
    }
    
    barrier();
    
    // Output
    uint finalVertexCount = min(sharedMeshletVertexCount, uint(MAX_VERTICES));
    uint finalPrimitiveCount = min(sharedMeshletPrimitiveCount, uint(MAX_PRIMITIVES));
    if (finalVertexCount == MAX_PRIMITIVES) {
        debugPrintfEXT("Max vertices reached: %u\n", finalVertexCount);
    }
    SetMeshOutputsEXT(finalVertexCount, finalPrimitiveCount);
    
    // Output vertices
    for (uint i = threadID; i < finalVertexCount; i += WORKGROUP_SIZE) {
        VertexData vData = sharedMeshletVertices[i];
        gl_MeshVerticesEXT[i].gl_Position = vData.position;
        vertexData[i].normal = vData.normal.xyz;
        
        uint writeIdx = taskPayloadIn.globalVertexOffset + i;
        if (writeIdx < vertices.vertex_data.length()) {
            vertices.vertex_data[writeIdx] = vData;
        }
    }
    
    // Output triangles
    uint globalIdxBase = taskPayloadIn.globalIndexOffset;
    for (uint i = threadID; i < finalPrimitiveCount; i += WORKGROUP_SIZE) {
        uvec3 localIndices = sharedMeshletIndices[i];
        
        if (localIndices.x < finalVertexCount && 
            localIndices.y < finalVertexCount && 
            localIndices.z < finalVertexCount) {
            
            gl_PrimitiveTriangleIndicesEXT[i] = localIndices;
            
            uint writeIdxBase = globalIdxBase + i * 3;
            if (writeIdxBase + 2 < indexBuffer.indices.length()) {
                indexBuffer.indices[writeIdxBase + 0] = localIndices.x + taskPayloadIn.globalVertexOffset;
                indexBuffer.indices[writeIdxBase + 1] = localIndices.y + taskPayloadIn.globalVertexOffset;
                indexBuffer.indices[writeIdxBase + 2] = localIndices.z + taskPayloadIn.globalVertexOffset;
            }
        } else {
            gl_PrimitiveTriangleIndicesEXT[i] = uvec3(0);
        }
    }
    
    // Write meshlet descriptor
    if (threadID == 0) {
        uint descWriteIdx = taskPayloadIn.globalMeshletDescOffset;
        if (descWriteIdx < meshletDesc.meshletDescriptors.length()) {
            meshletDesc.meshletDescriptors[descWriteIdx] = MeshletDescriptor(
                taskPayloadIn.globalVertexOffset,
                globalIdxBase,
                finalVertexCount,
                finalPrimitiveCount
            );
        }
        
        // debugPrintfEXT("FIXED-POINT DEDUP: Meshlet %u: V=%u P=%u | Deduped %u keys\n", 
        //               taskPayloadIn.taskId, finalVertexCount, finalPrimitiveCount, vertexKeyCount);
    }
}