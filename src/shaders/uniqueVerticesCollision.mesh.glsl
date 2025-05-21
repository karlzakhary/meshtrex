#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#define MAX_VERTICES 256u
#define MAX_PRIMITIVES 128u
#define WORKGROUP_SIZE 32

// --- Structures ---
// (MeshletDescriptor, VertexData, TaskPayload same as before)
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
};

taskPayloadSharedEXT TaskPayload taskPayloadIn;

// --- Bindings ---
// (UBO, volumeImage, mc.triTable, output SSBOs same as before)
layout(set = 0, binding = 0, std140) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;
layout(set = 0, binding = 6, std430) buffer Vertex_SSBO { VertexData vertex_data[]; } vertices;
layout(set = 0, binding = 8, std430) buffer Index_SSBO { uint indices[]; } indices_ssbo;
layout(set = 0, binding = 10, std430) buffer MeshletDesc_SSBO { MeshletDescriptor meshletDescriptors[]; } meshletDesc;


// --- Mesh Shader Output ---
layout(local_size_x = WORKGROUP_SIZE) in;
layout(max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;
layout(triangles) out;

layout(location = 0) out PerVertexData { vec3 normal; } vertexDataOut[];

// --- Shared Memory ---
shared VertexData sharedMeshletVertices[MAX_VERTICES];
shared uvec3 sharedMeshletIndices[MAX_PRIMITIVES];
shared uint sharedMeshletVertexCount;
shared uint sharedMeshletPrimitiveCount;

// Simple intra-meshlet vertex cache for deduplication
// Maps a discretized global coordinate to a local vertex index
// THIS IS A VERY BASIC HASH - collisions are possible and not handled robustly here.
// A production system might use a more robust hash or a small associative cache.
#define VERTEX_CACHE_SIZE 512 // Must be a power of 2 for simple modulo. Max verts is 256, so this allows some leeway.
shared uint vertexCacheIndices[VERTEX_CACHE_SIZE]; // Stores local index (0..MAX_VERTICES-1) or 0xFFFFFFFF
shared vec3 vertexCachePositions[VERTEX_CACHE_SIZE]; // Stores position to verify hash hit

// --- Helper Functions (getEdgeInfo, interpolateVertex, calculateNormal - same as before) ---
void getEdgeInfo(int edgeIndex, out ivec3 p1_offset, out ivec3 p2_offset, out int corner1_idx, out int corner2_idx, out int axis) {
    switch (edgeIndex) {
        case 0:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(1,0,0); corner1_idx=0; corner2_idx=1; axis=0; return;
        case 1:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,1,0); corner1_idx=1; corner2_idx=2; axis=1; return;
        case 2:  p1_offset=ivec3(0,1,0); p2_offset=ivec3(1,1,0); corner1_idx=3; corner2_idx=2; axis=0; return;
        case 3:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,1,0); corner1_idx=0; corner2_idx=3; axis=1; return;
        case 4:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(1,0,1); corner1_idx=4; corner2_idx=5; axis=0; return;
        case 5:  p1_offset=ivec3(1,0,1); p2_offset=ivec3(1,1,1); corner1_idx=5; corner2_idx=6; axis=1; return;
        case 6:  p1_offset=ivec3(0,1,1); p2_offset=ivec3(1,1,1); corner1_idx=7; corner2_idx=6; axis=0; return;
        case 7:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(0,1,1); corner1_idx=4; corner2_idx=7; axis=1; return;
        case 8:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,0,1); corner1_idx=0; corner2_idx=4; axis=2; return;
        case 9:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,0,1); corner1_idx=1; corner2_idx=5; axis=2; return;
        case 10: p1_offset=ivec3(1,1,0); p2_offset=ivec3(1,1,1); corner1_idx=2; corner2_idx=6; axis=2; return;
        case 11: p1_offset=ivec3(0,1,0); p2_offset=ivec3(0,1,1); corner1_idx=3; corner2_idx=7; axis=2; return;
        default: p1_offset=ivec3(0); p2_offset=ivec3(0); corner1_idx=-1; corner2_idx=-1; axis=-1; return;
    }
}

vec3 interpolateVertex(ivec3 p1_global, ivec3 p2_global, float val1, float val2) {
    if (abs(val1 - val2) < 1e-6f) { return vec3(p1_global); }
    float t = clamp((ubo.isovalue - val1) / (val2 - val1), 0.0f, 1.0f);
    return mix(vec3(p1_global), vec3(p2_global), t);
}

vec3 calculateNormal(ivec3 globalCellCoord_not_used, vec3 vertexPos_global) {
    ivec3 Npos = ivec3(round(vertexPos_global));
    ivec3 volMax = ivec3(ubo.volumeDim.xyz) - 1;
    float s1_x = float(imageLoad(volumeImage, clamp(Npos + ivec3(1,0,0), ivec3(0), volMax)).r);
    float s2_x = float(imageLoad(volumeImage, clamp(Npos - ivec3(1,0,0), ivec3(0), volMax)).r);
    float s1_y = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,1,0), ivec3(0), volMax)).r);
    float s2_y = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,1,0), ivec3(0), volMax)).r);
    float s1_z = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,0,1), ivec3(0), volMax)).r);
    float s2_z = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,0,1), ivec3(0), volMax)).r);
    vec3 grad = vec3(s2_x - s1_x, s2_y - s1_y, s2_z - s1_z);
    if (length(grad) < 1e-5f) return vec3(0, 1, 0);
    return -normalize(grad);
}

uint spatialHash(vec3 pos) {
    // Simple spatial hash. Scale and convert to int, then combine.
    // Adjust scale factor based on typical coordinate range and desired bucket distribution.
    const float scale = 1000.0f; // Example scale
    ivec3 ipos = ivec3(pos * scale);
    // Prime numbers for hashing components
    return uint(ipos.x * 73856093 ^ ipos.y * 19349663 ^ ipos.z * 83492791) & (VERTEX_CACHE_SIZE - 1);
}

void main() {
    if (gl_LocalInvocationIndex == 0) {
        sharedMeshletVertexCount = 0;
        sharedMeshletPrimitiveCount = 0;
        for (int i = 0; i < VERTEX_CACHE_SIZE; ++i) {
            vertexCacheIndices[i] = 0xFFFFFFFF; // Mark cache slots as empty
        }
    }
    barrier();

    uvec3 currentSubBlockDim = taskPayloadIn.subBlockDim;
    uint cellsInCurrentSubBlock = currentSubBlockDim.x * currentSubBlockDim.y * currentSubBlockDim.z;
    uvec3 subBlockBaseOrigin_global = taskPayloadIn.blockOrigin;

    for (uint linearCellIdx = gl_LocalInvocationIndex; linearCellIdx < cellsInCurrentSubBlock; linearCellIdx += WORKGROUP_SIZE) {
        uvec3 localCellCoord;
        localCellCoord.x = linearCellIdx % currentSubBlockDim.x;
        localCellCoord.y = (linearCellIdx / currentSubBlockDim.x) % currentSubBlockDim.y;
        localCellCoord.z = linearCellIdx / (currentSubBlockDim.x * currentSubBlockDim.y);

        ivec3 globalCellOrigin = ivec3(subBlockBaseOrigin_global) + ivec3(localCellCoord);

        float cornerValuesF[8];
        uint cubeCase = 0;
        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellOrigin + cornerOffset;
            uint valU = 0;
            if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                valU = imageLoad(volumeImage, cornerVolCoord).r;
            }
            cornerValuesF[i] = float(valU);
            if (cornerValuesF[i] >= ubo.isovalue) cubeCase |= (1 << i);
        }

        if (cubeCase == 0 || cubeCase == 255) continue;

        // This array is for deduplicating vertices generated *for the current cell only* by this thread.
        uint perCellEdgeVertexIndices[12];
        for(int k=0; k<12; ++k) perCellEdgeVertexIndices[k] = 0xFFFFFFFF;

        int baseTriTableIdx = int(cubeCase * 16); // MAX_TRI_INDICES from task shader
        for (int tri_idx_in_cell = 0; tri_idx_in_cell < 5; ++tri_idx_in_cell) { // Max 5 triangles
                                                                                int mcEdge0 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 0];
                                                                                if (mcEdge0 == -1) break;

                                                                                // Check if we can add more primitives or vertices (approximate for vertices)
                                                                                if (sharedMeshletPrimitiveCount >= MAX_PRIMITIVES || sharedMeshletVertexCount >= MAX_VERTICES - 3) {
                                                                                    // -3 for safety for new triangle
                                                                                    // debugPrintfEXT("MS WARN: Cell (%u,%u,%u) Task %u. At prim/vert limit (%u/%u). Skipping remaining tris for cell.\n", localCellCoord.x, localCellCoord.y, localCellCoord.z, taskPayloadIn.taskId, sharedMeshletPrimitiveCount, sharedMeshletVertexCount);
                                                                                    continue; // Skip rest of triangles for this cell for this thread
                                                                                }

                                                                                int mcEdge1 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 1];
                                                                                int mcEdge2 = mc.triTable[baseTriTableIdx + tri_idx_in_cell * 3 + 2];
                                                                                int mc_edge_ids_for_triangle[3] = {mcEdge0, mcEdge1, mcEdge2};

                                                                                uvec3 currentTriangleLocalIndices;
                                                                                bool triangleValid = true;

                                                                                for (int v_idx = 0; v_idx < 3; ++v_idx) {
                                                                                    int edgeID = mc_edge_ids_for_triangle[v_idx];
                                                                                    uint localVertexIndex;

                                                                                    // 1. Check per-cell cache (same thread, same cell, multiple tris using same edge)
                                                                                    if (perCellEdgeVertexIndices[edgeID] != 0xFFFFFFFF) {
                                                                                        localVertexIndex = perCellEdgeVertexIndices[edgeID];
                                                                                    } else {
                                                                                        // 2. Vertex not in per-cell cache, generate it and try to add to meshlet shared cache
                                                                                        ivec3 p1_offset, p2_offset; int c1_idx, c2_idx, mc_axis;
                                                                                        getEdgeInfo(edgeID, p1_offset, p2_offset, c1_idx, c2_idx, mc_axis);
                                                                                        vec3 vertPosGlobal = interpolateVertex(globalCellOrigin + p1_offset, globalCellOrigin + p2_offset, cornerValuesF[c1_idx], cornerValuesF[c2_idx]);

                                                                                        // 2a. Check shared vertex cache (simple hash based)
                                                                                        uint hashSlot = spatialHash(vertPosGlobal);
                                                                                        bool foundInSharedCache = false;

                                                                                        // Simple atomic compare-and-swap-like logic for cache slot
                                                                                        // This is a very basic approach; robust cache needs careful atomics for probing sequence on collision
                                                                                        if (vertexCacheIndices[hashSlot] != 0xFFFFFFFF) {
                                                                                            // Slot is used, check if it's our vertex (compare positions)
                                                                                            // THIS COMPARISON IS CRITICAL AND PRONE TO PRECISION ISSUES
                                                                                            if (distance(vertexCachePositions[hashSlot], vertPosGlobal) < 1e-5f) {
                                                                                                localVertexIndex = vertexCacheIndices[hashSlot];
                                                                                                foundInSharedCache = true;
                                                                                            } else {
                                                                                                // Hash collision! For this simple cache, we might just generate a new vertex.
                                                                                                // A more robust cache would probe or have better collision handling.
                                                                                                // For now, let's indicate a debug message if a collision happens and we generate new.
                                                                                                debugPrintfEXT("MS WARN: Hash collision at slot %u for Task %u, Cell (%u,%u,%u), edge %d. OldPos (%f,%f,%f), NewPos (%f,%f,%f)\n", hashSlot, taskPayloadIn.taskId, localCellCoord.x, localCellCoord.y, localCellCoord.z, edgeID, vertexCachePositions[hashSlot].x, vertexCachePositions[hashSlot].y, vertexCachePositions[hashSlot].z, vertPosGlobal.x, vertPosGlobal.y, vertPosGlobal.z);
                                                                                            }
                                                                                        }

                                                                                        if (foundInSharedCache) {
                                                                                            perCellEdgeVertexIndices[edgeID] = localVertexIndex; // Cache for this cell too
                                                                                        } else {
                                                                                            // Not found in shared cache by simple hash or collision, generate new
                                                                                            if (sharedMeshletVertexCount >= MAX_VERTICES) {
                                                                                                triangleValid = false; break; // Cannot add new vertex
                                                                                            }
                                                                                            vec3 vertNormGlobal = calculateNormal(globalCellOrigin, vertPosGlobal);
                                                                                            uint newIndex = atomicAdd(sharedMeshletVertexCount, 1u);
                                                                                            if (newIndex < MAX_VERTICES) {
                                                                                                sharedMeshletVertices[newIndex].position = vec4(vertPosGlobal, 1.0);
                                                                                                sharedMeshletVertices[newIndex].normal = vec4(vertNormGlobal, 0.0);
                                                                                                localVertexIndex = newIndex;
                                                                                                perCellEdgeVertexIndices[edgeID] = newIndex;

                                                                                                // Try to add to shared cache (overwrite on collision for this simple version)
                                                                                                // More robust: atomic compare_exchange to claim the slot
                                                                                                vertexCacheIndices[hashSlot] = newIndex; // Could be an atomic write if contention is high
                                                                                                vertexCachePositions[hashSlot] = vertPosGlobal; // Non-atomic, assumes index write makes it valid
                                                                                            } else {
                                                                                                atomicAdd(sharedMeshletVertexCount, -1u); // Revert
                                                                                                triangleValid = false; break;
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    currentTriangleLocalIndices[v_idx] = localVertexIndex;
                                                                                } // end for each vertex of triangle

                                                                                if (triangleValid) {
                                                                                    if (currentTriangleLocalIndices.x == currentTriangleLocalIndices.y ||
                                                                                    currentTriangleLocalIndices.x == currentTriangleLocalIndices.z ||
                                                                                    currentTriangleLocalIndices.y == currentTriangleLocalIndices.z) {
                                                                                        // debugPrintfEXT("MS WARN: Degenerate tri Task %u Cell (%u,%u,%u): %u,%u,%u\n", taskPayloadIn.taskId, localCellCoord.x,localCellCoord.y,localCellCoord.z, currentTriangleLocalIndices.x,currentTriangleLocalIndices.y,currentTriangleLocalIndices.z);
                                                                                        continue; // Skip degenerate
                                                                                    }
                                                                                    uint primIdx = atomicAdd(sharedMeshletPrimitiveCount, 1u);
                                                                                    if (primIdx < MAX_PRIMITIVES) {
                                                                                        sharedMeshletIndices[primIdx] = currentTriangleLocalIndices;
                                                                                    } else {
                                                                                        atomicAdd(sharedMeshletPrimitiveCount, -1u); // Revert
                                                                                        continue; // No more space for prims, skip rest of this cell's tris
                                                                                    }
                                                                                } else {
                                                                                    // debugPrintfEXT("MS INFO: Invalid triangle skipped Task %u Cell (%u,%u,%u)\n", taskPayloadIn.taskId, localCellCoord.x,localCellCoord.y,localCellCoord.z);
                                                                                }
        } // end triangle loop for cell
    } // end cell loop for this thread

    // --- Final Output Stage --- (same as your previous version)
    barrier();
    uint finalVertexCount = min(sharedMeshletVertexCount, MAX_VERTICES);
    uint finalPrimitiveCount = min(sharedMeshletPrimitiveCount, MAX_PRIMITIVES);

    SetMeshOutputsEXT(finalVertexCount, finalPrimitiveCount);

    uint globalVtxBase = taskPayloadIn.globalVertexOffset;
    for (uint i = gl_LocalInvocationIndex; i < finalVertexCount; i += WORKGROUP_SIZE) {
        VertexData vData = sharedMeshletVertices[i];
        gl_MeshVerticesEXT[i].gl_Position = vData.position;
        vertexDataOut[i].normal = vData.normal.xyz;
        vertices.vertex_data[globalVtxBase + i] = vData;
    }

    uint globalIdxBase = taskPayloadIn.globalIndexOffset;
    for (uint i = gl_LocalInvocationIndex; i < finalPrimitiveCount; i += WORKGROUP_SIZE) {
        uvec3 localIndicesForPrim = sharedMeshletIndices[i];
        gl_PrimitiveTriangleIndicesEXT[i] = localIndicesForPrim;
        indices_ssbo.indices[globalIdxBase + i * 3 + 0] = localIndicesForPrim.x + globalVtxBase;
        indices_ssbo.indices[globalIdxBase + i * 3 + 1] = localIndicesForPrim.y + globalVtxBase;
        indices_ssbo.indices[globalIdxBase + i * 3 + 2] = localIndicesForPrim.z + globalVtxBase;
    }

    barrier();
    if (gl_LocalInvocationIndex == 0) {
        if (finalVertexCount > 0 || finalPrimitiveCount > 0) {
            uint descWriteIdx = taskPayloadIn.globalMeshletDescOffset;
            if(descWriteIdx < 10000000) {
                // Basic sanity, use actual buffer size if available
                meshletDesc.meshletDescriptors[descWriteIdx].vertexOffset = globalVtxBase;
                meshletDesc.meshletDescriptors[descWriteIdx].indexOffset = globalIdxBase;
                meshletDesc.meshletDescriptors[descWriteIdx].vertexCount = finalVertexCount;
                meshletDesc.meshletDescriptors[descWriteIdx].primitiveCount = finalPrimitiveCount;
            } else {
                debugPrintfEXT("MS WARN: Task %u, invalid descWriteIdx %u\n", taskPayloadIn.taskId, descWriteIdx);
            }
        }
    }
}