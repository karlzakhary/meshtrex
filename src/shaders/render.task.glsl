#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_debug_printf: require

// These must match the mesh shader's limits
#define MAX_VERTS_PER_SUB_MESHLET 64
#define MAX_PRIMS_PER_SUB_MESHLET 126

// The maximum number of smaller meshlets a single large one can be split into.
#define MAX_SUB_MESHLETS 8

// --- Input Buffers ---
struct VertexData {
    vec4 position;
    vec4 normal;
};

layout(binding = 0) readonly buffer VertexBuffer {
    VertexData vertices[];
} vertexBuffer;

layout(binding = 1) readonly buffer IndexBuffer {
    uint indices[];
} indexBuffer;

struct MeshletDescriptor {
   uint vertexOffset;
   uint indexOffset;
   uint vertexCount;
   uint primitiveCount;
};

layout(binding = 2) readonly buffer MeshletDescriptorBuffer {
    MeshletDescriptor meshlets[];
} meshletDescriptorBuffer;

// --- Task Shader Output (Payload) ---
struct SubMeshlet {
    uint originalVertexIndices[MAX_VERTS_PER_SUB_MESHLET];
    uint firstTriangleIndex;
    uint vertexCount;
    uint primitiveCount;
};

struct TaskData {
    uint largeMeshletID;
    SubMeshlet subMeshlets[MAX_SUB_MESHLETS];
};

taskPayloadSharedEXT TaskData payload;

layout(local_size_x = 32) in;

void main()
{
    // Each workgroup processes one large input meshlet.
    uint largeMeshletID = gl_WorkGroupID.x;
    MeshletDescriptor largeMeshlet = meshletDescriptorBuffer.meshlets[largeMeshletID];

    // --- Trivial Case: The meshlet is already small enough ---
    if (largeMeshlet.vertexCount <= MAX_VERTS_PER_SUB_MESHLET && 
        largeMeshlet.primitiveCount <= MAX_PRIMS_PER_SUB_MESHLET) {
        if (gl_LocalInvocationID.x == 0) {
            payload.largeMeshletID = largeMeshletID;
            payload.subMeshlets[0].vertexCount = largeMeshlet.vertexCount;
            payload.subMeshlets[0].primitiveCount = largeMeshlet.primitiveCount;
            payload.subMeshlets[0].firstTriangleIndex = 0;

            for (uint i = 0; i < largeMeshlet.vertexCount; ++i) {
                payload.subMeshlets[0].originalVertexIndices[i] = largeMeshlet.vertexOffset + i;
            }
            EmitMeshTasksEXT(1, 1, 1);
        }
        return;
    }

    // --- Complex Case: Optimized splitting with vertex reuse tracking ---
    if (gl_LocalInvocationID.x == 0) {
        payload.largeMeshletID = largeMeshletID;
        
        // Create a mapping from local vertex index to global vertex index
        uint localToGlobal[256];
        for (uint i = 0; i < min(largeMeshlet.vertexCount, 256); ++i) {
            localToGlobal[i] = largeMeshlet.vertexOffset + i;
        }
        
        uint subMeshletCount = 0;
        uint triangleCursor = 0;
        
        // Track which triangles have been used
        uint triangleUsed[8]; // Support up to 256 triangles
        for(int i = 0; i < 8; ++i) triangleUsed[i] = 0;

        while (triangleCursor < largeMeshlet.primitiveCount && subMeshletCount < MAX_SUB_MESHLETS) {
            SubMeshlet currentSub;
            currentSub.vertexCount = 0;
            currentSub.primitiveCount = 0;
            currentSub.firstTriangleIndex = triangleCursor;

            // Track vertices used in THIS sub-meshlet
            uint vertexUsedInSub[8]; // Bitmask for up to 256 vertices
            for(int i = 0; i < 8; ++i) vertexUsedInSub[i] = 0;
            
            // Map from local vertex index to sub-meshlet vertex index
            uint localToSubIndex[256];
            for(int i = 0; i < 256; ++i) localToSubIndex[i] = 0xFFFFFFFF;
            
            // First pass: Try to add triangles that share vertices with already added triangles
            bool addedAnyTriangle = true;
            while (addedAnyTriangle && currentSub.primitiveCount < MAX_PRIMS_PER_SUB_MESHLET) {
                addedAnyTriangle = false;
                
                for (uint i = 0; i < largeMeshlet.primitiveCount; ++i) {
                    // Skip if triangle already used or we're at primitive limit
                    if ((triangleUsed[i / 32] & (1 << (i % 32))) != 0) continue;
                    if (currentSub.primitiveCount >= MAX_PRIMS_PER_SUB_MESHLET) break;
                    
                    uint triIndexOffset = largeMeshlet.indexOffset + (i * 3);
                    uint v_indices[3];
                    uint v_local[3];
                    bool validTriangle = true;
                    
                    // Get vertex indices and validate
                    for (int v = 0; v < 3; ++v) {
                        v_indices[v] = indexBuffer.indices[triIndexOffset + v];
                        if (v_indices[v] < largeMeshlet.vertexOffset || 
                            v_indices[v] >= largeMeshlet.vertexOffset + largeMeshlet.vertexCount) {
                            validTriangle = false;
                            break;
                        }
                        v_local[v] = v_indices[v] - largeMeshlet.vertexOffset;
                    }
                    
                    if (!validTriangle) continue;
                    
                    // Count new vertices this triangle would add
                    uint newVertCount = 0;
                    for (int v = 0; v < 3; ++v) {
                        if ((vertexUsedInSub[v_local[v] / 32] & (1 << (v_local[v] % 32))) == 0) {
                            newVertCount++;
                        }
                    }
                    
                    // Check if we can fit this triangle
                    if (currentSub.vertexCount + newVertCount > MAX_VERTS_PER_SUB_MESHLET) continue;
                    
                    // For the first triangle, always add it
                    // For subsequent triangles, prefer ones that share vertices
                    bool shouldAdd = (currentSub.primitiveCount == 0) || (newVertCount < 3);
                    
                    if (shouldAdd) {
                        // Add the triangle
                        currentSub.primitiveCount++;
                        triangleUsed[i / 32] |= (1 << (i % 32));
                        addedAnyTriangle = true;
                        
                        // Add new vertices to the sub-meshlet
                        for (int v = 0; v < 3; ++v) {
                            if ((vertexUsedInSub[v_local[v] / 32] & (1 << (v_local[v] % 32))) == 0) {
                                vertexUsedInSub[v_local[v] / 32] |= (1 << (v_local[v] % 32));
                                localToSubIndex[v_local[v]] = currentSub.vertexCount;
                                currentSub.originalVertexIndices[currentSub.vertexCount] = v_indices[v];
                                currentSub.vertexCount++;
                            }
                        }
                    }
                }
            }
            
            // Second pass: If we have room, greedily add any remaining triangles
            if (currentSub.primitiveCount < MAX_PRIMS_PER_SUB_MESHLET) {
                for (uint i = 0; i < largeMeshlet.primitiveCount; ++i) {
                    if ((triangleUsed[i / 32] & (1 << (i % 32))) != 0) continue;
                    if (currentSub.primitiveCount >= MAX_PRIMS_PER_SUB_MESHLET) break;
                    
                    uint triIndexOffset = largeMeshlet.indexOffset + (i * 3);
                    uint v_indices[3];
                    uint v_local[3];
                    bool validTriangle = true;
                    
                    for (int v = 0; v < 3; ++v) {
                        v_indices[v] = indexBuffer.indices[triIndexOffset + v];
                        if (v_indices[v] < largeMeshlet.vertexOffset || 
                            v_indices[v] >= largeMeshlet.vertexOffset + largeMeshlet.vertexCount) {
                            validTriangle = false;
                            break;
                        }
                        v_local[v] = v_indices[v] - largeMeshlet.vertexOffset;
                    }
                    
                    if (!validTriangle) continue;
                    
                    uint newVertCount = 0;
                    for (int v = 0; v < 3; ++v) {
                        if ((vertexUsedInSub[v_local[v] / 32] & (1 << (v_local[v] % 32))) == 0) {
                            newVertCount++;
                        }
                    }
                    
                    if (currentSub.vertexCount + newVertCount <= MAX_VERTS_PER_SUB_MESHLET) {
                        currentSub.primitiveCount++;
                        triangleUsed[i / 32] |= (1 << (i % 32));
                        
                        for (int v = 0; v < 3; ++v) {
                            if ((vertexUsedInSub[v_local[v] / 32] & (1 << (v_local[v] % 32))) == 0) {
                                vertexUsedInSub[v_local[v] / 32] |= (1 << (v_local[v] % 32));
                                localToSubIndex[v_local[v]] = currentSub.vertexCount;
                                currentSub.originalVertexIndices[currentSub.vertexCount] = v_indices[v];
                                currentSub.vertexCount++;
                            }
                        }
                    }
                }
            }
            
            // Finalize sub-meshlet
            if (currentSub.primitiveCount > 0) {
                // Update firstTriangleIndex to be the actual first triangle we used
                uint actualFirstTri = 0xFFFFFFFF;
                for (uint i = 0; i < largeMeshlet.primitiveCount; ++i) {
                    if ((triangleUsed[i / 32] & (1 << (i % 32))) != 0) {
                        actualFirstTri = min(actualFirstTri, i);
                    }
                }
                currentSub.firstTriangleIndex = actualFirstTri;
                
                payload.subMeshlets[subMeshletCount] = currentSub;
                subMeshletCount++;
            }
            
            // Find next unprocessed triangle
            while (triangleCursor < largeMeshlet.primitiveCount && 
                   (triangleUsed[triangleCursor / 32] & (1 << (triangleCursor % 32))) != 0) {
                triangleCursor++;
            }
        }
        
        if (subMeshletCount > 1) {
            debugPrintfEXT("TaskID: %u, Original Verts: %u, Prims: %u -> Split into %u sub-meshlets", 
                gl_WorkGroupID.x, largeMeshlet.vertexCount, largeMeshlet.primitiveCount, subMeshletCount);
            for(uint i = 0; i < subMeshletCount; ++i) {
                debugPrintfEXT("  Sub[%u]: firstTri: %u, prims: %u, verts: %u", 
                    i, payload.subMeshlets[i].firstTriangleIndex, 
                    payload.subMeshlets[i].primitiveCount, 
                    payload.subMeshlets[i].vertexCount);
            }
        }
        
        EmitMeshTasksEXT(subMeshletCount, 1, 1);
    }
}