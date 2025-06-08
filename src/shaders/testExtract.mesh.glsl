#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_arithmetic : require // For subgroupAdd potentially
#extension GL_EXT_shader_atomic_int64 : require // Or int32 if offsets fit
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_debug_printf : enable

// --- Configurable Parameters ---
// Dimensions of the blocks processed by ONE Task/Mesh invocation
// This should match the block size used in your filtering stage
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 8 // Example: Adjust to your actual block size
#define CELLS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)

#define MAX_VERTICES 256 // Standard MC can generate more vertices than PMB-style
#define MAX_PRIMITIVES 256 // Max triangles per meshlet output

// Maximum possible output per cell in classic MC
#define MAX_VERTS_PER_CELL 12
#define MAX_PRIMS_PER_CELL 5 // Max triangles per MC case

// Workgroup size for this Task Shader
#define WORKGROUP_SIZE 32

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

struct VertexData {
    vec4 position; // Use .xyz for position
    vec4 normal;   // Use .xyz for normal
};

struct TaskPayload {
    uint globalVertexOffset;     // Offset reserved in global Vertex Buffer
    uint globalIndexOffset;      // Offset reserved in global Index Buffer
    uint globalMeshletDescOffset;// Offset for this block's descriptor
    uvec3 blockOrigin;           // Absolute origin in volume coordinates
    uint taskId;
    // blockDimensions are implicit (BLOCK_DIM_X/Y/Z)
};

taskPayloadSharedEXT TaskPayload taskPayloadIn;

// --- End Structures ---
// --- Descriptor Set Bindings (Matching C++ ExtractionPipeline::createPipelineLayout) ---
// Binding 0: UBO
layout(set = 0, binding = 0, std140) uniform PushConstants { 
    // Use PushConstants name if that's the struct in C++ UBO
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} ubo;

// Binding 1: Volume Image (Input)
layout(set = 0, binding = 1, r8ui) uniform readonly uimage3D volumeImage;

// Binding 2: Active Block Count Buffer - Not needed

// Binding 3: Compacted Block IDs - Not needed

// Binding 4: MC Triangle Table (Input)
layout(set = 0, binding = 4, std430) readonly buffer MarchingCubesTriTable_SSBO { int triTable[]; } mc;

// Binding 5: MC NumVertices Table - Not needed

// Binding 6: Vertex Buffer (Output Data)
// Define buffer structure to match C++ VertexData (assuming vec4 version)
layout(set = 0, binding = 6, std430) buffer Vertex_SSBO { VertexData vertex_data[]; } vertices;

// Binding 7: Vertex Count Buffer - Not directly used by mesh shader (Task reserves space)

// Binding 8: Index Buffer (Output Data)
layout(set = 0, binding = 8, std430) buffer Index_SSBO { uint indices[]; } indices;

// Binding 9: Index Count Buffer - Not directly used by mesh shader

// Binding 10: Meshlet Descriptor Buffer (Output Data)
layout(set = 0, binding = 10, std430) buffer MeshletDesc_SSBO { MeshletDescriptor meshletDescriptors[]; } meshletDesc;

// Binding 11: Meshlet Descriptor Count Buffer - Not directly used by mesh shader
// --- End Descriptor Set Bindings ---

layout(local_size_x = WORKGROUP_SIZE) in;
layout(max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;
layout(triangles) out;

// Output vertex attributes (normal in this case)
layout(location = 0) out PerVertexData { vec3 normal; } vertexData[];

// --- Shared Memory ---
// Shared memory for accumulating output geometry within the workgroup before final write
// Size based on the updated MAX_VERTICES/PRIMITIVES limits
shared VertexData sharedMeshletVertices[MAX_VERTICES]; // Store VertexData struct
shared uvec3 sharedMeshletIndices[MAX_PRIMITIVES];     // Stores LOCAL indices (0 to N-1)
shared uint sharedMeshletVertexCount;                  // Use standard uint
shared uint sharedMeshletPrimitiveCount;               // Use standard uint
// --- End Shared Memory ---

// Maps a Marching Cubes edge index (0-11) to its defining corner vertices and axis.
// Corner indices (0-7) follow the standard MC pattern:
//   4----5
//  /|   /|
// 7----6 |    Y
// | 0--|-1    | / Z
// |/   |/     |/
// 3----2      o----X
//
void getEdgeInfo(int edgeIndex, out ivec3 p1_offset, out ivec3 p2_offset, out int corner1_idx, out int corner2_idx, out int axis) {
    switch (edgeIndex) {
        case 0:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(1,0,0); corner1_idx=0; corner2_idx=1; axis=0; return; // Edge 0-1 (X)
        case 1:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,1,0); corner1_idx=1; corner2_idx=2; axis=1; return; // Edge 1-2 (Y)
        case 2:  p1_offset=ivec3(0,1,0); p2_offset=ivec3(1,1,0); corner1_idx=3; corner2_idx=2; axis=0; return; // Edge 3-2 (X)
        case 3:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,1,0); corner1_idx=0; corner2_idx=3; axis=1; return; // Edge 0-3 (Y)
        case 4:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(1,0,1); corner1_idx=4; corner2_idx=5; axis=0; return; // Edge 4-5 (X)
        case 5:  p1_offset=ivec3(1,0,1); p2_offset=ivec3(1,1,1); corner1_idx=5; corner2_idx=6; axis=1; return; // Edge 5-6 (Y)
        case 6:  p1_offset=ivec3(0,1,1); p2_offset=ivec3(1,1,1); corner1_idx=7; corner2_idx=6; axis=0; return; // Edge 7-6 (X)
        case 7:  p1_offset=ivec3(0,0,1); p2_offset=ivec3(0,1,1); corner1_idx=4; corner2_idx=7; axis=1; return; // Edge 4-7 (Y)
        case 8:  p1_offset=ivec3(0,0,0); p2_offset=ivec3(0,0,1); corner1_idx=0; corner2_idx=4; axis=2; return; // Edge 0-4 (Z)
        case 9:  p1_offset=ivec3(1,0,0); p2_offset=ivec3(1,0,1); corner1_idx=1; corner2_idx=5; axis=2; return; // Edge 1-5 (Z)
        case 10: p1_offset=ivec3(1,1,0); p2_offset=ivec3(1,1,1); corner1_idx=2; corner2_idx=6; axis=2; return; // Edge 2-6 (Z)
        case 11: p1_offset=ivec3(0,1,0); p2_offset=ivec3(0,1,1); corner1_idx=3; corner2_idx=7; axis=2; return; // Edge 3-7 (Z)
        default: // Should not happen
                 p1_offset=ivec3(0); p2_offset=ivec3(0); corner1_idx=-1; corner2_idx=-1; axis=-1; return;
    }
}

vec3 interpolateVertex(ivec3 p1_global, ivec3 p2_global, float val1, float val2) {

    if (abs(val1 - val2) < 1e-6f) { return vec3(p1_global); }
    float t = clamp((ubo.isovalue - val1) / (val2 - val1), 0.0f, 1.0f);
    return mix(vec3(p1_global), vec3(p2_global), t);
}

vec3 calculateNormal(ivec3 globalCellCoord, vec3 vertexPos) {
    // Use imageLoad directly, or load into shared mem first and use that
    ivec3 Npos = ivec3(round(vertexPos));
    float s1_x = float(imageLoad(volumeImage, clamp(Npos + ivec3(1,0,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s2_x = float(imageLoad(volumeImage, clamp(Npos - ivec3(1,0,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s1_y = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,1,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s2_y = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,1,0), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s1_z = float(imageLoad(volumeImage, clamp(Npos + ivec3(0,0,1), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    float s2_z = float(imageLoad(volumeImage, clamp(Npos - ivec3(0,0,1), ivec3(0), ivec3(ubo.volumeDim.xyz)-1)).r);
    vec3 grad = vec3(s2_x - s1_x, s2_y - s1_y, s2_z - s1_z);
    if (length(grad) < 1e-5f) return vec3(0, 1, 0);
    return -normalize(grad);
}


void main() {
    // Initialize shared counters
    if (gl_LocalInvocationIndex == 0) {
        sharedMeshletVertexCount = 0;
        sharedMeshletPrimitiveCount = 0;
    }
    barrier();

    uvec3 blockOrigin = taskPayloadIn.blockOrigin;

    uvec3 blockOriginFromPayload = taskPayloadIn.blockOrigin;

    for (uint linearCellIndex = gl_LocalInvocationIndex;
    linearCellIndex < CELLS_PER_BLOCK;
    linearCellIndex += WORKGROUP_SIZE)
    {
        uvec3 localCellCoord;
        localCellCoord.x = linearCellIndex % BLOCK_DIM_X;
        localCellCoord.y = (linearCellIndex / BLOCK_DIM_X) % BLOCK_DIM_Y;
        localCellCoord.z = linearCellIndex / (BLOCK_DIM_X * BLOCK_DIM_Y);

        ivec3 globalCellCoord = ivec3(blockOriginFromPayload) + ivec3(localCellCoord);

        uint cubeCase = 0;
        float cornerValuesF[8];

        for (int i = 0; i < 8; ++i) {
            ivec3 cornerOffset = ivec3((i & 1), (i & 2) >> 1, (i & 4) >> 2);
            ivec3 cornerVolCoord = globalCellCoord + cornerOffset; // Calculate absolute corner coordinate

            uint valU = 0;
            // Check if the calculated cornerVolCoord is within the volume's bounds
            if (all(greaterThanEqual(cornerVolCoord, ivec3(0))) && all(lessThan(cornerVolCoord, ivec3(ubo.volumeDim.xyz)))) {
                valU = imageLoad(volumeImage, cornerVolCoord).r; // Sample if in bounds
            } else {
                // If out of bounds, this corner is effectively "outside"
                // For MC, this typically means it's below the isovalue or a default background value
                valU = 0; // Or an appropriate background value if your volume has one
            }

            cornerValuesF[i] = float(valU);
            if (cornerValuesF[i] >= ubo.isovalue) {
                cubeCase |= (1 << i);
            }
        }

        if (cubeCase != 0 && cubeCase != 255) {
            uint cellVertexIndices[12];
            for(int k=0; k<12; ++k) cellVertexIndices[k] = 0xFFFFFFFF;

            int baseTriTableIdx = int(cubeCase * 16);
            for (int tri = 0; tri < 15; tri += 3) {
                int edgeIndices[3] = {
                mc.triTable[baseTriTableIdx + tri + 0],
                mc.triTable[baseTriTableIdx + tri + 1],
                mc.triTable[baseTriTableIdx + tri + 2]
                };
                if (edgeIndices[0] == -1) break;

                uint currentTotalVerts_approx = atomicAdd(sharedMeshletVertexCount, 0u);
                uint currentTotalPrims_approx = atomicAdd(sharedMeshletPrimitiveCount, 0u);
                uint potentialNewVerts = 0;
                for(int k=0; k<3; ++k) if(edgeIndices[k] != -1 && cellVertexIndices[edgeIndices[k]] == 0xFFFFFFFF) potentialNewVerts++;

                if (currentTotalPrims_approx >= MAX_PRIMITIVES || (currentTotalVerts_approx + potentialNewVerts) > MAX_VERTICES) {
                    break;
                }

                uvec3 triangleLocalIndicesToShared;
                for (int v_idx = 0; v_idx < 3; ++v_idx) {
                    int edgeID = edgeIndices[v_idx];
                    if (edgeID < 0 || edgeID > 11) { triangleLocalIndicesToShared[0] = 0xFFFFFFFF; break; }

                    if (cellVertexIndices[edgeID] != 0xFFFFFFFF) {
                        triangleLocalIndicesToShared[v_idx] = cellVertexIndices[edgeID];
                    } else {
                        ivec3 p1_offset, p2_offset; int c1_idx, c2_idx, axis;
                        getEdgeInfo(edgeID, p1_offset, p2_offset, c1_idx, c2_idx, axis);
                        vec3 pos = interpolateVertex(globalCellCoord + p1_offset, globalCellCoord + p2_offset, cornerValuesF[c1_idx], cornerValuesF[c2_idx]);
                        vec3 norm = calculateNormal(globalCellCoord, pos);

                        uint meshletVertexIndex = atomicAdd(sharedMeshletVertexCount, 1u);

                        if (meshletVertexIndex < MAX_VERTICES) {
                            sharedMeshletVertices[meshletVertexIndex].position = vec4(pos, 1.0);
                            sharedMeshletVertices[meshletVertexIndex].normal = vec4(norm, 0.0);
                            cellVertexIndices[edgeID] = meshletVertexIndex;
                            triangleLocalIndicesToShared[v_idx] = meshletVertexIndex;
                        } else {
                            atomicAdd(sharedMeshletVertexCount, -1u);
                            triangleLocalIndicesToShared[0] = 0xFFFFFFFF;
                            break;
                        }
                    }
                }

                if (triangleLocalIndicesToShared[0] != 0xFFFFFFFF) {
                    uint meshletPrimIndex = atomicAdd(sharedMeshletPrimitiveCount, 1u);
                    if (meshletPrimIndex < MAX_PRIMITIVES) {
                        sharedMeshletIndices[meshletPrimIndex] = triangleLocalIndicesToShared;
                    } else {
                        atomicAdd(sharedMeshletPrimitiveCount, -1u);
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    barrier();

    // --- Final Output ---
    uint finalVertexCount = min(atomicAdd(sharedMeshletVertexCount, 0u), MAX_VERTICES);
    uint finalPrimitiveCount = min(atomicAdd(sharedMeshletPrimitiveCount, 0u), MAX_PRIMITIVES);

    SetMeshOutputsEXT(finalVertexCount, finalPrimitiveCount);

    // Output vertices from shared memory to fixed function & global buffer
    uint globalVtxBase = taskPayloadIn.globalVertexOffset;
    for (uint i = gl_LocalInvocationIndex; i < finalVertexCount; i += WORKGROUP_SIZE) {
        VertexData vData = sharedMeshletVertices[i]; // Read VertexData struct

        // Output for fixed function / rasterizer
        gl_MeshVerticesEXT[i].gl_Position = vData.position; // Output vec4
        vertexData[i].normal = vData.normal.xyz;          // Output vec3 normal

        // Write to global SSBO (Binding 6)
        uint writeIdx = globalVtxBase + i;
        // Ensure write index is within reasonable bounds if possible
        // Requires knowing the total size allocated for vertex_data buffer
        vertices.vertex_data[writeIdx] = vData; // Write the whole struct

        if (length(vData.position.xyz) < 0.01) {
            debugPrintfEXT("Warning in length!");
        }
    }

    // Output primitives (indices local to meshlet) & write global indices
    uint globalIdxBase = taskPayloadIn.globalIndexOffset;
    for (uint i = gl_LocalInvocationIndex; i < finalPrimitiveCount; i += WORKGROUP_SIZE) {
        uvec3 localIndices = sharedMeshletIndices[i];
        gl_PrimitiveTriangleIndicesEXT[i] = localIndices;

        // Write to global SSBO (Binding 8)
        uint writeIdxBase = globalIdxBase + i * 3;
        indices.indices[writeIdxBase + 0] = localIndices.x + globalVtxBase;
        indices.indices[writeIdxBase + 1] = localIndices.y + globalVtxBase;
        indices.indices[writeIdxBase + 2] = localIndices.z + globalVtxBase;
    }

    // Write Meshlet Descriptor (Thread 0)
    barrier();
    if (gl_LocalInvocationIndex == 0) {
        uint descWriteIdx = taskPayloadIn.globalMeshletDescOffset;
        if (descWriteIdx < 10000000) {
            // Basic sanity check
            // Write to global SSBO (Binding 10)
            meshletDesc.meshletDescriptors[descWriteIdx].vertexOffset = globalVtxBase;
            meshletDesc.meshletDescriptors[descWriteIdx].indexOffset = globalIdxBase;
            meshletDesc.meshletDescriptors[descWriteIdx].vertexCount = finalVertexCount;
            meshletDesc.meshletDescriptors[descWriteIdx].primitiveCount = finalPrimitiveCount;
        } else {
            debugPrintfEXT("  MS[%u] WARN: Received potentially invalid descriptor offset %u. Skipping descriptor write.\n", gl_WorkGroupID.x, descWriteIdx);
        }
    }
}
