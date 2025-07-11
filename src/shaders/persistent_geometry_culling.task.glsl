#version 460 core
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_arithmetic: require

// Task shader for culling and rendering persistent geometry
// Reads from global buffers populated by extraction pass

#define WORKGROUP_SIZE 32
#define MAX_MESHLETS_PER_WORKGROUP 8

// --- Structures ---
struct MeshletDescriptor {
    uint vertexOffset;
    uint indexOffset;
    uint vertexCount;
    uint primitiveCount;
};

struct CameraData {
    mat4 viewMatrix;
    mat4 projMatrix;
    mat4 viewProjMatrix;
    vec4 frustumPlanes[6];
    vec3 cameraPosition;
    float padding;
};

taskPayloadSharedEXT struct TaskPayload {
    uint meshletIndices[MAX_MESHLETS_PER_WORKGROUP];
    uint meshletCount;
} payload;

// --- Push Constants ---
layout(push_constant) uniform PushConstants {
    uint totalMeshlets;
    uint meshletsPerWorkgroup;
    float cullDistance;
    uint enableFrustumCulling;
    uint enableBackfaceCulling;
    uint enableOcclusionCulling;
    uint padding[2];
} pc;

// --- Descriptor Bindings ---
layout(set = 0, binding = 0, std140) uniform CameraBuffer {
    CameraData camera;
};

layout(set = 0, binding = 1, std430) readonly buffer MeshletDescriptorBuffer {
    MeshletDescriptor meshlets[];
};

layout(set = 0, binding = 2, std430) readonly buffer VertexPositions {
    vec4 positions[];
} vertexBuffer;

// Shared memory for culling results
shared uint sh_passedCulling[WORKGROUP_SIZE];
shared uint sh_meshletIndices[WORKGROUP_SIZE];
shared uint sh_totalPassed;

// --- Helper Functions ---
bool isMeshletInFrustum(uint meshletIndex) {
    MeshletDescriptor meshlet = meshlets[meshletIndex];
    
    // Calculate meshlet bounding sphere from vertices
    vec3 center = vec3(0.0);
    float radius = 0.0;
    
    // Sample a few vertices to estimate bounds (optimization)
    const uint sampleCount = min(8u, meshlet.vertexCount);
    for (uint i = 0; i < sampleCount; i++) {
        uint vIndex = meshlet.vertexOffset + (i * meshlet.vertexCount / sampleCount);
        vec3 pos = vertexBuffer.positions[vIndex].xyz;
        center += pos;
    }
    center /= float(sampleCount);
    
    // Calculate radius
    for (uint i = 0; i < sampleCount; i++) {
        uint vIndex = meshlet.vertexOffset + (i * meshlet.vertexCount / sampleCount);
        vec3 pos = vertexBuffer.positions[vIndex].xyz;
        radius = max(radius, length(pos - center));
    }
    
    // Test against frustum planes
    vec4 centerH = vec4(center, 1.0);
    for (int i = 0; i < 6; i++) {
        float distance = dot(centerH, camera.frustumPlanes[i]);
        if (distance < -radius) {
            return false;
        }
    }
    
    return true;
}

bool shouldCullMeshlet(uint meshletIndex) {
    MeshletDescriptor meshlet = meshlets[meshletIndex];
    
    // Distance culling
    if (pc.cullDistance > 0.0) {
        vec3 center = vec3(0.0);
        for (uint i = 0; i < min(4u, meshlet.vertexCount); i++) {
            center += vertexBuffer.positions[meshlet.vertexOffset + i].xyz;
        }
        center /= float(min(4u, meshlet.vertexCount));
        
        float distance = length(center - camera.cameraPosition);
        if (distance > pc.cullDistance) {
            return true;
        }
    }
    
    // Frustum culling
    if (pc.enableFrustumCulling != 0) {
        if (!isMeshletInFrustum(meshletIndex)) {
            return true;
        }
    }
    
    // TODO: Add backface culling if needed
    // TODO: Add occlusion culling with Hi-Z
    
    return false;
}

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_GlobalInvocationID.x;
    
    // Initialize shared memory
    if (tid == 0) {
        sh_totalPassed = 0;
    }
    sh_passedCulling[tid] = 0;
    barrier();
    
    // Process meshlets assigned to this workgroup
    uint meshletBase = gl_WorkGroupID.x * pc.meshletsPerWorkgroup;
    uint meshletIndex = meshletBase + tid;
    
    // Perform culling tests
    if (tid < pc.meshletsPerWorkgroup && meshletIndex < pc.totalMeshlets) {
        if (!shouldCullMeshlet(meshletIndex)) {
            sh_passedCulling[tid] = 1;
            sh_meshletIndices[tid] = meshletIndex;
        }
    }
    barrier();
    
    // Compact passed meshlets
    uint passed = sh_passedCulling[tid];
    uint offset = subgroupExclusiveAdd(passed);
    
    if (passed != 0) {
        uint writeIndex = atomicAdd(sh_totalPassed, 1);
        if (writeIndex < MAX_MESHLETS_PER_WORKGROUP) {
            payload.meshletIndices[writeIndex] = sh_meshletIndices[tid];
        }
    }
    barrier();
    
    // Emit mesh shader work
    if (tid == 0) {
        payload.meshletCount = min(sh_totalPassed, MAX_MESHLETS_PER_WORKGROUP);
        EmitMeshTasksEXT(payload.meshletCount, 1, 1);
    }
}