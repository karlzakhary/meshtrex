#version 460 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

layout(location = 0) in vec3 inNormal_world;
layout(location = 1) in vec3 inPosition_world;
// Add other inputs from vertex shader

layout(location = 0) out vec4 outColor;

// --- Binding 0: Scene Data (only need cameraPos usually) ---
layout(binding = 0, std140) uniform SceneUBO {
    mat4 viewProjectionMatrix; // Likely unused here
    mat4 modelMatrix;          // Likely unused here
    vec3 cameraPos_world;
} sceneData;

// --- Binding 1: Lighting Data ---
layout(binding = 1, std140) uniform LightingUBO {
    vec4 lightPosition_world; // w=1 point, w=0 directional
    vec4 lightColor;
    float ambientIntensity;
// other params...
} lightingData;

void main() {
    vec3 N = normalize(inNormal_world);
    vec3 V = normalize(sceneData.cameraPos_world - inPosition_world); // View vector

    // Calculate light vector L
    vec3 L;
    float attenuation = 1.0;
    if (lightingData.lightPosition_world.w == 0.0) {
        L = normalize(-lightingData.lightPosition_world.xyz); // Directional light
    } else {
        vec3 lightDelta = lightingData.lightPosition_world.xyz - inPosition_world;
        float dist = length(lightDelta);
        L = normalize(lightDelta);
        // Simple attenuation example
        // attenuation = 1.0 / (1.0 + 0.1*dist + 0.01*dist*dist);
    }

    // Simple Diffuse + Ambient + Specular (Blinn-Phong example)
    float diffuseFactor = max(dot(N, L), 0.0);
    vec3 diffuse = diffuseFactor * lightingData.lightColor.rgb * attenuation;

    vec3 ambient = lightingData.ambientIntensity * lightingData.lightColor.rgb; // Assume ambientIntensity is from UBO or constant

    vec3 H = normalize(L + V); // Halfway vector
    float specFactor = pow(max(dot(N, H), 0.0), 32.0); // Shininess = 32
    vec3 specular = specFactor * lightingData.lightColor.rgb * attenuation; // Assuming white specular highlights

    vec3 finalColor = ambient + diffuse + specular;

    outColor = vec4(finalColor, 1.0);
}