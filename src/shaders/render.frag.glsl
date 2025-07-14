#version 450
#extension GL_ARB_separate_shader_objects : enable

// Input from the mesh shader (interpolated by the rasterizer)
layout(location = 0) in PerVertexData {
    vec3 normal;
    vec3 worldPos;
} inData;

// The final color output to the framebuffer
layout(location = 0) out vec4 outColor;

// Uniforms for camera and lighting
layout(binding = 0, std140) uniform SceneUBO {
    mat4 viewProjectionMatrix;
    vec3 cameraPos;
    float padding;
    vec4 lightPos;
} scene;

void main() {
    // Normalize vectors for lighting calculation
    vec3 N = normalize(inData.normal);
    vec3 L = normalize(scene.lightPos.xyz - inData.worldPos);
    vec3 V = normalize(scene.cameraPos - inData.worldPos);
    vec3 H = normalize(L + V);

    // Basic Blinn-Phong lighting components
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * vec3(1.0);

    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * vec3(1.0);

    float spec = pow(max(dot(N, H), 0.0), 32.0);
    vec3 specular = 0.5 * spec * vec3(1.0);
    
    // Combine components for the final color
    vec3 finalColor = ambient + diffuse + specular;

    // Write the final, defined color to the output
    outColor = vec4(finalColor, 1.0);
}