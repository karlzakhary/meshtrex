#version 460 core
#extension GL_ARB_separate_shader_objects : enable

// Input from mesh shader
layout(location = 0) in vec3 inNormal_world;
layout(location = 1) in vec3 inPosition_world;

// Output color
layout(location = 0) out vec4 outColor;

// Binding 0: Scene Data (MVP matrices and camera position)
layout(set = 0, binding = 0, std140) uniform SceneUBO {
    mat4 viewProjectionMatrix;
    mat4 modelMatrix;
    vec3 cameraPos_world;
    float pad;
} scene;

void main() {
    // Normalize the interpolated normal
    vec3 N = normalize(inNormal_world);
    
    // View vector from fragment to camera
    vec3 V = normalize(scene.cameraPos_world - inPosition_world);
    
    // Simple hardcoded light for now (can be extended later)
    vec3 lightPos = vec3(10.0, 10.0, 10.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float ambientStrength = 0.2;
    
    // Light direction
    vec3 L = normalize(lightPos - inPosition_world);
    
    // Diffuse lighting
    float diffuseFactor = max(dot(N, L), 0.0);
    vec3 diffuse = diffuseFactor * lightColor;
    
    // Ambient lighting
    vec3 ambient = ambientStrength * lightColor;
    
    // Specular lighting (Blinn-Phong)
    vec3 H = normalize(L + V); // Halfway vector
    float specFactor = pow(max(dot(N, H), 0.0), 32.0); // Shininess = 32
    vec3 specular = specFactor * lightColor;
    
    // Combine lighting components
    vec3 result = ambient + diffuse + specular;
    
    // Output final color
    outColor = vec4(result, 1.0);
}