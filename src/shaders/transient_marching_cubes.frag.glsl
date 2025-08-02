#version 460 core

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple lighting calculation
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(fragNormal);
    
    // Basic diffuse lighting
    float NdotL = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = vec3(0.8, 0.8, 0.8) * NdotL;
    
    // Ambient light
    vec3 ambient = vec3(0.2, 0.2, 0.2);
    
    // Debug: Color based on position to visualize geometry
    vec3 posColor = fragPos / 64.0; // Normalize to [0,1] for 64^3 volume
    
    // Final color - mix lighting with position color for debugging
    vec3 color = ambient + diffuse;
    color = mix(color, posColor, 0.5);
    outColor = vec4(color, 1.0);
}