#version 460 core

// Define this to enable block-based color debugging (must match mesh shader)
#define DEBUG_BLOCK_COLORS 0

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragPos;
#if DEBUG_BLOCK_COLORS
layout(location = 2) in flat uint blockID;
#endif

layout(location = 0) out vec4 outColor;

#if DEBUG_BLOCK_COLORS
// Generate a distinct color from block ID
vec3 blockIDToColor(uint id) {
    // Use a hash function to generate distinct colors per block
    float seed = float(id + 1u);  // Add 1 to avoid pure black for ID 0
    float r = fract(sin(seed * 12.9898) * 43758.5453);
    float g = fract(sin(seed * 78.233) * 43758.5453);
    float b = fract(sin(seed * 93.989) * 43758.5453);
    // Ensure minimum brightness for visibility
    return vec3(0.3 + 0.7 * r, 0.3 + 0.7 * g, 0.3 + 0.7 * b);
}
#endif

void main() {
    // Simple lighting calculation
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(fragNormal);
    
    // Basic diffuse lighting
    float NdotL = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = vec3(0.8, 0.8, 0.8) * NdotL;
    
    // Ambient light
    vec3 ambient = vec3(0.2, 0.2, 0.2);
    
#if DEBUG_BLOCK_COLORS
    // Debug mode: Color based on block ID
    vec3 blockColor = blockIDToColor(blockID);
    // Apply lighting to the block color
    float lightIntensity = 0.4 + 0.6 * NdotL;  // Keep minimum brightness
    vec3 finalColor = blockColor * lightIntensity;
#else
    // Normal rendering: standard lighting
    vec3 finalColor = ambient + diffuse;
#endif
    
    outColor = vec4(finalColor, 1.0);
}