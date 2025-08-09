#version 460
#extension GL_EXT_mesh_shader : require

// Fragment shader for temporal transient extraction
// Provides shading for both pass 1 (PVS_prev) and pass 2 (PVS_curr-prev)

// Input from mesh shader
layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorldPos;
layout(location = 2) flat in uint inRenderPass;

// Output
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 10, std140) uniform ShadingParameters {
    vec3 lightDir;        // Light direction (normalized)
    vec3 viewPos;         // Camera position
    vec3 baseColor;       // Base material color
    vec3 prevPassColor;   // Color for previous frame geometry
    vec3 newPassColor;    // Color for newly visible geometry
    float ambientStrength;
    float diffuseStrength;
    float specularStrength;
    float shininess;
    uint enableDebugColors; // 1 = show different colors for passes
} shading;

void main() {
    // Normalize the input normal
    vec3 normal = normalize(inNormal);
    
    // Choose base color based on render pass (if debug colors enabled)
    vec3 materialColor = shading.baseColor;
    if (shading.enableDebugColors != 0) {
        // Use different colors to visualize temporal coherence
        if (inRenderPass == 0) {
            // Pass 1: PVS_prev - geometry from previous frame (blue-ish)
            materialColor = shading.prevPassColor;
        } else {
            // Pass 2: PVS_curr-prev - newly visible geometry (red-ish)
            materialColor = shading.newPassColor;
        }
    }
    
    // Ambient component
    vec3 ambient = shading.ambientStrength * materialColor;
    
    // Diffuse component
    vec3 lightDir = normalize(-shading.lightDir);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = shading.diffuseStrength * diff * materialColor;
    
    // Specular component
    vec3 viewDir = normalize(shading.viewPos - inWorldPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shading.shininess);
    vec3 specular = shading.specularStrength * spec * vec3(1.0);
    
    // Combine all components
    vec3 result = ambient + diffuse + specular;
    
    // Apply simple tone mapping
    result = result / (result + vec3(1.0));
    
    // Gamma correction
    result = pow(result, vec3(1.0/2.2));
    
    outColor = vec4(result, 1.0);
    
    // Optional: Add subtle outline based on normal direction for better visibility
    float outline = 1.0 - dot(viewDir, normal);
    outline = smoothstep(0.0, 0.3, outline);
    outColor.rgb = mix(outColor.rgb, vec3(0.0), outline * 0.2);
}