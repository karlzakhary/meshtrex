#version 450

layout(location = 0) in PerVertexData {
    vec3 normal;
} inData;

layout(location = 0) out vec4 outColor;

void main()
{
    // Normalize the interpolated normal
    vec3 N = normalize(inData.normal);
    
    // Simple directional light pointing down and slightly forward
    vec3 L = normalize(vec3(0.3, -0.7, -0.5));
    
    // View direction (assuming camera looks down -Z in view space)
    vec3 V = vec3(0.0, 0.0, 1.0);
    
    // Lambertian diffuse
    float NdotL = max(dot(N, -L), 0.0);
    vec3 diffuse = vec3(0.8, 0.78, 0.75) * NdotL;
    
    // Blinn-Phong specular
    vec3 H = normalize(-L + V);
    float NdotH = max(dot(N, H), 0.0);
    float specular = pow(NdotH, 32.0) * 0.5;
    
    // Ambient light with slight blue tint
    vec3 ambient = vec3(0.15, 0.15, 0.18);
    
    // Rim lighting for better shape definition
    float rimFactor = 1.0 - max(dot(V, N), 0.0);
    vec3 rim = vec3(0.1, 0.12, 0.15) * pow(rimFactor, 2.0);
    
    // Combine lighting
    vec3 finalColor = ambient + diffuse + vec3(specular) + rim;
    
    // Add subtle color variation based on normal direction
    vec3 colorVariation = N * 0.05 + 0.95;
    finalColor *= colorVariation;
    
    // Tone mapping for better visual quality
    finalColor = finalColor / (finalColor + vec3(1.0));
    finalColor = pow(finalColor, vec3(1.0/2.2)); // Gamma correction
    
    outColor = vec4(finalColor, 1.0);
}