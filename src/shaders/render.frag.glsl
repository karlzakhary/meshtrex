#version 450

layout(location = 0) in PerVertexData {
    vec3 normal;
    flat uint meshletID;
} inData;

layout(location = 0) out vec4 outColor;

void main()
{
    // Generate a unique color for each meshlet
    vec3 meshletColor;
    {
        // Use golden ratio to generate visually distinct colors
        float golden_ratio = 1.618033988749895;
        float hue = fract(float(inData.meshletID) * golden_ratio * 0.1);
        
        // Convert HSV to RGB
        float h = hue * 6.0;
        float c = 0.8; // Saturation
        float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
        
        vec3 rgb;
        if (h < 1.0)      rgb = vec3(c, x, 0.0);
        else if (h < 2.0) rgb = vec3(x, c, 0.0);
        else if (h < 3.0) rgb = vec3(0.0, c, x);
        else if (h < 4.0) rgb = vec3(0.0, x, c);
        else if (h < 5.0) rgb = vec3(x, 0.0, c);
        else              rgb = vec3(c, 0.0, x);
        
        // Add value (brightness)
        meshletColor = rgb + vec3(0.2);
    }
    
    // Normalize the interpolated normal
    vec3 N = normalize(inData.normal);
    
    // Simple directional light pointing down and slightly forward
    vec3 L = normalize(vec3(0.3, -0.7, -0.5));
    
    // View direction (assuming camera looks down -Z in view space)
    vec3 V = vec3(0.0, 0.0, 1.0);
    
    // Lambertian diffuse with meshlet color
    float NdotL = max(dot(N, -L), 0.0);
    vec3 diffuse = meshletColor * NdotL * 0.8;
    
    // Blinn-Phong specular
    vec3 H = normalize(-L + V);
    float NdotH = max(dot(N, H), 0.0);
    float specular = pow(NdotH, 32.0) * 0.3;
    
    // Ambient light with meshlet color influence
    vec3 ambient = meshletColor * 0.2;
    
    // Rim lighting for better shape definition
    float rimFactor = 1.0 - max(dot(V, N), 0.0);
    vec3 rim = vec3(0.1, 0.12, 0.15) * pow(rimFactor, 2.0);
    
    // Combine lighting
    vec3 finalColor = ambient + diffuse + vec3(specular) + rim;
    
    // Add subtle color variation based on normal direction
    vec3 colorVariation = N * 0.05 + 0.95;
    finalColor *= colorVariation;
    
    // Add edge highlighting for meshlet boundaries
    vec3 ddx_pos = dFdx(gl_FragCoord.xyz);
    vec3 ddy_pos = dFdy(gl_FragCoord.xyz);
    float edge_factor = length(fwidth(inData.normal)) * 10.0;
    edge_factor = smoothstep(0.0, 1.0, edge_factor);
    finalColor = mix(finalColor, vec3(0.1), edge_factor * 0.3);
    
    // Tone mapping for better visual quality
    finalColor = finalColor / (finalColor + vec3(1.0));
    finalColor = pow(finalColor, vec3(1.0/2.2)); // Gamma correction
    
    outColor = vec4(finalColor, 1.0);
}