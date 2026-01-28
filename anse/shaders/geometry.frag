#version 450

layout(location=0) in vec3 fragColor;
layout(location=1) in float fragEmit;

layout(location=0) out vec4 outColor;

layout(set=0, binding=0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eye;
    vec4 target;
    float energy;
    float time;
    int node_count;
    float warp_strength;
    float noise_scale;
    float membrane_thickness;
    float spike_density;
    float _pad;
} ubo;

void main() {
    vec3 col = fragColor;
    
    // Add Arcane bioluminescence
    float pulse = 0.8 + 0.4 * sin(ubo.time * 2.5);
    col *= (fragEmit * pulse * (1.0 + ubo.energy * 2.0));
    
    // Slight transparency/glow effect
    outColor = vec4(col, 1.0);
}
