#version 450

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;
layout(location=2) in float inEmit;

layout(location=0) out vec3 fragColor;
layout(location=1) out float fragEmit;

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
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos, 1.0);
    fragColor = inColor;
    fragEmit = inEmit;
}
