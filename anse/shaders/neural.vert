#version 450

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;
layout(location=2) in float inEmit;

layout(location=0) out vec2 fragUV;
layout(location=1) out vec3 fragRayOrigin;
layout(location=2) out vec3 fragRayDir;

layout(set=0, binding=0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eye;
    vec4 target;
    float energy;
    float time;
    int node_count;
    float _pad;
} ubo;

void main() {
    gl_Position = vec4(inPos.x, inPos.y, 0.0, 1.0);
    fragUV = inPos.xy;

    mat4 invView = inverse(ubo.view);
    mat4 invProj = inverse(ubo.proj);

    // Standard Raymarching camera setup
    vec4 rayOriginWorld = invView * vec4(0, 0, 0, 1);
    vec4 targetNDC = vec4(inPos.x, inPos.y, 1.0, 1.0);
    vec4 targetWorld = invView * invProj * targetNDC;
    targetWorld /= targetWorld.w;

    fragRayOrigin = rayOriginWorld.xyz;
    fragRayDir = normalize(targetWorld.xyz - rayOriginWorld.xyz);
}
