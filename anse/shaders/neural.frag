#version 450

layout(location=0) in vec2 fragUV;
layout(location=1) in vec3 fragRayOrigin;
layout(location=2) in vec3 fragRayDir;

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

struct ShaderNode {
    vec4 pos_radius;
};

layout(std140, set=0, binding=1) readonly buffer NodeBuffer {
    ShaderNode nodes[];
};

layout(std140, set=0, binding=2) readonly buffer SeedBuffer {
    ShaderNode seeds[];
};

// Hash-based noise
float hash(float n) { return fract(sin(n) * 43758.5453123); }
float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f*f*(3.0-2.0*f);
    float n = i.x + i.y*157.0 + 113.0*i.z;
    return mix(mix(mix(hash(n+  0.0), hash(n+  1.0), f.x),
                   mix(hash(n+157.0), hash(n+158.0), f.x), f.y),
               mix(mix(hash(n+113.0), hash(n+114.0), f.x),
                   mix(hash(n+270.0), hash(n+271.0), f.x), f.y), f.z);
}

// Spectral color map
vec3 getSpectralColor(float t) {
    vec3 c = 0.5 + 0.5 * cos(6.28318 * (vec3(1.0, 0.66, 0.33) + t));
    return clamp(c, 0.0, 1.0);
}

// Curl noise for organic flowing membranes
vec3 curl_noise(vec3 p) {
    const float e = 0.1;
    float dx = noise(vec3(p.x + e, p.y, p.z)) - noise(vec3(p.x - e, p.y, p.z));
    float dy = noise(vec3(p.x, p.y + e, p.z)) - noise(vec3(p.x, p.y - e, p.z));
    float dz = noise(vec3(p.x, p.y, p.z + e)) - noise(vec3(p.x, p.y, p.z - e));
    return vec3(dy - dz, dz - dx, dx - dy);
}

// RD density sampling — GNG nodes carve cavities
float sample_RD(vec3 p) {
    float total = 0.0;
    for (int i = 0; i < ubo.node_count; i++) {
        vec3 delta = p - nodes[i].pos_radius.xyz;
        float d2 = dot(delta, delta);
        if (d2 < 16.0) {
            total += exp(-d2 * 0.15);
        }
    }
    return total;
}

vec3 warp_noise(vec3 p) {
    return vec3(
        noise(p * 1.3 + ubo.time * 0.1),
        noise(p * 1.7 + ubo.time * 0.1 + 10.0),
        noise(p * 1.1 + ubo.time * 0.1 + 20.0)
    );
}

// FBM for hierarchical erosion
float fbm3(vec3 p) {
    return noise(p*1.0)*0.35 + noise(p*3.0)*0.18 + noise(p*9.0)*0.05;
}

// Scene SDF — corrected per docupd.md
float map(vec3 p) {
    float radius = 10.0;
    float distSq = dot(p, p);
    if (distSq > 225.0) return sqrt(distSq) - radius;

    // CORRECTED: Reduced curl warp (was 0.8, now 0.28)
    p += curl_noise(p * 0.35) * 0.28;

    // CORRECTED: Increased domain warp boost (was 4.0, now 6.5)
    float ws = ubo.warp_strength * 6.5;
    vec3 p_warp = p + warp_noise(p) * ws;

    // CORRECTED: Three-scale cellular at 1.8, 3.8, 7.5 (was 0.6, 1.5, 3.0)
    float f1A = 1e10, f2A = 1e10;
    float f1B = 1e10, f2B = 1e10;
    float f1C = 1e10, f2C = 1e10;
    vec3 pA = p_warp * 1.8;
    vec3 pB = p_warp * 3.8;
    vec3 pC = p_warp * 7.5;

    for (int i = 0; i < 64; i++) {
        vec3 s = seeds[i].pos_radius.xyz;
        float dA = length(pA - s);
        float dB = length(pB - s);
        float dC = length(pC - s);
        if (dA < f1A) { f2A = f1A; f1A = dA; } else if (dA < f2A) f2A = dA;
        if (dB < f1B) { f2B = f1B; f1B = dB; } else if (dB < f2B) f2B = dB;
        if (dC < f1C) { f2C = f1C; f1C = dC; } else if (dC < f2C) f2C = dC;
    }

    // CORRECTED: Weights 0.55, 0.30, 0.15 (was 0.6, 0.3, 0.1)
    float cell = 0.55*(f2A-f1A) + 0.30*(f2B-f1B) + 0.15*(f2C-f1C);

    // CORRECTED: Thickness 0.055 * radius (was 0.08 * radius)
    float thickness = 0.055 * radius * (0.8 + ubo.membrane_thickness * 0.4);
    float d_membrane = abs(cell) - thickness;

    // Membrane inflation to prevent flat plates
    d_membrane -= noise(p * 0.6) * 0.35;

    float d_sphere = sqrt(distSq) - radius;
    float d = max(d_sphere, d_membrane);

    // CORRECTED: RD cavities SUBTRACT (carve holes), was += 0.6
    float rd_val = sample_RD(p);
    d -= rd_val * 1.6;

    // CORRECTED: Hierarchical erosion (3 octaves)
    d += fbm3(p);

    // CORRECTED: Spike along membrane ridges (was junction formula)
    float spike = pow(abs(cell), 3.5);
    d -= spike * ubo.spike_density * 0.35;

    return d;
}

// Normal via gradient
vec3 calcNormal(vec3 p) {
    const float eps = 0.01;
    vec2 e = vec2(1.0, -1.0) * eps;
    return normalize(
        e.xyy * map(p + e.xyy) +
        e.yyx * map(p + e.yyx) +
        e.yxy * map(p + e.yxy) +
        e.xxx * map(p + e.xxx)
    );
}

void main() {
    vec3 camPos = ubo.eye.xyz;
    vec3 camTar = ubo.target.xyz;
    vec3 forward = normalize(camTar - camPos);
    vec3 right = normalize(cross(forward, vec3(0,1,0)));
    vec3 up = cross(right, forward);

    float aspect = 1.777;
    vec3 rd = normalize(forward + fragUV.x * right * aspect * 0.7 + fragUV.y * up * 0.7);
    vec3 ro = camPos;

    const float MAX_DIST = 100.0;
    const float SURF_DIST = 0.008;
    const int MAX_STEPS = 120;

    // Ray-sphere bounding guard (r=15)
    vec3 oc = ro;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - 225.0;
    float h = b*b - c;
    if (h < 0.0) {
        float grid = step(0.98, fract(fragUV.x * 20.0)) + step(0.98, fract(fragUV.y * 10.0));
        outColor = vec4(vec3(0.01, 0.01, 0.05) + grid * 0.02, 1.0);
        return;
    }
    float t = max(0.0, -b - sqrt(h));

    bool hit = false;
    vec3 p;
    for(int i = 0; i < MAX_STEPS; i++) {
        p = ro + rd * t;
        float d = map(p);
        if(abs(d) < SURF_DIST) { hit = true; break; }
        if (t > MAX_DIST) break;
        // CORRECTED: Fixed step multiplier 0.45 (was adaptive 0.4-0.8)
        t += d * 0.45;
    }

    if (!hit) {
        float grid = step(0.98, fract(fragUV.x * 20.0)) + step(0.98, fract(fragUV.y * 10.0));
        outColor = vec4(vec3(0.01, 0.01, 0.05) + grid * 0.02, 1.0);
        return;
    }

    vec3 n = calcNormal(p);
    vec3 viewDir = normalize(ro - p);
    float NdotV = max(dot(n, viewDir), 0.0);
    float fresnel = pow(1.0 - NdotV, 3.0);

    // Dark Hextech palette
    vec3 baseCol = mix(vec3(0.01, 0.005, 0.08), vec3(0.01, 0.08, 0.12), clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 energyCol = vec3(0.0, 0.6, 1.0);

    float pulse = noise(p * 0.4 + ubo.time * 0.4) * 0.5 + 0.5;
    float interference = NdotV * 0.4 + pulse * 0.6;
    vec3 iridescentCol = getSpectralColor(interference + ubo.energy * 0.2);

    float distFromCenter = length(p);
    float glow = exp(-(10.5 - distFromCenter) * 1.8);

    vec3 finalCol = baseCol * 0.4;
    finalCol = mix(finalCol, iridescentCol, 0.35 * (1.0 - NdotV));
    finalCol += energyCol * glow * (2.5 + ubo.energy * 5.0);
    finalCol += iridescentCol * fresnel * (1.2 + ubo.energy * 1.5);
    finalCol += energyCol * pow(pulse, 6.0) * ubo.energy * 1.5;

    outColor = vec4(clamp(finalCol, 0.0, 1.0), 1.0);
}
