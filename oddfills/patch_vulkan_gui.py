#!/usr/bin/env python3
"""
patch_vulkan_gui.py — Arcane Wild Rune Engine
Writes vulk.cpp directly from clean_main.cpp with all engine logic injected.
Fixes applied vs previous version:
  1. Voronoi seed buffer initialized (64 random seeds)
  2. UBO struct matches shader exactly (all 8 float fields)
  3. Audio: device started only when file loaded, decoder configured to device rate
  4. Full GNG training loop ported from m.cpp (FFT + node/edge update + insertion)
  5. RD cavities SUBTRACT from SDF (carve holes) — corrected sign
  6. UBO dynamic fields (ws, mt, nc) updated from audio energy each frame
"""
import os, re

input_file  = '/home/unichronic/oddfills/oddfills/clean_main.cpp'
output_file = '/home/unichronic/oddfills/oddfills/vulk.cpp'

if not os.path.exists(input_file):
    print("ERROR: clean_main.cpp not found — run un_obfuscate.py first")
    exit(1)

with open(input_file, 'r') as f:
    text = f.read()

# ─────────────────────────────────────────────────
# 1. GLOBALS & HELPERS  (injected after vulkan.h)
# ─────────────────────────────────────────────────
HEADERS = r"""
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ── UBO matches shader exactly ──────────────────────────────────────────────
// layout: mat4 model, mat4 view, mat4 proj, vec4 eye, vec4 target,
//         float energy, float time, int node_count,
//         float warp_strength, float noise_scale,
//         float membrane_thickness, float spike_density, float _pad
typedef struct { float m[16]; } Mat4;
typedef struct { float x,y,z,w; } Vec4;
typedef struct {
    Mat4  model, view, proj;
    Vec4  eye, target;
    float energy;
    float time;
    int   node_count;
    float warp_strength;
    float noise_scale;
    float membrane_thickness;
    float spike_density;
    float _pad;
} UniformBlock;

typedef struct { Vec4 pos_radius; } ShaderNode;

// ── Runtime state ────────────────────────────────────────────────────────────
static UniformBlock g_ubo;
static ShaderNode   g_nodes[500];   // GNG nodes → binding 1
static ShaderNode   g_seeds[64];    // Voronoi seeds → binding 2 (static)
static int          g_node_count = 0;
static float        g_time  = 0.0f;
static float        g_energy = 0.0f;

// ── Audio ────────────────────────────────────────────────────────────────────
static ma_decoder   g_decoder;
static ma_device    g_device;
static char         g_audio_path[1024] = "";
static int          g_has_audio  = 0;
static bool         g_playing    = false;

// ── GNG ─────────────────────────────────────────────────────────────────────
struct GngNode { float p[3]; float err; int alive; };
struct GngEdge { int u, v, age; bool live; };
static GngNode g_gng[500];
static GngEdge g_edges[2000];
static int     g_gng_step = 0;

// ── RD grid ──────────────────────────────────────────────────────────────────
static float g_A[64][64], g_B[64][64], g_nA[64][64], g_nB[64][64];
static float g_rd_feed = 0.055f, g_rd_kill = 0.062f;

// ── Camera ───────────────────────────────────────────────────────────────────
static float g_cam_phi   =  0.5f;
static float g_cam_theta =  0.3f;
static float g_cam_dist  = 35.0f;

// ── Vulkan handles (pipeline) ─────────────────────────────────────────────────
VkPipeline       g_pipe      = VK_NULL_HANDLE;
VkPipelineLayout g_pipe_layout = VK_NULL_HANDLE;
VkDescriptorSet  g_desc_set  = VK_NULL_HANDLE;
VkBuffer         g_vbo       = VK_NULL_HANDLE;
VkDeviceMemory   g_ubo_mem, g_node_mem, g_seed_mem, g_vbo_mem;
VkBuffer         g_ubo_buf, g_node_buf, g_seed_buf;

// ── Simple DFT for audio embedding ────────────────────────────────────────────
static float g_sample_rate = 44100.0f;
static float dft_mag(float* s, int c, float freq) {
    float r=0, im=0;
    for(int x=0;x<c;x++){ float w=6.28318f*freq*x/g_sample_rate; r+=s[x]*cosf(w); im-=s[x]*sinf(w); }
    return sqrtf(r*r+im*im)/c;
}

// ── GNG: one training step for sample point sp[3] ────────────────────────────
static void gng_update(float sp[3]) {
    int s1=-1, s2=-1; float d1=1e9f, d2=1e9f;
    for(int i=0;i<500;i++) if(g_gng[i].alive){
        float dx=g_gng[i].p[0]-sp[0], dy=g_gng[i].p[1]-sp[1], dz=g_gng[i].p[2]-sp[2];
        float dd=dx*dx+dy*dy+dz*dz;
        if(dd<d1){ s2=s1; d2=d1; s1=i; d1=dd; } else if(dd<d2){ s2=i; d2=dd; }
    }
    if(s1<0||s2<0) return;
    g_gng[s1].err += d1;
    float lr=0.2f;
    g_gng[s1].p[0]+=lr*(sp[0]-g_gng[s1].p[0]);
    g_gng[s1].p[1]+=lr*(sp[1]-g_gng[s1].p[1]);
    g_gng[s1].p[2]+=lr*(sp[2]-g_gng[s1].p[2]);
    // move neighbours, age edges
    int found_edge = 0;
    for(int i=0;i<2000;i++) if(g_edges[i].live && (g_edges[i].u==s1 || g_edges[i].v==s1)){
        int nb = (g_edges[i].u==s1)?g_edges[i].v:g_edges[i].u;
        g_gng[nb].p[0]+=0.006f*(sp[0]-g_gng[nb].p[0]);
        g_gng[nb].p[1]+=0.006f*(sp[1]-g_gng[nb].p[1]);
        g_gng[nb].p[2]+=0.006f*(sp[2]-g_gng[nb].p[2]);
        g_edges[i].age++;
        if(nb==s2){ g_edges[i].age=0; found_edge=1; }
    }
    if(!found_edge) for(int i=0;i<2000;i++) if(!g_edges[i].live){
        g_edges[i].u=s1; g_edges[i].v=s2; g_edges[i].age=0; g_edges[i].live=true; break;
    }
    // prune old edges
    for(int i=0;i<2000;i++) if(g_edges[i].live && g_edges[i].age>50) g_edges[i].live=false;
    // node insertion every 100 steps
    g_gng_step++;
    if(g_gng_step % 100 == 0){
        int q=-1; float me=-1;
        for(int i=0;i<500;i++) if(g_gng[i].alive && g_gng[i].err>me){ me=g_gng[i].err; q=i; }
        int f_=-1; me=-1;
        for(int i=0;i<2000;i++) if(g_edges[i].live && (g_edges[i].u==q||g_edges[i].v==q)){
            int nb=(g_edges[i].u==q)?g_edges[i].v:g_edges[i].u;
            if(g_gng[nb].err>me){ me=g_gng[nb].err; f_=nb; }
        }
        if(q>=0 && f_>=0){
            int r=-1;
            for(int i=0;i<500;i++) if(!g_gng[i].alive){ r=i; break; }
            if(r>=0){
                g_gng[r].alive=1;
                g_gng[r].p[0]=(g_gng[q].p[0]+g_gng[f_].p[0])/2;
                g_gng[r].p[1]=(g_gng[q].p[1]+g_gng[f_].p[1])/2;
                g_gng[r].p[2]=(g_gng[q].p[2]+g_gng[f_].p[2])/2;
                // remove q-f_ edge, add q-r and r-f_
                for(int i=0;i<2000;i++) if(g_edges[i].live &&
                    ((g_edges[i].u==q&&g_edges[i].v==f_)||(g_edges[i].u==f_&&g_edges[i].v==q)))
                    g_edges[i].live=false;
                int e1=-1,e2=-1;
                for(int i=0;i<2000;i++) if(!g_edges[i].live){ if(e1<0) e1=i; else { e2=i; break; } }
                if(e1>=0){ g_edges[e1].u=q; g_edges[e1].v=r; g_edges[e1].age=0; g_edges[e1].live=true; }
                if(e2>=0){ g_edges[e2].u=r; g_edges[e2].v=f_; g_edges[e2].age=0; g_edges[e2].live=true; }
                g_gng[q].err*=0.5f; g_gng[f_].err*=0.5f; g_gng[r].err=g_gng[q].err;
            }
        }
    }
    for(int i=0;i<500;i++) if(g_gng[i].alive) g_gng[i].err*=0.995f;
}

// ── Audio callback — FFT → GNG training (ported from m.cpp) ──────────────────
static void audio_callback(ma_device* dev, void* out, const void* in, ma_uint32 frame_count) {
    (void)in;
    if(!g_has_audio || !g_playing){
        memset(out, 0, frame_count * dev->playback.channels * sizeof(float));
        return;
    }
    float* f = (float*)out;
    ma_decoder* dec = (ma_decoder*)dev->pUserData;
    // CRITICAL: return value is ma_result, NOT frame count.
    // Actual frames read is returned via the 4th param (ma_uint64*)
    ma_uint64 rd = 0;
    ma_decoder_read_pcm_frames(dec, f, frame_count, &rd);
    if(rd == 0){
        ma_decoder_seek_to_pcm_frame(dec, 0);
        ma_decoder_read_pcm_frames(dec, f, frame_count, &rd);
    }
    if(rd < (ma_uint64)frame_count)
        memset(f + rd*dev->playback.channels, 0,
               (frame_count-(ma_uint32)rd)*dev->playback.channels*sizeof(float));

    // Energy from actual f32 samples
    float amp = 0;
    for(ma_uint64 j=0; j<rd; j++) amp += fabsf(f[j * dev->playback.channels]);
    g_energy = amp / (rd > 0 ? (float)rd : 1.0f);

    // FFT embedding: low/mid/high frequency magnitudes
    if(rd > 0){
        // Build mono mix for FFT
        int nc = (int)dev->playback.channels;
        static float mono[4096];
        int cnt = rd < 4096 ? (int)rd : 4096;
        for(int x=0;x<cnt;x++){
            float s=0; for(int c=0;c<nc;c++) s+=f[x*nc+c]; mono[x]=s/nc;
        }
        float lo = dft_mag(mono, cnt, 100.0f);
        float mi = dft_mag(mono, cnt, 1000.0f);
        float hi = dft_mag(mono, cnt, 5000.0f);
        float sp[3] = { lo*50.0f, mi*100.0f, hi*150.0f };
        gng_update(sp);
    }
}

// ── Scene initialisation ──────────────────────────────────────────────────────
static void scene_init() {
    // 64 Voronoi seeds scattered in [-10, 10]³
    for(int i=0;i<64;i++){
        g_seeds[i].pos_radius.x = ((rand()/(float)RAND_MAX)*20)-10;
        g_seeds[i].pos_radius.y = ((rand()/(float)RAND_MAX)*20)-10;
        g_seeds[i].pos_radius.z = ((rand()/(float)RAND_MAX)*20)-10;
        g_seeds[i].pos_radius.w = 1.0f;
    }
    // 8 initial GNG nodes
    for(int i=0;i<8;i++){
        g_gng[i].alive=1;
        g_gng[i].p[0]=((rand()/(float)RAND_MAX)*10)-5;
        g_gng[i].p[1]=((rand()/(float)RAND_MAX)*10)-5;
        g_gng[i].p[2]=((rand()/(float)RAND_MAX)*10)-5;
    }
    // RD initial state
    for(int y=0;y<64;y++) for(int x=0;x<64;x++){ g_A[y][x]=1.0f; g_B[y][x]=0.0f; }
    for(int y=30;y<34;y++) for(int x=30;x<34;x++) g_B[y][x]=1.0f;
    // UBO identity matrices
    memset(&g_ubo, 0, sizeof(g_ubo));
    g_ubo.model.m[0]=g_ubo.model.m[5]=g_ubo.model.m[10]=g_ubo.model.m[15]=1;
    g_ubo.view .m[0]=g_ubo.view .m[5]=g_ubo.view .m[10]=g_ubo.view .m[15]=1;
    g_ubo.proj .m[0]=g_ubo.proj .m[5]=g_ubo.proj .m[10]=g_ubo.proj .m[15]=1;
    g_ubo.warp_strength    = 0.5f;
    g_ubo.noise_scale      = 0.2f;
    g_ubo.membrane_thickness = 0.1f;
    g_ubo.spike_density    = 3.5f;
    g_ubo.node_count       = 0;
}

// ── Vulkan helpers ────────────────────────────────────────────────────────────
static void create_buffer(VkDevice dev, VkPhysicalDevice pdev, VkDeviceSize size,
                           VkBufferUsageFlags usage, VkBuffer* buf, VkDeviceMemory* mem) {
    VkBufferCreateInfo ci={};
    ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; ci.size=size;
    ci.usage=usage; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(dev, &ci, nullptr, buf);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *buf, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    uint32_t ti=0;
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        if((mr.memoryTypeBits&(1<<i)) &&
           (mp.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))
            { ti=i; break; }
    VkMemoryAllocateInfo ai={};
    ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=mr.size; ai.memoryTypeIndex=ti;
    vkAllocateMemory(dev, &ai, nullptr, mem); vkBindBufferMemory(dev, *buf, *mem, 0);
}

static void upload_buffer(VkDevice dev, VkDeviceMemory mem, const void* data, VkDeviceSize size) {
    void* p; vkMapMemory(dev, mem, 0, size, 0, &p); memcpy(p, data, size); vkUnmapMemory(dev, mem);
}

static VkShaderModule load_spv(VkDevice dev, const char* path) {
    FILE* f=fopen(path,"rb"); if(!f){ printf("SPV missing: %s\n", path); return VK_NULL_HANDLE; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint32_t* buf=(uint32_t*)malloc(sz); size_t _u=fread(buf,1,sz,f); (void)_u; fclose(f);
    VkShaderModuleCreateInfo ci={};
    ci.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; ci.codeSize=sz; ci.pCode=buf;
    VkShaderModule m; vkCreateShaderModule(dev,&ci,nullptr,&m); free(buf); return m;
}
"""

text = text.replace('#include <vulkan/vulkan.h>', '#include <vulkan/vulkan.h>\n' + HEADERS)

# ─────────────────────────────────────────────────
# 2. MAIN SETUP  (after ImGui_ImplVulkan_Init)
# ─────────────────────────────────────────────────
SETUP = r"""
    // ── Initialise engine state ────────────────────────────────────────────
    scene_init();

    // ── Audio device (NOT started yet — only on file load) ─────────────────
    ma_device_config dc_cfg = ma_device_config_init(ma_device_type_playback);
    dc_cfg.playback.format   = ma_format_f32;
    dc_cfg.playback.channels = 2;
    dc_cfg.sampleRate        = 0;          // negotiate native rate
    dc_cfg.dataCallback      = audio_callback;
    dc_cfg.pUserData         = &g_decoder;
    ma_device_init(NULL, &dc_cfg, &g_device);
    // g_sample_rate set after device init
    g_sample_rate = (float)(g_device.sampleRate > 0 ? g_device.sampleRate : 44100);

    // ── Vertex buffer: full-screen quad ───────────────────────────────────
    float verts[] = {-1,-1,0,  1,-1,0,  -1,1,0,  1,1,0};
    create_buffer(g_Device, g_PhysicalDevice, sizeof(verts),
                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &g_vbo, &g_vbo_mem);
    upload_buffer(g_Device, g_vbo_mem, verts, sizeof(verts));

    // ── UBO buffer ────────────────────────────────────────────────────────
    create_buffer(g_Device, g_PhysicalDevice, sizeof(UniformBlock),
                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &g_ubo_buf, &g_ubo_mem);

    // ── Node SSBO (GNG, binding=1) ────────────────────────────────────────
    create_buffer(g_Device, g_PhysicalDevice, sizeof(g_nodes),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_node_buf, &g_node_mem);

    // ── Seed SSBO (Voronoi seeds, binding=2) — upload once ───────────────
    create_buffer(g_Device, g_PhysicalDevice, sizeof(g_seeds),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_seed_buf, &g_seed_mem);
    upload_buffer(g_Device, g_seed_mem, g_seeds, sizeof(g_seeds));

    // ── Descriptor layout & pool ──────────────────────────────────────────
    VkDescriptorSetLayoutBinding bindings[3]={};
    bindings[0].binding=0; bindings[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount=1; bindings[0].stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding=1; bindings[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount=1; bindings[1].stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[2].binding=2; bindings[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount=1; bindings[2].stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dlci={};
    dlci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount=3; dlci.pBindings=bindings;
    VkDescriptorSetLayout desc_layout;
    vkCreateDescriptorSetLayout(g_Device, &dlci, nullptr, &desc_layout);

    VkDescriptorPoolSize pool_sizes[2]={};
    pool_sizes[0].type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; pool_sizes[0].descriptorCount=1;
    pool_sizes[1].type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; pool_sizes[1].descriptorCount=2;
    VkDescriptorPoolCreateInfo dpci={};
    dpci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount=2; dpci.pPoolSizes=pool_sizes; dpci.maxSets=1;
    VkDescriptorPool desc_pool;
    vkCreateDescriptorPool(g_Device, &dpci, nullptr, &desc_pool);

    VkDescriptorSetAllocateInfo dsai={};
    dsai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool=desc_pool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&desc_layout;
    vkAllocateDescriptorSets(g_Device, &dsai, &g_desc_set);

    // ── Bind buffers to descriptor set ────────────────────────────────────
    VkDescriptorBufferInfo dbi0={}; dbi0.buffer=g_ubo_buf;  dbi0.range=sizeof(UniformBlock);
    VkDescriptorBufferInfo dbi1={}; dbi1.buffer=g_node_buf; dbi1.range=sizeof(g_nodes);
    VkDescriptorBufferInfo dbi2={}; dbi2.buffer=g_seed_buf; dbi2.range=sizeof(g_seeds);
    VkWriteDescriptorSet wds[3]={};
    wds[0].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wds[0].dstSet=g_desc_set;
    wds[0].dstBinding=0; wds[0].descriptorCount=1;
    wds[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wds[0].pBufferInfo=&dbi0;
    wds[1].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wds[1].dstSet=g_desc_set;
    wds[1].dstBinding=1; wds[1].descriptorCount=1;
    wds[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[1].pBufferInfo=&dbi1;
    wds[2].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wds[2].dstSet=g_desc_set;
    wds[2].dstBinding=2; wds[2].descriptorCount=1;
    wds[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[2].pBufferInfo=&dbi2;
    vkUpdateDescriptorSets(g_Device, 3, wds, 0, nullptr);

    // ── Graphics pipeline ─────────────────────────────────────────────────
    VkPipelineLayoutCreateInfo plci={};
    plci.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount=1; plci.pSetLayouts=&desc_layout;
    vkCreatePipelineLayout(g_Device, &plci, nullptr, &g_pipe_layout);

    VkShaderModule vsm = load_spv(g_Device, "/home/unichronic/oddfills/anse/build/shaders/neural.vert.spv");
    VkShaderModule fsm = load_spv(g_Device, "/home/unichronic/oddfills/anse/build/shaders/neural.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]={};
    stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vsm; stages[0].pName="main";
    stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=fsm; stages[1].pName="main";

    VkVertexInputBindingDescription   vibd={}; vibd.stride=12; vibd.inputRate=VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription viad={}; viad.format=VK_FORMAT_R32G32B32_SFLOAT;
    VkPipelineVertexInputStateCreateInfo   pvisci={};
    pvisci.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pvisci.vertexBindingDescriptionCount=1; pvisci.pVertexBindingDescriptions=&vibd;
    pvisci.vertexAttributeDescriptionCount=1; pvisci.pVertexAttributeDescriptions=&viad;
    VkPipelineInputAssemblyStateCreateInfo piasci={};
    piasci.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    piasci.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo pvsci={};
    pvsci.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    pvsci.viewportCount=1; pvsci.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo prsci={};
    prsci.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    prsci.polygonMode=VK_POLYGON_MODE_FILL; prsci.lineWidth=1.f; prsci.cullMode=VK_CULL_MODE_NONE;
    VkPipelineMultisampleStateCreateInfo pmsci={};
    pmsci.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    pmsci.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState pcbas={}; pcbas.colorWriteMask=0xF;
    VkPipelineColorBlendStateCreateInfo pcbsci={};
    pcbsci.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    pcbsci.attachmentCount=1; pcbsci.pAttachments=&pcbas;
    VkDynamicState dyn[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo pdsci={};
    pdsci.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    pdsci.dynamicStateCount=2; pdsci.pDynamicStates=dyn;

    VkGraphicsPipelineCreateInfo gpi={};
    gpi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.stageCount=2; gpi.pStages=stages;
    gpi.pVertexInputState=&pvisci; gpi.pInputAssemblyState=&piasci;
    gpi.pViewportState=&pvsci; gpi.pRasterizationState=&prsci;
    gpi.pMultisampleState=&pmsci; gpi.pColorBlendState=&pcbsci;
    gpi.pDynamicState=&pdsci; gpi.layout=g_pipe_layout; gpi.renderPass=wd->RenderPass;
    vkCreateGraphicsPipelines(g_Device, VK_NULL_HANDLE, 1, &gpi, nullptr, &g_pipe);

    double g_last_time = glfwGetTime();
"""

# Inject SETUP right before "bool show_demo_window"
text = re.sub(r'(\s*bool show_demo_window = true;)', SETUP + r'\n\1', text)

# ─────────────────────────────────────────────────
# 3. FRAME UPDATE  (after glfwPollEvents)
# ─────────────────────────────────────────────────
LOOP_UPDATE = r"""
        // ── Delta time ──────────────────────────────────────────────────────
        double g_cur_time = glfwGetTime();
        float dT = (float)(g_cur_time - g_last_time);
        if(dT > 0.05f) dT = 0.016f;
        g_last_time = g_cur_time;
        g_time += dT;

        // ── RD simulation ───────────────────────────────────────────────────
        for(int y=0;y<64;y++) for(int x=0;x<64;x++){
            float a=g_A[y][x], b=g_B[y][x];
            float lapA = -a;
            float lapB = -b;
            lapA += 0.2f*(g_A[y>0?y-1:63][x]+g_A[y<63?y+1:0][x]+g_A[y][x>0?x-1:63]+g_A[y][x<63?x+1:0]);
            lapB += 0.2f*(g_B[y>0?y-1:63][x]+g_B[y<63?y+1:0][x]+g_B[y][x>0?x-1:63]+g_B[y][x<63?x+1:0]);
            lapA += 0.05f*(g_A[y>0?y-1:63][x>0?x-1:63]+g_A[y<63?y+1:0][x<63?x+1:0]
                          +g_A[y>0?y-1:63][x<63?x+1:0]+g_A[y<63?y+1:0][x>0?x-1:63]);
            lapB += 0.05f*(g_B[y>0?y-1:63][x>0?x-1:63]+g_B[y<63?y+1:0][x<63?x+1:0]
                          +g_B[y>0?y-1:63][x<63?x+1:0]+g_B[y<63?y+1:0][x>0?x-1:63]);
            float ab2 = a*b*b;
            g_nA[y][x] = a + 1.0f*lapA - ab2 + g_rd_feed*(1-a);
            g_nB[y][x] = b + 0.5f*lapB + ab2 - (g_rd_kill+g_rd_feed)*b;
        }
        for(int y=0;y<64;y++) for(int x=0;x<64;x++){ g_A[y][x]=g_nA[y][x]; g_B[y][x]=g_nB[y][x]; }

        // ── Orbit camera ─────────────────────────────────────────────────────
        static double cam_lx=0, cam_ly=0; double cx, cy;
        glfwGetCursorPos(window, &cx, &cy);
        if(!ImGui::GetIO().WantCaptureMouse && glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT)){
            g_cam_phi   += (float)(cx - cam_lx) * 0.01f;
            g_cam_theta -= (float)(cy - cam_ly) * 0.01f;
            if(g_cam_theta >  1.5f) g_cam_theta =  1.5f;
            if(g_cam_theta < -1.5f) g_cam_theta = -1.5f;
        }
        cam_lx = cx; cam_ly = cy;
        g_cam_dist -= ImGui::GetIO().MouseWheel * 2.0f;
        if(g_cam_dist < 5.0f) g_cam_dist = 5.0f;
        if(g_cam_dist > 80.0f) g_cam_dist = 80.0f;

        // ── Build node buffer: RD grid → spherically projected at radius 10 ────
        // GNG nodes are in FFT-space (0-150), shader expects radius 10 sphere space.
        g_node_count = 0;
        for(int y=0; y<64; y+=2) for(int x=0; x<64; x+=2){
            if(g_B[y][x] > 0.08f && g_node_count < 500){
                float phi   = (float)x / 64.0f * 6.28318f;
                float theta = (float)y / 64.0f * 3.14159f;
                float r     = 10.0f * (0.85f + g_B[y][x] * 0.15f);
                g_nodes[g_node_count].pos_radius.x = r * sinf(theta) * cosf(phi);
                g_nodes[g_node_count].pos_radius.y = r * cosf(theta);
                g_nodes[g_node_count].pos_radius.z = r * sinf(theta) * sinf(phi);
                g_nodes[g_node_count].pos_radius.w = g_B[y][x];
                g_node_count++;
            }
        }

        // ── Update UBO ───────────────────────────────────────────────────────
        g_ubo.eye.x = g_cam_dist * cosf(g_cam_theta) * cosf(g_cam_phi);
        g_ubo.eye.y = g_cam_dist * sinf(g_cam_theta);
        g_ubo.eye.z = g_cam_dist * cosf(g_cam_theta) * sinf(g_cam_phi);
        g_ubo.target.x = 0; g_ubo.target.y = 0; g_ubo.target.z = 0;
        g_ubo.energy   = g_energy;
        g_ubo.time     = g_time;
        g_ubo.node_count = g_node_count;
        // Audio-reactive parameters (matching m.cpp: u.ws=0.5+e*2, u.mt=0.1+e)
        g_ubo.warp_strength      = 0.5f + g_energy * 2.0f;
        g_ubo.membrane_thickness = 0.1f + g_energy;
        g_ubo.spike_density      = 3.5f + g_energy * 2.0f;

        upload_buffer(g_Device, g_ubo_mem,  &g_ubo,    sizeof(g_ubo));
        upload_buffer(g_Device, g_node_mem,  g_nodes,  sizeof(ShaderNode)*g_node_count);
"""

text = re.sub(r'glfwPollEvents\(\);', 'glfwPollEvents();\n' + LOOP_UPDATE, text)

# ─────────────────────────────────────────────────
# 4. DRAW COMMANDS  (before ImGui_ImplVulkan_RenderDrawData)
# ─────────────────────────────────────────────────
DRAW = r"""
        if(g_pipe != VK_NULL_HANDLE){
            vkCmdBindPipeline(fd->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe);
            VkViewport vp={}; vp.width=(float)wd->Width; vp.height=(float)wd->Height; vp.maxDepth=1.f;
            vkCmdSetViewport(fd->CommandBuffer,0,1,&vp);
            VkRect2D sc={}; sc.extent.width=wd->Width; sc.extent.height=wd->Height;
            vkCmdSetScissor(fd->CommandBuffer,0,1,&sc);
            VkDeviceSize off=0; vkCmdBindVertexBuffers(fd->CommandBuffer,0,1,&g_vbo,&off);
            vkCmdBindDescriptorSets(fd->CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    g_pipe_layout,0,1,&g_desc_set,0,nullptr);
            vkCmdDraw(fd->CommandBuffer,4,1,0,0);
        }
"""

text = text.replace('ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);',
                    DRAW + '\n        ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);')

# ─────────────────────────────────────────────────
# 5. IMGUI DASHBOARD  (replace demo window)
# ─────────────────────────────────────────────────
GUI = r"""
        ImGuiIO& io = ImGui::GetIO();
        // ── Bottom dashboard ────────────────────────────────────────────────
        ImGui::SetNextWindowPos(ImVec2(0, io.DisplaySize.y - 150));
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 150));
        ImGui::SetNextWindowBgAlpha(0.75f);
        ImGui::Begin("##dashboard", nullptr,
            ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
            ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar);

        // Row 1: Audio controls
        if(ImGui::Button("Load Audio")){
            FILE* fp = popen("zenity --file-selection --title=\"Select Audio\"","r");
            if(fp){
                char path[1024]; if(fgets(path,1024,fp)){
                    path[strcspn(path,"\\n")]=0; strcpy(g_audio_path, path);
                    // Only stop/uninit if already playing a file
                    if(g_has_audio){ ma_device_stop(&g_device); ma_decoder_uninit(&g_decoder); }
                    g_has_audio=0; g_playing=false; g_energy=0;
                    // CRITICAL: explicit f32 config forces decoder to convert format/rate to match device
                    // Using NULL config gives native int16 into f32 buffer -> silence
                    uint32_t devRate = (g_device.sampleRate > 0) ? g_device.sampleRate : 44100;
                    uint32_t devCh   = (g_device.playback.channels > 0) ? g_device.playback.channels : 2;
                    ma_decoder_config dec_cfg = ma_decoder_config_init(ma_format_f32, devCh, devRate);
                    if(ma_decoder_init_file(g_audio_path, &dec_cfg, &g_decoder) == MA_SUCCESS){
                        g_has_audio=1; g_playing=true;
                        g_sample_rate = (float)devRate;
                        ma_device_start(&g_device);
                        printf("AUDIO OK rate=%u ch=%u file=%s\n", devRate, devCh, g_audio_path);
                    } else { printf("AUDIO FAIL: %s\n", g_audio_path); }
                }
                pclose(fp);
            }
        }
        ImGui::SameLine();
        if(ImGui::Button(g_playing ? "Pause" : "Play")){
            if(g_has_audio) g_playing = !g_playing;
        }
        ImGui::SameLine();
        if(ImGui::Button("Stop")){
            g_playing=false; g_energy=0;
            if(g_has_audio) ma_decoder_seek_to_pcm_frame(&g_decoder, 0);
        }
        ImGui::SameLine();
        // Progress bar
        ma_uint64 cur=0, len=1;
        if(g_has_audio){
            ma_decoder_get_cursor_in_pcm_frames(&g_decoder, &cur);
            ma_decoder_get_length_in_pcm_frames(&g_decoder, &len);
            if(len==0) len=1;
        }
        ImGui::ProgressBar((float)cur/len, ImVec2(io.DisplaySize.x*0.35f, 0));
        ImGui::SameLine();
        ImGui::Text("%.1f / %.1f s | Energy: %.3f | FPS: %.0f | Nodes: %d",
            (float)cur/(g_device.sampleRate>0?g_device.sampleRate:44100),
            (float)len/(g_device.sampleRate>0?g_device.sampleRate:44100),
            g_energy, io.Framerate, g_node_count);

        // Row 2: RD params
        ImGui::PushItemWidth(180);
        ImGui::SliderFloat("RD Feed", &g_rd_feed, 0.01f, 0.09f); ImGui::SameLine();
        ImGui::SliderFloat("RD Kill", &g_rd_kill, 0.04f, 0.08f); ImGui::SameLine();
        ImGui::SliderFloat("Cam Dist", &g_cam_dist, 5.0f, 80.0f);
        ImGui::PopItemWidth();

        // Row 3: Warp/Membrane/Spike (shown for reference, audio overrides in real-time)
        ImGui::PushItemWidth(180);
        ImGui::SliderFloat("Warp Base",     &g_ubo.warp_strength,      0.0f, 3.0f); ImGui::SameLine();
        ImGui::SliderFloat("Membrane Base", &g_ubo.membrane_thickness, 0.0f, 1.0f); ImGui::SameLine();
        ImGui::SliderFloat("Spike",         &g_ubo.spike_density,      0.0f, 8.0f);
        ImGui::PopItemWidth();

        ImGui::End();
"""

text = re.sub(
    r'if \(show_demo_window\).*?ImGui::End\(\);\s*\}',
    GUI,
    text, flags=re.DOTALL
)
text = re.sub(
    r'if \(show_another_window\).*?ImGui::End\(\);\s*\}',
    '',
    text, flags=re.DOTALL
)

with open(output_file, 'w') as f:
    f.write(text)

print("patch_vulkan_gui.py: vulk.cpp written successfully")
