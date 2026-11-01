#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "groot_simd.h"

namespace groot {

// Contiguous pool: ~9.2 GiB (fits within 16 GB RAM envelope with 0 swapping)
constexpr size_t GROOT_POOL_SIZE = 9880000000ULL; // ~9.20 GiB
constexpr size_t SCRATCH_STAGING_BYTES = 250ULL * 1024ULL * 1024ULL; // 250 MB Direct-I/O staging
constexpr size_t HOT_EXPERT_L3_BYTES = 10ULL * 1024ULL * 1024ULL;    // 10 MB Resident L3 Slice

enum class GrootLayerType {
    GDN_LINEAR,
    ATTENTION,
    NEXTN_PREDICT
};

struct GrootExpert {
    const void* gate_ptr = nullptr; // IQ3_XXS [2048, 512] (~392 KB)
    const void* up_ptr   = nullptr; // IQ4_XS  [2048, 512] (~544 KB)
    const void* down_ptr = nullptr; // IQ4_XS  [512, 2048] (~544 KB)
    size_t total_bytes   = 0;       // ~1.48 MB
    bool is_hot_in_l3    = false;   // Pinned in 10 MB L3 Resident Cache
};

struct GrootLayer {
    int layer_idx = 0;
    GrootLayerType type = GrootLayerType::GDN_LINEAR;

    // Normalizations (F32)
    const float* attn_norm = nullptr;          // [2048]
    const float* post_attention_norm = nullptr; // [2048]

    // GDN specific (30 layers)
    const void*  gdn_qkv = nullptr;            // Q6_K / Q5_K [2048, 8192]
    const void*  gdn_gate = nullptr;           // Q8_0 / Q6_K [2048, 4096]
    const float* gdn_conv1d = nullptr;         // F32 [4, 8192]
    const float* gdn_ssm_a = nullptr;          // F32 [32]
    const float* gdn_ssm_alpha = nullptr;      // F32 [2048, 32]
    const float* gdn_ssm_beta = nullptr;       // F32 [2048, 32]
    const float* gdn_ssm_dt_bias = nullptr;    // F32 [32]
    const float* gdn_ssm_norm = nullptr;       // F32 [128]
    const void*  gdn_out = nullptr;            // Q8_0 / Q6_K [4096, 2048]

    // Attention specific (10 + 1 layers)
    const void*  attn_q = nullptr;             // Q6_K / Q8_0 [2048, 2048]
    const void*  attn_k = nullptr;             // Q6_K / Q8_0 [2048, 256]
    const void*  attn_v = nullptr;             // Q6_K / Q8_0 [2048, 256]
    const void*  attn_out = nullptr;           // Q6_K / Q8_0 [2048, 2048]

    // MoE Router & Shared Expert
    const uint16_t* router_fp16 = nullptr;     // F16 [128, 2048] (converted / mapped)
    const void*  ffn_gate_shexp = nullptr;     // Q8_0 [2048, 512]
    const void*  ffn_up_shexp = nullptr;       // Q8_0 [2048, 512]
    const void*  ffn_down_shexp = nullptr;     // Q8_0 [512, 2048]
    const float* ffn_gate_inp_shexp = nullptr; // F32 [2048]

    // 128 MoE Experts
    GrootExpert experts[NUM_EXPERTS];

    size_t layer_bytes = 0;
};

class GrootTiler {
public:
    GrootTiler();
    ~GrootTiler();

    GrootTiler(const GrootTiler&) = delete;
    GrootTiler& operator=(const GrootTiler&) = delete;

    // Load GGUF model via Win32 Direct-I/O bypass into contiguous ~9.2 GiB memory pool
    bool load_and_tile(const std::string& gguf_path);

    // Profile activation frequencies across coding/chat tokens and pin heavy-hitters into 10 MB L3 slice
    void profile_and_pin_zipfian_l3_cache();

    // Warm router weights in L3 cache
    void warm_router_l3_cache();

    // Prefault memory pool
    void prefault_pool();

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    const GrootLayer& get_layer(int idx) const { return m_layers[idx]; }
    const void* get_token_embd() const { return m_token_embd; }
    const float* get_output_norm() const { return m_output_norm; }
    const void* get_output_weight() const { return m_output_weight; }

    // Zipfian Hot-Expert Telemetry
    double get_hot_expert_hit_rate() const { return m_hot_hit_rate; }
    size_t get_pinned_hot_expert_count() const { return m_hot_expert_indices.size(); }
    const uint8_t* get_l3_resident_slice() const { return m_l3_resident_slice; }

private:
    uint8_t* m_pool = nullptr;
    size_t m_pool_size = 0;
    size_t m_bytes_loaded = 0;
    double m_load_time_sec = 0.0;
    size_t m_peak_ws_bytes = 0;

    // Pinned 10 MB Resident L3 Cache Slice
    uint8_t* m_l3_resident_slice = nullptr;
    std::unordered_set<int> m_hot_expert_indices;
    double m_hot_hit_rate = 0.0;

    // Model Architecture
    GrootLayer m_layers[NUM_LAYERS];
    const void* m_token_embd = nullptr;
    const float* m_output_norm = nullptr;
    const void* m_output_weight = nullptr;

    void update_peak_memory();
};

} // namespace groot
