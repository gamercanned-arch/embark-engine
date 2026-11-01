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
#include "ornith_simd.h"

namespace ornith {

constexpr int D_MODEL = 4096;
constexpr int D_FF = 12288;
constexpr int NUM_LAYERS = 33;
constexpr int NUM_GDN_LAYERS = 24;
constexpr int NUM_ATTN_LAYERS = 9;

// Scratch buffer size: 250 MB
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL;

// Base memory pool size: ~5.59 GiB (covers raw 5.37 GiB + 102.81 MB expanded scales + alignments)
constexpr size_t EXPANDED_POOL_SIZE = 6000000000ULL;

struct OrnithLayer {
    int layer_idx = 0;
    bool is_attention = false; // true for 3, 7, 11, 15, 19, 23, 27, 31, 32

    // GDN-specific weights
    const void* attn_gate = nullptr;      // Q4_K expanded [4096, 4096]
    const void* attn_qkv = nullptr;       // Q6_K [4096, 8192]
    const float* ssm_conv1d = nullptr;    // F32 [4, 8192]
    const float* ssm_a = nullptr;         // F32 [32]
    const void* ssm_alpha = nullptr;      // Q4_K expanded [4096, 32]
    const void* ssm_beta = nullptr;       // Q4_K expanded [4096, 32]
    const float* ssm_dt_bias = nullptr;   // F32 [32]
    const float* ssm_norm = nullptr;      // F32 [128]
    const void* ssm_out = nullptr;        // Q4_K expanded [4096, 4096]

    // Full Attention-specific weights
    const void* attn_q = nullptr;         // Q4_K expanded [4096, 8192]
    const void* attn_k = nullptr;         // Q4_K expanded [4096, 1024]
    const void* attn_v = nullptr;         // Q6_K [4096, 1024]
    const void* attn_output = nullptr;    // Q4_K expanded [4096, 4096]
    const float* attn_q_norm = nullptr;   // F32 [256]
    const float* attn_k_norm = nullptr;   // F32 [256]

    // Shared layer weights & norms
    const float* attn_norm = nullptr;     // F32 [4096]
    const float* post_attn_norm = nullptr;// F32 [4096]
    const void* ffn_gate = nullptr;       // Q4_K expanded [4096, 12288]
    const void* ffn_up = nullptr;         // Q4_K expanded [4096, 12288]
    const void* ffn_down = nullptr;       // Q6_K [12288, 4096]

    size_t layer_bytes = 0;
};

class OrnithTiler {
public:
    OrnithTiler();
    ~OrnithTiler();

    OrnithTiler(const OrnithTiler&) = delete;
    OrnithTiler& operator=(const OrnithTiler&) = delete;

    // Open GGUF with Win32 Direct-I/O sequential scan flags,
    // stage chunks into 250 MB scratch buffer,
    // pre-expand Q4_K scales into contiguous pool (+102.8 MB),
    // and map GDN / Attention layers.
    bool load_and_stage(const std::string& gguf_path);

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    const OrnithLayer& get_layer(int idx) const { return m_layers[idx]; }
    const void* get_token_embd() const { return m_token_embd; }
    const float* get_output_norm() const { return m_output_norm; }
    const void* get_output_weight() const { return m_output_weight; }

    static bool is_full_attention_layer(int layer_idx) {
        return (layer_idx == 3 || layer_idx == 7 || layer_idx == 11 || layer_idx == 15 ||
                layer_idx == 19 || layer_idx == 23 || layer_idx == 27 || layer_idx == 31 ||
                layer_idx == 32);
    }

private:
    uint8_t* m_pool = nullptr;
    size_t   m_pool_size = EXPANDED_POOL_SIZE;
    uint8_t* m_scratch_buf = nullptr;
    size_t   m_scratch_size = SCRATCH_BUFFER_BYTES;

    size_t m_pool_offset = 0;
    size_t m_bytes_loaded = 0;
    size_t m_peak_ws_bytes = 0;
    double m_load_time_sec = 0.0;

    std::vector<OrnithLayer> m_layers;
    const void* m_token_embd = nullptr;
    const float* m_output_norm = nullptr;
    const void* m_output_weight = nullptr;

    void update_peak_memory();
    void prefault_pool();
};

} // namespace ornith
