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
#include "gemma4_simd.h"

namespace gemma4 {

constexpr int D_MODEL = 3840;
constexpr int D_FF = 15360;
constexpr int NUM_LAYERS = 48;
constexpr int VOCAB_SIZE = 262144;

// Scratch buffer size: 250 MB
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL;

// Base memory pool sizes
// Original tensors: 7,024.34 MB (7,365,584,896 bytes)
// Pre-expanded scales: +139.0 MB (+145,744,384 bytes)
constexpr size_t EXPANDED_POOL_SIZE = 7511329280ULL; // ~7,163.34 MB (6.99 GiB)

struct Gemma4Layer {
    int layer_idx = 0;
    bool is_global = false; // Layers 5, 11, 17, 23, 29, 35, 41, 47

    int q_dim = 4096; // 4096 for SWA, 8192 for Global
    int k_dim = 2048; // 2048 for SWA, 512 for Global
    int v_dim = 2048; // 2048 for SWA, 0 for Global (TIED TO K)
    int o_dim = 4096; // 4096 for SWA, 8192 for Global

    // Attention weights
    const void* attn_q = nullptr; // Q4_K expanded
    const void* attn_k = nullptr; // Q4_K expanded
    const void* attn_v = nullptr; // Q4_K expanded or Q6_K; nullptr for Global (TIED)
    const void* attn_o = nullptr; // Q4_K expanded
    bool v_is_q6_k = false;

    // FFN weights
    const void* ffn_gate = nullptr; // Q4_K expanded [3840, 15360]
    const void* ffn_up   = nullptr; // Q4_K expanded [3840, 15360]
    const void* ffn_down = nullptr; // Q4_K expanded or Q6_K [15360, 3840]
    bool down_is_q6_k = false;

    // Norms
    const float* attn_norm = nullptr;
    const float* attn_q_norm = nullptr;
    const float* attn_k_norm = nullptr;
    const float* attn_o_norm = nullptr;
    const float* ffn_norm = nullptr;
    const float* post_ffw_norm = nullptr;
    const float* layer_output_scale = nullptr;

    size_t layer_bytes = 0;
};

class Gemma4Tiler {
public:
    Gemma4Tiler();
    ~Gemma4Tiler();

    Gemma4Tiler(const Gemma4Tiler&) = delete;
    Gemma4Tiler& operator=(const Gemma4Tiler&) = delete;

    // Open GGUF with Win32 Direct-I/O sequential scan flags,
    // stage chunks into 250 MB scratch buffer,
    // pre-expand Q4_K scales into 6.99 GiB contiguous pool (+139 MB),
    // and map tied K/V projections on global layers.
    bool load_and_stage(const std::string& gguf_path);

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    const Gemma4Layer& get_layer(int idx) const { return m_layers[idx]; }
    const void* get_token_embd() const { return m_token_embd; }
    const float* get_output_norm() const { return m_output_norm; }

private:
    uint8_t* m_pool = nullptr;
    size_t   m_pool_size = EXPANDED_POOL_SIZE;
    uint8_t* m_scratch_buf = nullptr;
    size_t   m_scratch_size = SCRATCH_BUFFER_BYTES;

    size_t m_pool_offset = 0;
    size_t m_bytes_loaded = 0;
    size_t m_peak_ws_bytes = 0;
    double m_load_time_sec = 0.0;

    std::vector<Gemma4Layer> m_layers;
    const void* m_token_embd = nullptr;
    const float* m_output_norm = nullptr;

    void update_peak_memory();
    void prefault_pool();
};

} // namespace gemma4
