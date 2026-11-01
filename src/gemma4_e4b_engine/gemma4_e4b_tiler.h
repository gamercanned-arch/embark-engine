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
#include "gemma4_e4b_simd.h"

namespace gemma4_e4b {

constexpr int D_MODEL = 2560;
constexpr int D_FF = 10240;
constexpr int NUM_LAYERS = 42;
constexpr int HEAD_COUNT = 8;
constexpr int HEAD_COUNT_KV = 2;
constexpr int HEAD_DIM = 256;
constexpr int SLIDING_WINDOW = 512;
constexpr int SHARED_KV_LAYERS = 18;

// Staging scratch buffer: 250 MB
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL;

// Base memory pool size: ~5.3 GiB (covers raw 4.62 GiB + 60.15 MB expanded scales + alignments)
constexpr size_t EXPANDED_POOL_SIZE = 5600000000ULL;

struct Gemma4E4BLayer {
    int layer_idx = 0;

    // Attention weights & norms
    const float* attn_norm = nullptr;           // F32 [2560]
    const void*  attn_q = nullptr;              // Q4_K expanded [2560, 2048]
    const void*  attn_k = nullptr;              // Q4_K expanded [2560, 512]
    const void*  attn_v = nullptr;              // Q6_K [2560, 512]
    const void*  attn_output = nullptr;         // Q4_K expanded [2048, 2560]
    const float* attn_q_norm = nullptr;         // F32 [256]
    const float* attn_k_norm = nullptr;         // F32 [256]

    // FFN weights & norms
    const float* ffn_norm = nullptr;            // F32 [2560]
    const void*  ffn_gate = nullptr;            // Q4_K expanded [2560, 10240]
    const void*  ffn_up = nullptr;              // Q4_K expanded [2560, 10240]
    const void*  ffn_down = nullptr;            // Q6_K [10240, 2560]

    // Residual & projection layers
    const float* inp_gate = nullptr;            // F32 [2560, 256]
    const float* proj = nullptr;                // F32 [256, 2560]
    const float* post_attention_norm = nullptr; // F32 [2560]
    const float* post_ffw_norm = nullptr;       // F32 [2560]
    const float* post_norm = nullptr;           // F32 [2560]
    const float* layer_output_scale = nullptr;  // F32 [1]

    size_t layer_bytes = 0;
};

class Gemma4E4BTiler {
public:
    Gemma4E4BTiler();
    ~Gemma4E4BTiler();

    Gemma4E4BTiler(const Gemma4E4BTiler&) = delete;
    Gemma4E4BTiler& operator=(const Gemma4E4BTiler&) = delete;

    // Open GGUF with Win32 Direct-I/O sequential scan flags,
    // stage chunks into 250 MB scratch buffer,
    // pre-expand Q4_K scales into contiguous pool (+60.15 MB),
    // and map 42 layers.
    bool load_and_stage(const std::string& gguf_path);

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    size_t get_expanded_pool_bytes() const { return m_pool_offset; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    const Gemma4E4BLayer& get_layer(int idx) const { return m_layers[idx]; }
    const void* get_token_embd() const { return m_token_embd; }
    const float* get_output_norm() const { return m_output_norm; }
    const void* get_per_layer_token_embd() const { return m_per_layer_token_embd; }
    const void* get_per_layer_model_proj() const { return m_per_layer_model_proj; }

private:
    uint8_t* m_pool = nullptr;
    size_t   m_pool_size = EXPANDED_POOL_SIZE;
    uint8_t* m_scratch_buf = nullptr;
    size_t   m_scratch_size = SCRATCH_BUFFER_BYTES;

    size_t   m_pool_offset = 0;
    size_t   m_bytes_loaded = 0;
    double   m_load_time_sec = 0.0;
    size_t   m_peak_ws_bytes = 0;

    std::vector<Gemma4E4BLayer> m_layers;
    const void*  m_token_embd = nullptr;
    const float* m_output_norm = nullptr;
    const void*  m_per_layer_token_embd = nullptr;
    const void*  m_per_layer_model_proj = nullptr;

    void update_peak_memory();
    void prefault_pool();
};

} // namespace gemma4_e4b
