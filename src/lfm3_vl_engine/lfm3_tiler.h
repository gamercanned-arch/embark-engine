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
#include "lfm3_simd.h"

namespace lfm3 {

constexpr int D_MODEL = 2048;
constexpr int D_FF = 7680;
constexpr int NUM_LAYERS = 24;
constexpr int NUM_SHORTCONV_LAYERS = 13;
constexpr int NUM_ATTN_LAYERS = 11;
constexpr int HEAD_COUNT = 16;
constexpr int HEAD_COUNT_KV = 4;
constexpr int HEAD_DIM = 128;
constexpr int SLIDING_WINDOW = 512;

// Scratch buffer size: 250 MB for chunked Direct-I/O staging
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL;

// Base contiguous pool: ~1.35 GB (covers raw 1.02 GiB + 19.88 MB pre-expanded scales + alignments)
constexpr size_t EXPANDED_POOL_SIZE = 1350000000ULL;

enum class LayerType {
    SHORTCONV,
    ATTENTION
};

struct LFM3Layer {
    int layer_idx = 0;
    LayerType type = LayerType::SHORTCONV;
    int sub_idx = 0; // index into shortconv (0..12) or attn (0..10)

    // Common RMSNorms
    const float* attn_norm = nullptr;          // F32 [2048]
    const float* post_attention_norm = nullptr; // F32 [2048]
    const float* ffn_norm = nullptr;           // F32 [2048]
    const float* post_ffw_norm = nullptr;      // F32 [2048]

    // Common FFN
    const void*  ffn_gate = nullptr;           // Q4_K expanded [2048, 7680]
    const void*  ffn_up = nullptr;             // Q4_K expanded [2048, 7680]
    const void*  ffn_down = nullptr;           // Q6_K [7680, 2048]

    // ShortConv specific (13 layers)
    const void*  shortconv_in_proj = nullptr;  // Q4_K expanded [2048, 6144]
    const float* shortconv_conv = nullptr;     // F32 [7, 2048]
    const void*  shortconv_out_proj = nullptr; // Q4_K expanded [2048, 2048]

    // Attention specific (11 layers)
    const void*  attn_q = nullptr;             // Q4_K expanded [2048, 2048]
    const float* attn_q_norm = nullptr;        // F32 [128]
    const void*  attn_k = nullptr;             // Q4_K expanded [2048, 512]
    const float* attn_k_norm = nullptr;        // F32 [128]
    const void*  attn_v = nullptr;             // Q6_K [2048, 512]
    const void*  attn_gate = nullptr;          // Q4_K expanded [2048, 16]
    const void*  attn_output = nullptr;        // Q4_K expanded [2048, 2048]

    size_t layer_bytes = 0;
};

class LFM3Tiler {
public:
    LFM3Tiler();
    ~LFM3Tiler();

    LFM3Tiler(const LFM3Tiler&) = delete;
    LFM3Tiler& operator=(const LFM3Tiler&) = delete;

    // Open GGUF with Win32 Direct-I/O sequential scan flags,
    // stage chunks into 250 MB scratch buffer,
    // pre-expand Q4_K scales into contiguous pool (+19.88 MB),
    // and map all 24 layers (13 ShortConv + 11 Attention).
    bool load_and_stage(const std::string& gguf_path);

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    const LFM3Layer& get_layer(int idx) const { return m_layers[idx]; }
    const void* get_token_embd() const { return m_token_embd; }
    const float* get_output_norm() const { return m_output_norm; }
    const float* get_rope_freqs() const { return m_rope_freqs; }

private:
    uint8_t* m_pool = nullptr;
    size_t   m_pool_size = EXPANDED_POOL_SIZE;
    uint8_t* m_scratch_buf = nullptr;
    size_t   m_scratch_size = SCRATCH_BUFFER_BYTES;

    size_t   m_pool_offset = 0;
    size_t   m_bytes_loaded = 0;
    double   m_load_time_sec = 0.0;
    size_t   m_peak_ws_bytes = 0;

    std::vector<LFM3Layer> m_layers;
    const void*  m_token_embd = nullptr;
    const float* m_output_norm = nullptr;
    const float* m_rope_freqs = nullptr;

    void update_peak_memory();
    void prefault_pool();
};

} // namespace lfm3
