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
#include "lfm2_simd.h"

namespace lfm2 {

// Base contiguous pool: ~450 MB (Model raw weights ~379 MB + aligned offsets)
constexpr size_t LFM2_POOL_SIZE = 450ULL * 1024ULL * 1024ULL; // 471,859,200 bytes
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL;

enum class LayerType {
    SHORTCONV,
    ATTENTION
};

struct LFM2Layer {
    int layer_idx = 0;
    LayerType type = LayerType::SHORTCONV;
    int sub_idx = 0; // index into shortconv (0..9) or attn (0..5)

    // Common RMSNorms
    const float* attn_norm = nullptr; // F32 [1024]
    const float* ffn_norm = nullptr;  // F32 [1024]

    // Common Dense FFN (all Q8_0)
    const block_q8_0* ffn_gate = nullptr; // [4608, 1024]
    const block_q8_0* ffn_up = nullptr;   // [4608, 1024]
    const block_q8_0* ffn_down = nullptr; // [1024, 4608]

    // ShortConv specific (10 layers)
    const block_q8_0* shortconv_in_proj = nullptr;  // [3072, 1024]
    const float*      shortconv_conv = nullptr;     // F32 [3, 1024]
    const block_q8_0* shortconv_out_proj = nullptr; // [1024, 1024]

    // Attention specific (6 layers)
    const block_q8_0* attn_q = nullptr;        // [1024, 1024]
    const float*      attn_q_norm = nullptr;   // F32 [64]
    const block_q8_0* attn_k = nullptr;        // [512, 1024]
    const float*      attn_k_norm = nullptr;   // F32 [64]
    const block_q8_0* attn_v = nullptr;        // [512, 1024]
    const block_q8_0* attn_output = nullptr;   // [1024, 1024]

    size_t layer_bytes = 0;
};

class LFM2Tiler {
public:
    LFM2Tiler();
    ~LFM2Tiler();

    LFM2Tiler(const LFM2Tiler&) = delete;
    LFM2Tiler& operator=(const LFM2Tiler&) = delete;

    // Open GGUF with Win32 Direct-I/O sequential scan flags,
    // stage chunks into 250 MB scratch buffer,
    // allocate 450 MB contiguous memory pool,
    // and map all 16 layers (10 ShortConv + 6 Attention) + global weights.
    bool load_and_stage(const std::string& gguf_path);

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    const LFM2Layer& get_layer(int idx) const { return m_layers[idx]; }
    const block_q8_0* get_token_embd() const { return m_token_embd; }
    const float* get_output_norm() const { return m_output_norm; }

private:
    uint8_t* m_pool = nullptr;
    size_t m_pool_size = 0;
    size_t m_pool_offset = 0;

    uint8_t* m_scratch_buf = nullptr;
    size_t m_scratch_size = 0;

    std::vector<LFM2Layer> m_layers;
    const block_q8_0* m_token_embd = nullptr;
    const float* m_output_norm = nullptr;

    size_t m_bytes_loaded = 0;
    double m_load_time_sec = 0.0;
    size_t m_peak_ws_bytes = 0;

    void update_peak_memory();
    void prefault_pool();
};

} // namespace lfm2
