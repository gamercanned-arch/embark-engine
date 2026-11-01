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
#include "spark_simd.h"

namespace spark {

constexpr int D_MODEL = 2560;
constexpr int D_FF = 10240;
constexpr int NUM_LAYERS = 36;
constexpr int HEAD_COUNT = 16;
constexpr int HEAD_COUNT_KV = 4;
constexpr int HEAD_DIM = 256;
constexpr int SLIDING_WINDOW = 512;

// Scratch buffer size: 250 MB
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL;

// Base memory pool size: ~4.0 GB (covers raw 3.30 GiB + 28.13 MB expanded scales + alignments)
constexpr size_t EXPANDED_POOL_SIZE = 4200000000ULL;

struct SparkLayer {
    int layer_idx = 0;

    // Layer weights & norms
    const float* attn_norm = nullptr;     // F32 [2560]
    const void*  attn_qkv = nullptr;      // Q6_K [2560, 6144]
    const void*  attn_gate = nullptr;     // BF16 [2560, 16]
    const void*  attn_output = nullptr;   // Q8_0 [4096, 2560]

    const float* ffn_norm = nullptr;      // F32 [2560]
    const void*  ffn_gate = nullptr;      // Q4_K expanded [2560, 10240]
    const void*  ffn_up = nullptr;        // Q4_K expanded [2560, 10240]
    const void*  ffn_down = nullptr;      // Q6_K [10240, 2560]

    size_t layer_bytes = 0;
};

class SparkTiler {
public:
    SparkTiler();
    ~SparkTiler();

    SparkTiler(const SparkTiler&) = delete;
    SparkTiler& operator=(const SparkTiler&) = delete;

    // Open GGUF with Win32 Direct-I/O sequential scan flags,
    // stage chunks into 250 MB scratch buffer,
    // pre-expand Q4_K scales into contiguous pool (+28.13 MB),
    // and map dense transformer layers.
    bool load_and_stage(const std::string& gguf_path);

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    const SparkLayer& get_layer(int idx) const { return m_layers[idx]; }
    const void* get_token_embd() const { return m_token_embd; }
    const float* get_output_norm() const { return m_output_norm; }
    const void* get_output_weight() const { return m_output_weight; }

private:
    uint8_t* m_pool = nullptr;
    size_t   m_pool_size = EXPANDED_POOL_SIZE;
    uint8_t* m_scratch_buf = nullptr;
    size_t   m_scratch_size = SCRATCH_BUFFER_BYTES;

    size_t   m_pool_offset = 0;
    size_t   m_bytes_loaded = 0;
    double   m_load_time_sec = 0.0;
    size_t   m_peak_ws_bytes = 0;

    std::vector<SparkLayer> m_layers;
    const void*  m_token_embd = nullptr;
    const float* m_output_norm = nullptr;
    const void*  m_output_weight = nullptr;

    void update_peak_memory();
    void prefault_pool();
};

} // namespace spark
