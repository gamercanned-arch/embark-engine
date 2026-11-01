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
#include "embed_simd.h"

namespace embed_gemma {

constexpr int D_MODEL = 512;
constexpr int D_FF = 2048;
constexpr int D_OUT = 768;
constexpr int NUM_LAYERS = 24;
constexpr int NUM_HEADS = 4;
constexpr int SLIDING_WINDOW = 1024;

// Vision Model Constants
constexpr int VISION_D_MODEL = 768;
constexpr int VISION_D_FF = 3072;
constexpr int VISION_NUM_LAYERS = 16;
constexpr int VISION_NUM_HEADS = 12;
constexpr int VISION_IMAGE_SIZE = 224;
constexpr int VISION_PATCH_SIZE = 16;
constexpr int VISION_NUM_PATCHES = (VISION_IMAGE_SIZE / VISION_PATCH_SIZE) * (VISION_IMAGE_SIZE / VISION_PATCH_SIZE); // 196

// Direct-I/O Staging Scratch Buffer: 250 MB
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL;

// Base Contiguous Memory Pool: ~1.40 GB (fits 222 MB text + 935 MB vision + alignments)
constexpr size_t EXPANDED_POOL_SIZE = 1450000000ULL;

struct EmbedGemmaLayer {
    int layer_idx = 0;
    bool is_sliding_window = true; // true for 0..4, 6..10, false for 5, 11, 17, 23
    int q_heads = 4;
    int kv_heads = 2;              // 2 for SWA, 1 for global
    int head_dim = 256;            // 256 for SWA (1024 total), 512 for global (2048 total)
    int q_dim = 1024;              // 1024 or 2048
    int kv_dim = 512;              // 512

    // Attention weights & norms
    const float* attn_norm = nullptr;          // F32 [512]
    const void*  attn_q = nullptr;             // Q6_K [512, q_dim]
    const void*  attn_k = nullptr;             // Q6_K [512, 512]
    const void*  attn_v = nullptr;             // Q6_K [512, 512]
    const void*  attn_output = nullptr;        // Q6_K [q_dim, 512]
    const float* attn_q_norm = nullptr;        // F32 [head_dim]
    const float* attn_k_norm = nullptr;        // F32 [head_dim]
    const float* post_attention_norm = nullptr;// F32 [512]

    // FeedForward (GeGLU) weights & norms
    const float* ffn_norm = nullptr;           // F32 [512]
    const void*  ffn_gate = nullptr;           // Q6_K [512, 2048]
    const void*  ffn_up = nullptr;             // Q6_K [512, 2048]
    const void*  ffn_down = nullptr;           // Q6_K [2048, 512]
    const float* post_ffw_norm = nullptr;      // F32 [512]

    // Per-layer projections
    const void*  inp_gate = nullptr;           // Q8_0 [512, 512]
    const void*  proj = nullptr;               // Q8_0 [512, 512]
    const float* post_norm = nullptr;          // F32 [512]

    // Layer output scalar
    float layer_output_scale = 1.0f;           // F32 [1]

    size_t layer_bytes = 0;
};

struct VisionLayer {
    int layer_idx = 0;
    const float* ln1_weight = nullptr;         // F32 [768]
    const void*  attn_q = nullptr;             // F16 [768, 768]
    const void*  attn_k = nullptr;             // F16 [768, 768]
    const void*  attn_v = nullptr;             // F16 [768, 768]
    const void*  attn_out = nullptr;           // F16 [768, 768]
    const float* ln2_weight = nullptr;         // F32 [768]
    const void*  ffn_gate = nullptr;           // F16 [768, 3072]
    const void*  ffn_up = nullptr;             // F16 [768, 3072]
    const void*  ffn_down = nullptr;           // F16 [3072, 768]
};

class EmbedTiler {
public:
    EmbedTiler();
    ~EmbedTiler();

    EmbedTiler(const EmbedTiler&) = delete;
    EmbedTiler& operator=(const EmbedTiler&) = delete;

    // Direct-I/O staging loader for text backbone + mmproj vision projector
    bool load_and_stage(const std::string& text_gguf_path, const std::string& mmproj_gguf_path = "");

    // Getters
    const uint8_t* get_pool() const { return m_pool; }
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_bytes_loaded() const { return m_bytes_loaded; }
    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

    // Text model tensors
    const EmbedGemmaLayer& get_layer(int idx) const { return m_layers[idx]; }
    const void*  get_token_embd() const { return m_token_embd; }
    const void*  get_per_layer_model_proj() const { return m_per_layer_model_proj; }
    const float* get_per_layer_proj_norm() const { return m_per_layer_proj_norm; }
    const float* get_output_norm() const { return m_output_norm; }
    const void*  get_output_proj() const { return m_output_proj; }

    // Vision model tensors
    bool has_vision() const { return m_has_vision; }
    const void*  get_vision_patch_embd() const { return m_vision_patch_embd; }
    const float* get_vision_position_embd() const { return m_vision_position_embd; }
    const void*  get_mm_input_projection() const { return m_mm_input_projection; }
    const VisionLayer& get_vision_layer(int idx) const { return m_vision_layers[idx]; }

private:
    uint8_t* m_pool = nullptr;
    size_t   m_pool_size = 0;
    size_t   m_pool_offset = 0;

    uint8_t* m_scratch_buf = nullptr;
    size_t   m_scratch_size = 0;

    size_t   m_bytes_loaded = 0;
    double   m_load_time_sec = 0.0;
    size_t   m_peak_ws_bytes = 0;

    // Model structures
    std::vector<EmbedGemmaLayer> m_layers;
    const void*  m_token_embd = nullptr;
    const void*  m_per_layer_model_proj = nullptr;
    const float* m_per_layer_proj_norm = nullptr;
    const float* m_output_norm = nullptr;
    const void*  m_output_proj = nullptr;

    // Vision structures
    bool m_has_vision = false;
    const void*  m_vision_patch_embd = nullptr;
    const float* m_vision_position_embd = nullptr;
    const void*  m_mm_input_projection = nullptr;
    std::vector<VisionLayer> m_vision_layers;

    void update_peak_memory();
    void prefault_pool();
    bool stage_text_gguf(const std::string& path);
    bool stage_vision_gguf(const std::string& path);
};

} // namespace embed_gemma
