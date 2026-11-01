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

namespace zen3 {

// Architectural Constants for LFM 2.5 24B A2B (lfm2moe)
constexpr int D_MODEL = 2048;
constexpr int N_EXPERTS = 64;
constexpr int TOP_K = 4;
constexpr int N_MOE_LAYERS = 38;
constexpr int EXPERT_FFN_DIM = 1536;

// Standard MoE Layer Expert Dimensions & Byte Sizes (36 layers)
constexpr size_t STD_GATE_BYTES = 1007616; // IQ2_S [2048, 1536]
constexpr size_t STD_UP_BYTES   = 1007616; // IQ2_S [2048, 1536]
constexpr size_t STD_DOWN_BYTES = 1351680; // IQ3_S [1536, 2048]
constexpr size_t STD_EXPERT_BYTES = STD_GATE_BYTES + STD_UP_BYTES + STD_DOWN_BYTES; // 3,366,912 bytes (~3.21 MB)
constexpr size_t STD_LAYER_BYTES  = N_EXPERTS * STD_EXPERT_BYTES;                   // 215,482,368 bytes (~205.5 MB)

// High-Precision Layers 7 & 8 Expert Dimensions & Byte Sizes (2 layers)
constexpr size_t HP_GATE_BYTES = 1351680;  // IQ3_S [2048, 1536]
constexpr size_t HP_UP_BYTES   = 1351680;  // IQ3_S [2048, 1536]
constexpr size_t HP_DOWN_BYTES = 1671168;  // IQ4_XS [1536, 2048]
constexpr size_t HP_EXPERT_BYTES = HP_GATE_BYTES + HP_UP_BYTES + HP_DOWN_BYTES;    // 4,374,528 bytes (~4.17 MB)
constexpr size_t HP_LAYER_BYTES  = N_EXPERTS * HP_EXPERT_BYTES;                    // 279,969,792 bytes (~267.0 MB)

// Total Contiguous Expert Pool: 36 standard + 2 high-precision layers
constexpr size_t TOTAL_EXPERT_POOL_BYTES = 36ULL * STD_LAYER_BYTES + 2ULL * HP_LAYER_BYTES; // 8,317,304,832 bytes (7.81 GB)

// Router Matrix Pool: 38 layers * 64 experts * 2048 dim converted to FP16 (uint16_t)
constexpr size_t ROUTER_LAYER_FLOATS = N_EXPERTS * D_MODEL; // 131,072 floats
constexpr size_t ROUTER_LAYER_FP16_BYTES = ROUTER_LAYER_FLOATS * sizeof(uint16_t); // 262,144 bytes (256 KB)
constexpr size_t TOTAL_ROUTER_POOL_BYTES = N_MOE_LAYERS * ROUTER_LAYER_FP16_BYTES;  // 9,961,472 bytes (~9.73 MB)

// Reusable Scratch Buffer Size: 250 MB
constexpr size_t SCRATCH_BUFFER_BYTES = 250ULL * 1024ULL * 1024ULL; // 262,144,000 bytes

// Quantization format enum
enum class QuantType {
    IQ2_S = 0,
    IQ3_S = 1,
    IQ4_XS = 2,
    FP32  = 3,
    FP16  = 4
};

// Contiguous Expert slice layout
struct PackedExpert {
    const uint8_t* expert_base;
    const uint8_t* gate_ptr;
    const uint8_t* up_ptr;
    const uint8_t* down_ptr;
    size_t gate_bytes;
    size_t up_bytes;
    size_t down_bytes;
    size_t total_bytes;
    QuantType gate_type;
    QuantType up_type;
    QuantType down_type;
};

// MoE Layer Descriptor with pointers into the contiguous pool and L3 router buffer
struct PackedMoELayer {
    int model_layer_idx;     // 2 .. 39
    int moe_idx;             // 0 .. 37
    size_t layer_pool_offset;
    size_t expert_bytes;
    size_t layer_total_bytes;
    PackedExpert experts[N_EXPERTS];
    const uint16_t* router_fp16; // Pinned in L3 cache (256 KB per layer)
};

// Chunked Staging Loader & Contiguous Tiler
class LFM2Tiler {
public:
    LFM2Tiler();
    ~LFM2Tiler();

    // Prevent copying
    LFM2Tiler(const LFM2Tiler&) = delete;
    LFM2Tiler& operator=(const LFM2Tiler&) = delete;

    // Open GGUF with Win32 FILE_FLAG_SEQUENTIAL_SCAN, allocate 7.81 GB contiguous pool,
    // tile layer-by-layer through 250 MB reusable scratch buffer, and convert router weights to FP16.
    bool load_and_tile(const std::string& gguf_path);

    // Memory pointers
    const uint8_t* get_expert_pool() const { return m_expert_pool; }
    size_t get_expert_pool_size() const { return m_expert_pool_size; }

    const uint16_t* get_router_buffer() const { return m_router_fp16_pool; }
    size_t get_router_buffer_size() const { return m_router_pool_size; }

    const PackedMoELayer& get_layer(int moe_layer_idx) const { return m_layers[moe_layer_idx]; }
    int get_num_layers() const { return N_MOE_LAYERS; }

    // Peak working set tracking
    size_t get_peak_working_set_bytes() const { return m_peak_ws_during_load; }
    double get_peak_working_set_gb() const {
        return (double)m_peak_ws_during_load / (1024.0 * 1024.0 * 1024.0);
    }

    // Touch all router FP16 weights to warm/pin them in Zen 3's 16 MB L3 cache
    void warm_router_l3_cache();

    // Prefault pool pages to ensure stable OS page tables and zero page-fault jitter
    void prefault_pool();

private:
    uint8_t* m_expert_pool = nullptr;
    size_t m_expert_pool_size = 0;

    uint16_t* m_router_fp16_pool = nullptr;
    size_t m_router_pool_size = 0;

    uint8_t* m_scratch_buf = nullptr;
    size_t m_scratch_buf_size = 0;

    std::vector<PackedMoELayer> m_layers;
    size_t m_peak_ws_during_load = 0;

    void update_peak_working_set();
};

} // namespace zen3
