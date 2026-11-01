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

// Standard MoE Layer Expert Dimensions & Byte Sizes (IQ3_XXS model)
constexpr size_t STD_GATE_BYTES = 1007616; // IQ2_S [2048, 1536]
constexpr size_t STD_UP_BYTES   = 1007616; // IQ2_S [2048, 1536]
constexpr size_t STD_DOWN_BYTES = 1351680; // IQ3_S [1536, 2048]
constexpr size_t STD_EXPERT_BYTES = STD_GATE_BYTES + STD_UP_BYTES + STD_DOWN_BYTES; // 3,366,912 bytes (~3.21 MB)
constexpr size_t STD_LAYER_BYTES  = N_EXPERTS * STD_EXPERT_BYTES;                   // 215,482,368 bytes (~205.5 MB)

// High-Precision Layers 7 & 8 Expert Dimensions & Byte Sizes (IQ3_XXS model)
constexpr size_t HP_GATE_BYTES = 1351680;  // IQ3_S [2048, 1536]
constexpr size_t HP_UP_BYTES   = 1351680;  // IQ3_S [2048, 1536]
constexpr size_t HP_DOWN_BYTES = 1671168;  // IQ4_XS [1536, 2048]
constexpr size_t HP_EXPERT_BYTES = HP_GATE_BYTES + HP_UP_BYTES + HP_DOWN_BYTES;    // 4,374,528 bytes (~4.17 MB)
constexpr size_t HP_LAYER_BYTES  = N_EXPERTS * HP_EXPERT_BYTES;                    // 279,969,792 bytes (~267.0 MB)

// Q3_K_XL Standard Expert Dimensions (36 layers)
constexpr size_t Q3_GATE_BYTES = 1204224; // IQ3_XXS [2048, 1536]
constexpr size_t Q3_UP_BYTES   = 1204224; // IQ3_XXS [2048, 1536]
constexpr size_t Q3_DOWN_BYTES = 1671168; // IQ4_XS [1536, 2048]
constexpr size_t Q3_EXPERT_BYTES = Q3_GATE_BYTES + Q3_UP_BYTES + Q3_DOWN_BYTES; // 4,079,616 bytes (~3.89 MB)
constexpr size_t Q3_LAYER_BYTES  = N_EXPERTS * Q3_EXPERT_BYTES;                   // 261,095,424 bytes (~249.0 MB)

// Q3_K_XL High-Precision Layers 7 & 8 (2 layers)
constexpr size_t Q3_HP_GATE_BYTES = 1671168; // IQ4_XS [2048, 1536]
constexpr size_t Q3_HP_UP_BYTES   = 1671168; // IQ4_XS [2048, 1536]
constexpr size_t Q3_HP_DOWN_BYTES = 2580480; // Q6_K   [1536, 2048]
constexpr size_t Q3_HP_EXPERT_BYTES = Q3_HP_GATE_BYTES + Q3_HP_UP_BYTES + Q3_HP_DOWN_BYTES; // 5,922,816 bytes (~5.65 MB)
constexpr size_t Q3_HP_LAYER_BYTES  = N_EXPERTS * Q3_HP_EXPERT_BYTES;                         // 379,060,224 bytes (~361.5 MB)

constexpr size_t Q3_TOTAL_EXPERT_POOL_BYTES = 36ULL * Q3_LAYER_BYTES + 2ULL * Q3_HP_LAYER_BYTES; // 10,157,555,712 bytes (9.46 GiB)

// Total Contiguous Expert Pool for IQ3_XXS model
constexpr size_t TOTAL_EXPERT_POOL_BYTES = 36ULL * STD_LAYER_BYTES + 2ULL * HP_LAYER_BYTES; // 8,317,304,832 bytes (7.81 GB)

// Router Matrix Pool: 38 layers * 64 experts * 2048 dim converted to FP16 (uint16_t)
constexpr size_t ROUTER_LAYER_FLOATS = N_EXPERTS * D_MODEL; // 131,072 floats
constexpr size_t ROUTER_LAYER_FP16_BYTES = ROUTER_LAYER_FLOATS * sizeof(uint16_t); // 262,144 bytes (256 KB)
constexpr size_t TOTAL_ROUTER_POOL_BYTES = N_MOE_LAYERS * ROUTER_LAYER_FP16_BYTES;  // 9,961,472 bytes (~9.73 MB)

// Stationary L3 Activation Buffers for 512-Token Prefill (Fits 8x inside Zen 3's 16 MB L3)
constexpr int MAX_PREFILL_TOKENS = 512;
constexpr size_t ACTIVATION_BUFFER_FLOATS = (size_t)MAX_PREFILL_TOKENS * D_MODEL; // 1,048,576
constexpr size_t ACTIVATION_BUFFER_BYTES  = ACTIVATION_BUFFER_FLOATS * sizeof(uint16_t); // 2,097,152 (2.0 MB)
constexpr size_t L3_ROUTER_PINGPONG_BYTES = 2ULL * ROUTER_LAYER_FP16_BYTES; // 524,288 bytes (512 KB)

// Reusable Scratch Buffer Size: 400 MB (fits all layer tensors in single pass)
constexpr size_t SCRATCH_BUFFER_BYTES = 400ULL * 1024ULL * 1024ULL;

// Quantization format enum
enum class QuantType {
    IQ2_S = 0,
    IQ3_S = 1,
    IQ4_XS = 2,
    FP32  = 3,
    FP16  = 4,
    IQ3_XXS = 5,
    Q6_K = 6
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

    // Stationary L3 Activation Buffers (Zero-Copy UMA, 64-byte aligned)
    uint16_t* get_act_in() { return m_act_in; }
    uint16_t* get_act_out() { return m_act_out; }
    const uint16_t* get_act_in() const { return m_act_in; }
    const uint16_t* get_act_out() const { return m_act_out; }
    void swap_activation_buffers() { std::swap(m_act_in, m_act_out); }

    // L3 Ping-Pong Router Buffers (512 KB total footprint)
    uint16_t* get_l3_router_active() { return m_l3_router_active; }
    uint16_t* get_l3_router_prefetch() { return m_l3_router_prefetch; }
    void prefetch_router_layer(int layer_idx, bool to_prefetch_buf);

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

    // Stationary L3 Buffers (Allocated aligned to 64 bytes)
    uint16_t* m_act_in = nullptr;
    uint16_t* m_act_out = nullptr;
    uint16_t* m_l3_router_active = nullptr;
    uint16_t* m_l3_router_prefetch = nullptr;

    uint8_t* m_scratch_buf = nullptr;
    size_t m_scratch_buf_size = 0;

    std::vector<PackedMoELayer> m_layers;
    size_t m_peak_ws_during_load = 0;

    void update_peak_working_set();
};

} // namespace zen3
