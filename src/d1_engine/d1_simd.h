#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

namespace d1 {

#ifndef QK_K
#define QK_K 256
#endif

#pragma pack(push, 1)

// Standard GGUF Q4_K Block (144 bytes)
struct block_q4_K {
    uint16_t d;          // fp16 scale
    uint16_t dmin;       // fp16 min
    uint8_t  scales[12]; // 6-bit packed scales and mins
    uint8_t  qs[128];    // 256 4-bit nibbles
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

// Pre-expanded Q4_K Block with aligned 8-bit scales and mins (+4 bytes = 148 bytes)
// Eliminates scalar bit shifts during inner dot-product execution
struct block_q4_K_expanded {
    uint16_t d;           // fp16 scale
    uint16_t dmin;        // fp16 min
    uint8_t  d_scales[8]; // 8 unpacked scales (0..63)
    uint8_t  m_scales[8]; // 8 unpacked minimums (0..63)
    uint8_t  qs[128];     // 256 4-bit nibbles
};
static_assert(sizeof(block_q4_K_expanded) == 148, "block_q4_K_expanded must be 148 bytes");

// Standard GGUF Q6_K Block (210 bytes)
struct block_q6_K {
    uint8_t  ql[128];    // lower 4 bits
    uint8_t  qh[64];     // upper 2 bits
    int8_t   scales[16]; // signed 8-bit scales
    uint16_t d;          // fp16 super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

// Standard GGUF Q8_0 Block (34 bytes)
struct block_q8_0 {
    uint16_t d;       // fp16 scale
    int8_t   qs[32];  // 32 int8 values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// Activation Quantization Block Q8_K (292 bytes)
struct block_q8_K {
    float   d;              // scale
    int8_t  qs[QK_K];       // 256 int8 quantized values
    int16_t bsums[QK_K/16]; // 16 int16 block sums
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");

#pragma pack(pop)

// Hardware F16C to FP32 conversion
static inline float fp16_to_fp32(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

// Hardware FP32 to F16C conversion
static inline uint16_t fp32_to_fp16(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, 0);
    return (uint16_t)_mm_cvtsi128_si32(h);
}

// Horizontal sum of 8 floats in __m256
static inline float hsum_float_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

// Non-temporal streaming load with MOVNTDQA when 32-byte aligned to bypass L3 pollution
static inline __m256i load_streaming_256(const void* ptr) {
    if (((uintptr_t)ptr & 31) == 0) {
        return _mm256_stream_load_si256((const __m256i*)ptr);
    } else {
        _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
        return _mm256_loadu_si256((const __m256i*)ptr);
    }
}

// Unpack a raw Q4_K block into pre-expanded block with aligned 8-bit uint8_t scales
static inline void expand_q4_k_block(const block_q4_K* src, block_q4_K_expanded* dst) {
    dst->d = src->d;
    dst->dmin = src->dmin;
    const uint8_t* q = src->scales;

    // Decode 8 scales and 8 mins
    for (int j = 0; j < 4; ++j) {
        dst->d_scales[j] = q[j] & 63;
        dst->m_scales[j] = q[j + 4] & 63;
    }
    for (int j = 4; j < 8; ++j) {
        dst->d_scales[j] = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        dst->m_scales[j] = (q[j + 4] >> 4)   | ((q[j] >> 6) << 4);
    }
    memcpy(dst->qs, src->qs, 128);
}

// Quantize activation row to Q8_K
static inline void quantize_row_q8_k_avx2(const float* __restrict x, block_q8_K* __restrict y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float* xi = x + i * QK_K;
        float amax = 0.0f;
        for (int l = 0; l < QK_K; ++l) {
            float v = std::fabs(xi[l]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 127.0f / amax : 0.0f;
        y[i].d = d;

        for (int j = 0; j < 16; ++j) {
            int16_t sum = 0;
            for (int l = 0; l < 16; ++l) {
                int val = (int)std::round(xi[j * 16 + l] * id);
                val = (std::max)(-128, (std::min)(127, val));
                y[i].qs[j * 16 + l] = (int8_t)val;
                sum += (int16_t)val;
            }
            y[i].bsums[j] = sum;
        }
    }
}

// Shuffle tables for Q4_K
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

// -----------------------------------------------------------------------------
// AVX2 Fused Q4_K Dot Product using Pre-Expanded Aligned Scales & Non-Temporal Streaming
// Completely eliminates scalar 6-bit unpacking shifts in the inner loop
// -----------------------------------------------------------------------------
static inline float vec_dot_q4_k_expanded_q8_k_zen3(
    const block_q4_K_expanded* __restrict x,
    const block_q8_K*          __restrict y,
    int k)
{
    const int nb = k / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0x0F);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*)(x + i + 2), _MM_HINT_NTA);
            _mm_prefetch((const char*)(x + i + 2)->qs + 64, _MM_HINT_NTA);
            _mm_prefetch((const char*)(y + i + 2), _MM_HINT_T0);
            _mm_prefetch((const char*)(y + i + 2)->qs + 64, _MM_HINT_T0);
        }

        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        // Directly load pre-expanded 8-byte scales and 8-byte minimums
        // Zero scalar bit shifts!
        __m128i sc_raw = _mm_loadl_epi64((const __m128i*)x[i].d_scales);
        __m128i mn_raw = _mm_loadl_epi64((const __m128i*)x[i].m_scales);

        __m256i scales_16 = _mm256_cvtepu8_epi16(sc_raw);
        __m128i mins_16   = _mm_cvtepu8_epi16(mn_raw);

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(mins_16, q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_castsi256_si128(scales_16);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            // Non-temporal streaming weight load to bypass L3 pollution
            const __m256i q4bits = load_streaming_256(q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8);
            const __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            q8 += 64;

            const __m256i dot_l = _mm256_maddubs_epi16(q4l, q8l);
            const __m256i dot_h = _mm256_maddubs_epi16(q4h, q8h);

            const __m256i p16l = _mm256_madd_epi16(scale_l, dot_l);
            const __m256i p16h = _mm256_madd_epi16(scale_h, dot_h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    __m128 acc_tot = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    acc_tot = _mm_add_ps(acc_tot, acc_m);
    acc_tot = _mm_add_ps(acc_tot, _mm_movehl_ps(acc_tot, acc_tot));
    acc_tot = _mm_add_ss(acc_tot, _mm_shuffle_ps(acc_tot, acc_tot, 0x55));
    return _mm_cvtss_f32(acc_tot);
}

// -----------------------------------------------------------------------------
// AVX2 Fused Q6_K Dot Product with Non-Temporal Streaming Weight Loads
// -----------------------------------------------------------------------------
static inline float vec_dot_q6_k_q8_k_zen3(
    const block_q6_K* __restrict x,
    const block_q8_K* __restrict y,
    int k)
{
    const int nb = k / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*)(x + i + 2), _MM_HINT_NTA);
            _mm_prefetch((const char*)(y + i + 2), _MM_HINT_T0);
        }

        const float d = y[i].d * fp16_to_fp32(x[i].d);

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        const int8_t*  q8 = y[i].qs;

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < 4; ++j) {
            const __m256i ql_bytes = load_streaming_256(ql); ql += 32;
            const __m256i qh_bytes = _mm256_set1_epi64x(*(const int64_t*)(qh + j * 8));

            const __m256i ql0 = _mm256_and_si256(ql_bytes, m15);
            const __m256i ql1 = _mm256_and_si256(_mm256_srli_epi16(ql_bytes, 4), m15);

            const __m256i qh0 = _mm256_and_si256(qh_bytes, m3);
            const __m256i qh1 = _mm256_and_si256(_mm256_srli_epi16(qh_bytes, 2), m3);

            const __m256i w0 = _mm256_sub_epi8(_mm256_or_si256(ql0, _mm256_slli_epi16(qh0, 4)), _mm256_set1_epi8(32));
            const __m256i w1 = _mm256_sub_epi8(_mm256_or_si256(ql1, _mm256_slli_epi16(qh1, 4)), _mm256_set1_epi8(32));

            const __m256i a0 = _mm256_loadu_si256((const __m256i*)q8);
            const __m256i a1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            q8 += 64;

            const __m256i d0 = _mm256_maddubs_epi16(w0, a0);
            const __m256i d1 = _mm256_maddubs_epi16(w1, a1);

            const __m256i scale = _mm256_set1_epi16(sc[j * 2]);
            const __m256i p0 = _mm256_madd_epi16(scale, d0);
            const __m256i p1 = _mm256_madd_epi16(scale, d1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return hsum_float_8(acc);
}

// -----------------------------------------------------------------------------
// Fused ShortConv AVX2 Kernel
// Circular Shift + Conv1D (K=7) + FMA + Gating in registers
// 13 layers x 2048 channels. State is 6 history values per channel.
// Total state across 13 layers: 13 * 6 * 2048 * 2B = 319.5 KB (FP16) / 373 KB (with 7 slots).
// 100% sits resident inside the 16 MB unified L3 cache with 0 DRAM roundtrips!
// -----------------------------------------------------------------------------
struct D1ShortConvState {
    static constexpr int NUM_SHORTCONV_LAYERS = 13;
    static constexpr int CHANNELS = 2048;
    static constexpr int KERNEL_SIZE = 7;
    static constexpr int HISTORY_LEN = KERNEL_SIZE - 1; // 6

    // State buffer: [13 layers][6 history slots][2048 channels] in FP16
    // Stored as FP16 to sit resident inside 16 MB L3 cache with 0 DRAM roundtrips!
    uint16_t state[NUM_SHORTCONV_LAYERS][HISTORY_LEN][CHANNELS];

    D1ShortConvState() {
        memset(state, 0, sizeof(state));
    }

    // Fused AVX2 Kernel:
    // Computes:
    // 1. Elementwise bx = b * x (where b, x each have length 2048)
    // 2. Conv1D with K=7 across 2048 channels:
    //    conv_out[ch] = sum_{k=0..5} (kernel[k, ch] * state[k, ch]) + kernel[6, ch] * bx[ch]
    // 3. Circular register shift: state[0..4] <- state[1..5], state[5] <- bx
    // 4. Output gating: out[ch] = c[ch] * conv_out[ch]
    void forward_fused_avx2(
        int layer_conv_idx,
        const float* __restrict b,
        const float* __restrict c,
        const float* __restrict x,
        const float* __restrict kernel_weights, // [7, 2048]
        float*       __restrict out)
    {
        uint16_t* lyr_state = &state[layer_conv_idx][0][0];

        // Process 8 channels per loop iteration (2048 / 8 = 256 iterations)
        for (int ch = 0; ch < CHANNELS; ch += 8) {
            __m256 vb = _mm256_loadu_ps(b + ch);
            __m256 vx = _mm256_loadu_ps(x + ch);
            __m256 vc = _mm256_loadu_ps(c + ch);

            // bx = b * x
            __m256 v_bx = _mm256_mul_ps(vb, vx);

            // Load 6 history vectors from L3-resident state
            __m128i h0 = _mm_loadu_si128((const __m128i*)&lyr_state[0 * CHANNELS + ch]);
            __m128i h1 = _mm_loadu_si128((const __m128i*)&lyr_state[1 * CHANNELS + ch]);
            __m128i h2 = _mm_loadu_si128((const __m128i*)&lyr_state[2 * CHANNELS + ch]);
            __m128i h3 = _mm_loadu_si128((const __m128i*)&lyr_state[3 * CHANNELS + ch]);
            __m128i h4 = _mm_loadu_si128((const __m128i*)&lyr_state[4 * CHANNELS + ch]);
            __m128i h5 = _mm_loadu_si128((const __m128i*)&lyr_state[5 * CHANNELS + ch]);

            __m256 s0 = _mm256_cvtph_ps(h0);
            __m256 s1 = _mm256_cvtph_ps(h1);
            __m256 s2 = _mm256_cvtph_ps(h2);
            __m256 s3 = _mm256_cvtph_ps(h3);
            __m256 s4 = _mm256_cvtph_ps(h4);
            __m256 s5 = _mm256_cvtph_ps(h5);

            // Load 7 kernel weights for these 8 channels
            __m256 w0 = _mm256_loadu_ps(kernel_weights + 0 * CHANNELS + ch);
            __m256 w1 = _mm256_loadu_ps(kernel_weights + 1 * CHANNELS + ch);
            __m256 w2 = _mm256_loadu_ps(kernel_weights + 2 * CHANNELS + ch);
            __m256 w3 = _mm256_loadu_ps(kernel_weights + 3 * CHANNELS + ch);
            __m256 w4 = _mm256_loadu_ps(kernel_weights + 4 * CHANNELS + ch);
            __m256 w5 = _mm256_loadu_ps(kernel_weights + 5 * CHANNELS + ch);
            __m256 w6 = _mm256_loadu_ps(kernel_weights + 6 * CHANNELS + ch);

            // FMA dot product in registers
            __m256 acc = _mm256_mul_ps(w0, s0);
            acc = _mm256_fmadd_ps(w1, s1, acc);
            acc = _mm256_fmadd_ps(w2, s2, acc);
            acc = _mm256_fmadd_ps(w3, s3, acc);
            acc = _mm256_fmadd_ps(w4, s4, acc);
            acc = _mm256_fmadd_ps(w5, s5, acc);
            acc = _mm256_fmadd_ps(w6, v_bx, acc);

            // Circular shift update in L3 cache
            _mm_storeu_si128((__m128i*)&lyr_state[0 * CHANNELS + ch], h1);
            _mm_storeu_si128((__m128i*)&lyr_state[1 * CHANNELS + ch], h2);
            _mm_storeu_si128((__m128i*)&lyr_state[2 * CHANNELS + ch], h3);
            _mm_storeu_si128((__m128i*)&lyr_state[3 * CHANNELS + ch], h4);
            _mm_storeu_si128((__m128i*)&lyr_state[4 * CHANNELS + ch], h5);
            _mm_storeu_si128((__m128i*)&lyr_state[5 * CHANNELS + ch], _mm256_cvtps_ph(v_bx, 0));

            // Gated output: out = c * acc
            __m256 v_out = _mm256_mul_ps(vc, acc);
            _mm256_storeu_ps(out + ch, v_out);
        }
    }
};

// -----------------------------------------------------------------------------
// d1-2B Active Layer Sliding Window KV Cache for 11 Attention Layers
// Architecture:
// - 11 Attention Layers
// - Sliding Window = 512 tokens
// - 16 Query Heads, 4 KV Heads (4:1 GQA), head_dim = 128
// - Per token: 4 KV heads * 128 = 512 elements for K, 512 elements for V (1024 elements = 2048 bytes in FP16)
// - Per layer: 512 tokens * 2048 bytes = 1.00 MB
// - Total across all 11 attention layers = 11.0 MB
// - Entire KV cache + 373 KB ShortConv state sits 100% inside 16 MB unified L3 cache!
// -----------------------------------------------------------------------------
struct D1ActiveLayerKVCache {
    static constexpr int NUM_ATTN_LAYERS = 11;
    static constexpr int SLIDING_WINDOW = 512;
    static constexpr int HEAD_COUNT = 16;
    static constexpr int HEAD_COUNT_KV = 4;
    static constexpr int HEAD_DIM = 128;
    static constexpr int KV_DIM = HEAD_COUNT_KV * HEAD_DIM; // 512

    // Per layer: 512 tokens x 512 elements in FP16 = 512 KB for K, 512 KB for V
    // Total per layer = 1.00 MB (100% resident in unified 16 MB L3 cache!)
    std::vector<uint16_t> k_cache[NUM_ATTN_LAYERS];
    std::vector<uint16_t> v_cache[NUM_ATTN_LAYERS];

    D1ActiveLayerKVCache() {
        for (int l = 0; l < NUM_ATTN_LAYERS; ++l) {
            k_cache[l].resize(SLIDING_WINDOW * KV_DIM, 0);
            v_cache[l].resize(SLIDING_WINDOW * KV_DIM, 0);
        }
    }

    void insert(int attn_layer_idx, int token_pos, const float* k, const float* v) {
        if (attn_layer_idx < 0 || attn_layer_idx >= NUM_ATTN_LAYERS) return;
        int slot = token_pos % SLIDING_WINDOW;
        uint16_t* kp = &k_cache[attn_layer_idx][slot * KV_DIM];
        uint16_t* vp = &v_cache[attn_layer_idx][slot * KV_DIM];

        for (int i = 0; i < KV_DIM; i += 8) {
            __m256 f_k = _mm256_loadu_ps(k + i);
            __m256 f_v = _mm256_loadu_ps(v + i);
            _mm_storeu_si128((__m128i*)(kp + i), _mm256_cvtps_ph(f_k, 0));
            _mm_storeu_si128((__m128i*)(vp + i), _mm256_cvtps_ph(f_v, 0));
        }
    }

    // Multi-head attention with head gating:
    // 16 query heads, 4 KV heads (4:1 GQA).
    // Gating: each query head has a scalar gate weight g_h, applied as sigmoid(g_h) * attn_out[h]
    void attend(int attn_layer_idx, int current_pos, const float* q, const float* gate, float* out_attn) {
        if (attn_layer_idx < 0 || attn_layer_idx >= NUM_ATTN_LAYERS) return;
        int active_tokens = (std::min)(current_pos + 1, SLIDING_WINDOW);
        const uint16_t* kp_base = k_cache[attn_layer_idx].data();
        const uint16_t* vp_base = v_cache[attn_layer_idx].data();

        const float inv_sqrt_d = 1.0f / std::sqrt(128.0f); // 1.0 / 11.3137

        for (int h = 0; h < HEAD_COUNT; ++h) {
            int kv_h = h / (HEAD_COUNT / HEAD_COUNT_KV); // h / 4
            const float* q_h = q + h * HEAD_DIM;
            float* out_h = out_attn + h * HEAD_DIM;
            memset(out_h, 0, HEAD_DIM * sizeof(float));

            // Sigmoid gating factor
            float g_val = gate ? gate[h] : 0.0f;
            float sig_g = 1.0f / (1.0f + std::exp(-g_val));

            float max_score = -1e9f;
            float scores[SLIDING_WINDOW];

            for (int t = 0; t < active_tokens; ++t) {
                const uint16_t* k_t = kp_base + t * KV_DIM + kv_h * HEAD_DIM;
                __m256 dot_acc = _mm256_setzero_ps();
                for (int d = 0; d < HEAD_DIM; d += 8) {
                    __m128i h_k = _mm_loadu_si128((const __m128i*)(k_t + d));
                    __m256 f_k = _mm256_cvtph_ps(h_k);
                    __m256 f_q = _mm256_loadu_ps(q_h + d);
                    dot_acc = _mm256_fmadd_ps(f_q, f_k, dot_acc);
                }
                float score = hsum_float_8(dot_acc) * inv_sqrt_d;
                scores[t] = score;
                if (score > max_score) max_score = score;
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int t = 0; t < active_tokens; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int t = 0; t < active_tokens; ++t) {
                scores[t] *= inv_sum;
            }

            // Weighted sum of V scaled by head gate sigmoid
            for (int t = 0; t < active_tokens; ++t) {
                float weight = scores[t] * sig_g;
                __m256 v_w = _mm256_set1_ps(weight);
                const uint16_t* v_t = vp_base + t * KV_DIM + kv_h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d += 8) {
                    __m128i h_v = _mm_loadu_si128((const __m128i*)(v_t + d));
                    __m256 f_v = _mm256_cvtph_ps(h_v);
                    __m256 cur_out = _mm256_loadu_ps(out_h + d);
                    cur_out = _mm256_fmadd_ps(v_w, f_v, cur_out);
                    _mm256_storeu_ps(out_h + d, cur_out);
                }
            }
        }
    }
};

} // namespace d1
