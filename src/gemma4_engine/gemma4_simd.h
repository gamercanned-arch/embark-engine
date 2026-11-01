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

namespace gemma4 {

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

// Pre-expanded Q4_K Block with aligned 8-bit scales and mins (+4 bytes = 148 bytes, +139 MB overall)
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
    uint8_t  ql[128]; // lower 4 bits
    uint8_t  qh[64];  // upper 2 bits
    int8_t   scales[16]; // signed 8-bit scales
    uint16_t d;       // fp16 super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

// Activation Quantization Block Q8_K (276 bytes)
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

// Unpack a raw Q4_K block into pre-expanded block
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
                val = std::max(-128, std::min(127, val));
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

            // Dual ALU execution on Zen 3: maddubs and madd_epi16
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

        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q4bits1 = load_streaming_256(q4); q4 += 32;
            const __m256i q4bits2 = load_streaming_256(q4); q4 += 32;
            const __m256i q4bitsH = load_streaming_256(qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8((char)-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            int is = j * 4;
            const __m128i sc0 = _mm_set1_epi8(((const int8_t*)x[i].scales)[is + 0]);
            const __m128i sc1 = _mm_set1_epi8(((const int8_t*)x[i].scales)[is + 1]);
            const __m128i sc2 = _mm_set1_epi8(((const int8_t*)x[i].scales)[is + 2]);
            const __m128i sc3 = _mm_set1_epi8(((const int8_t*)x[i].scales)[is + 3]);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return hsum_float_8(acc);
}

// -----------------------------------------------------------------------------
// Vectorized RMSNorm Kernel
// out = (x / sqrt(mean(x^2) + eps)) * weight
// -----------------------------------------------------------------------------
static inline void rmsnorm_avx2(
    float* __restrict out,
    const float* __restrict x,
    const float* __restrict weight,
    int n,
    float eps = 1e-6f)
{
    __m256 sum2 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum2 = _mm256_fmadd_ps(vx, vx, sum2);
    }
    float s = hsum_float_8(sum2);
    for (; i < n; ++i) {
        s += x[i] * x[i];
    }
    float scale = 1.0f / std::sqrt((s / n) + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = weight ? _mm256_loadu_ps(weight + i) : _mm256_set1_ps(1.0f);
        __m256 vy = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
        _mm256_storeu_ps(out + i, vy);
    }
    for (; i < n; ++i) {
        float w = weight ? weight[i] : 1.0f;
        out[i] = x[i] * scale * w;
    }
}

// -----------------------------------------------------------------------------
// Fused GeGLU: out = GELU(gate) * up
// Using fast tanh approximation: GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// -----------------------------------------------------------------------------
static inline void geglu_fused_avx2(
    float* __restrict out,
    const float* __restrict gate,
    const float* __restrict up,
    int n)
{
    const float sqrt_2_pi = 0.79788456f;
    const float coeff = 0.044715f;

    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        float u = up[i];
        float x3 = g * g * g;
        float inner = sqrt_2_pi * (g + coeff * x3);
        float th = std::tanh(inner);
        float gelu = 0.5f * g * (1.0f + th);
        out[i] = gelu * u;
    }
}

// -----------------------------------------------------------------------------
// 1024-Token Sliding Window KV Cache Structure
// Resident in 16 MB L3 cache of Ryzen 7 7730U
// SWA layer: W = 1024 tokens, n_head_kv = 8, head_dim = 256 -> 4096 floats = 8 KB/token -> 8 MB per active layer
// Global layer: n_head_kv = 1, head_dim = 512, V = K (tied) -> 512 floats = 1 KB/token
// -----------------------------------------------------------------------------
struct Gemma4KVCache {
    static constexpr int SWA_WINDOW = 1024;
    static constexpr int SWA_HEADS = 8;
    static constexpr int SWA_HEAD_DIM = 256;
    static constexpr int SWA_KV_DIM = SWA_HEADS * SWA_HEAD_DIM; // 2048

    static constexpr int GLOBAL_HEADS = 1;
    static constexpr int GLOBAL_HEAD_DIM = 512;
    static constexpr int GLOBAL_KV_DIM = GLOBAL_HEADS * GLOBAL_HEAD_DIM; // 512

    // SWA Ring Buffer for active layer: [1024, 2048] FP16 (4 MB for K, 4 MB for V = 8 MB total)
    // Fits 100% in 16 MB L3 cache
    alignas(64) uint16_t swa_k[SWA_WINDOW * SWA_KV_DIM];
    alignas(64) uint16_t swa_v[SWA_WINDOW * SWA_KV_DIM];

    // Global Key Ring Buffer for active layer: [2048, 512] FP16 (2 MB total, V is tied to K)
    alignas(64) uint16_t global_k[2048 * GLOBAL_KV_DIM];

    int head_pos = 0;

    void insert_swa(int pos, const float* k_vec, const float* v_vec) {
        int slot = pos % SWA_WINDOW;
        uint16_t* dst_k = swa_k + slot * SWA_KV_DIM;
        uint16_t* dst_v = swa_v + slot * SWA_KV_DIM;
        for (int i = 0; i < SWA_KV_DIM; ++i) {
            dst_k[i] = fp32_to_fp16(k_vec[i]);
            dst_v[i] = fp32_to_fp16(v_vec[i]);
        }
    }

    void insert_global(int pos, const float* k_vec) {
        int slot = pos % 2048;
        uint16_t* dst_k = global_k + slot * GLOBAL_KV_DIM;
        for (int i = 0; i < GLOBAL_KV_DIM; ++i) {
            dst_k[i] = fp32_to_fp16(k_vec[i]);
            // Notice: V = K (tied!), no dst_v needed
        }
    }
};

} // namespace gemma4
