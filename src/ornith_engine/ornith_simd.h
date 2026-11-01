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

namespace ornith {

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

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        const int8_t*  q8 = y[i].qs;

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < 4; ++j) {
            // Load 32 bytes of ql with streaming load
            const __m256i ql_bytes = load_streaming_256(ql); ql += 32;
            const __m256i qh_bytes = _mm256_set1_epi64x(*(const int64_t*)(qh + j * 8));

            const __m256i ql0 = _mm256_and_si256(ql_bytes, m15);
            const __m256i ql1 = _mm256_and_si256(_mm256_srli_epi16(ql_bytes, 4), m15);

            const __m256i qh0 = _mm256_and_si256(qh_bytes, m3);
            const __m256i qh1 = _mm256_and_si256(_mm256_srli_epi16(qh_bytes, 2), m3);

            // Reconstruct 6-bit signed weights: (ql | (qh << 4)) - 32
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
// Zen 3 GDN Recurrent State Update (Partitioned Across 8 Physical Cores)
// Recurrent state per layer: [4096, 128] = 524,288 FP16 elements = 1.05 MB
// Each core processes 512 rows = 65,536 FP16 elements = 134 KB
// Fits 100% inside private 512 KB L2 cache (2.80 ns latency)!
// -----------------------------------------------------------------------------
struct GDNThreadState {
    // 512 rows x 128 cols in FP16 = 65,536 elements = 131,072 bytes (128 KiB)
    alignas(64) uint16_t state[512 * 128];
};

static inline void gdn_state_step_avx2(
    GDNThreadState* __restrict s,
    const float*    __restrict k_vec, // 128 elements
    const float*    __restrict v_vec, // 512 elements (portion for this thread)
    const float*    __restrict q_vec, // 128 elements
    float*          __restrict out_y, // 512 elements
    float decay)
{
    const __m256 vdecay = _mm256_set1_ps(decay);

    for (int r = 0; r < 512; ++r) {
        float vr = v_vec[r];
        const __m256 v_val = _mm256_set1_ps(vr);
        uint16_t* s_row = &s->state[r * 128];

        __m256 y_acc = _mm256_setzero_ps();

        // 128 elements processed in 16 chunks of 8 floats
        for (int c = 0; c < 128; c += 8) {
            // Load previous FP16 state from private L2 cache (2.80 ns)
            __m128i h_s = _mm_loadu_si128((const __m128i*)(s_row + c));
            __m256 f_s = _mm256_cvtph_ps(h_s);

            __m256 k_chunk = _mm256_loadu_ps(k_vec + c);
            __m256 q_chunk = _mm256_loadu_ps(q_vec + c);

            // S_new = decay * S_old + V * K
            __m256 s_new = _mm256_fmadd_ps(v_val, k_chunk, _mm256_mul_ps(vdecay, f_s));

            // Write back to private L2 cache
            __m128i h_new = _mm256_cvtps_ph(s_new, 0);
            _mm_storeu_si128((__m128i*)(s_row + c), h_new);

            // Output Y = S_new * Q
            y_acc = _mm256_fmadd_ps(s_new, q_chunk, y_acc);
        }

        out_y[r] = hsum_float_8(y_acc);
    }
}

// -----------------------------------------------------------------------------
// Full Attention Layer KV Cache for 9 Attention Layers
// Head count: 16 Q heads, 4 KV heads (GQA 4:1), head_dim = 128 (4 * 128 = 512 KV dim)
// -----------------------------------------------------------------------------
struct OrnithFullAttentionKVCache {
    static constexpr int MAX_CTX = 4096;
    static constexpr int KV_DIM = 512; // 4 heads * 128 head_dim = 512
    static constexpr int NUM_ATTN_LAYERS = 9;

    // KV Cache for 9 attention layers: 9 * 4096 * 512 * 2 bytes = 37.7 MB
    std::vector<uint16_t> k_cache[NUM_ATTN_LAYERS];
    std::vector<uint16_t> v_cache[NUM_ATTN_LAYERS];

    OrnithFullAttentionKVCache() {
        for (int l = 0; l < NUM_ATTN_LAYERS; ++l) {
            k_cache[l].resize(MAX_CTX * KV_DIM, 0);
            v_cache[l].resize(MAX_CTX * KV_DIM, 0);
        }
    }

    void insert(int layer_idx, int pos, const float* k, const float* v) {
        if (layer_idx < 0 || layer_idx >= NUM_ATTN_LAYERS || pos >= MAX_CTX) return;
        uint16_t* kp = &k_cache[layer_idx][pos * KV_DIM];
        uint16_t* vp = &v_cache[layer_idx][pos * KV_DIM];
        for (int i = 0; i < KV_DIM; i += 8) {
            __m256 f_k = _mm256_loadu_ps(k + i);
            __m256 f_v = _mm256_loadu_ps(v + i);
            _mm_storeu_si128((__m128i*)(kp + i), _mm256_cvtps_ph(f_k, 0));
            _mm_storeu_si128((__m128i*)(vp + i), _mm256_cvtps_ph(f_v, 0));
        }
    }
};

} // namespace ornith
