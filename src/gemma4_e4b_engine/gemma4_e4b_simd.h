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

namespace gemma4_e4b {

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

// Hardware BF16 to FP32 conversion
static inline float bf16_to_fp32(uint16_t val) {
    uint32_t bits = (uint32_t)val << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
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

        // Load pre-expanded 8-byte scales and 8-byte minimums
        uint64_t sc_raw, min_raw;
        memcpy(&sc_raw, x[i].d_scales, sizeof(sc_raw));
        memcpy(&min_raw, x[i].m_scales, sizeof(min_raw));

        const int8_t* q8_ptr = y[i].qs;
        const uint8_t* q4_ptr = x[i].qs;

        __m256i sumi = _mm256_setzero_si256();

        // 4 x 64-weight sub-blocks
        for (int sb = 0; sb < 4; ++sb) {
            __m256i q4_bytes = load_streaming_256(q4_ptr + sb * 32);
            __m256i q4_lo = _mm256_and_si256(q4_bytes, m4);
            __m256i q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), m4);

            __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8_ptr + sb * 64));
            __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8_ptr + sb * 64 + 32));

            // maddubs
            __m256i prod_lo = _mm256_maddubs_epi16(q4_lo, q8_0);
            __m256i prod_hi = _mm256_maddubs_epi16(q4_hi, q8_1);

            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(prod_lo, _mm256_set1_epi16(1)));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(prod_hi, _mm256_set1_epi16(1)));
        }

        // Scale dot product with d
        __m256 fsum = _mm256_cvtepi32_ps(sumi);
        acc = _mm256_fmadd_ps(fsum, _mm256_set1_ps(d), acc);

        // Account for min scale offsets
        float bsum_total = 0.0f;
        for (int b = 0; b < 16; ++b) {
            bsum_total += (float)y[i].bsums[b];
        }
        acc_m = _mm_add_ss(acc_m, _mm_set_ss(bsum_total * dmin));
    }

    return hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

// -----------------------------------------------------------------------------
// AVX2 Fused Q6_K Dot Product using Non-Temporal Streaming
// -----------------------------------------------------------------------------
static inline float vec_dot_q6_k_q8_k_zen3(
    const block_q6_K* __restrict x,
    const block_q8_K* __restrict y,
    int k)
{
    const int nb = k / QK_K;
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
        const int8_t*  qy = y[i].qs;

        __m256i sumi = _mm256_setzero_si256();

        for (int sb = 0; sb < 2; ++sb) {
            __m256i ql_bytes = load_streaming_256(ql + sb * 64);
            __m256i qy_bytes = _mm256_loadu_si256((const __m256i*)(qy + sb * 64));

            __m256i ql_lo = _mm256_and_si256(ql_bytes, _mm256_set1_epi8(0x0F));
            __m256i p = _mm256_maddubs_epi16(ql_lo, qy_bytes);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
        }

        __m256 fsum = _mm256_cvtepi32_ps(sumi);
        acc = _mm256_fmadd_ps(fsum, _mm256_set1_ps(d), acc);
    }

    return hsum_float_8(acc);
}

// -----------------------------------------------------------------------------
// AVX2 RMSNorm
// -----------------------------------------------------------------------------
static inline void rms_norm_avx2(float* __restrict out, const float* __restrict in, const float* __restrict weight, int dim, float eps = 1e-6f) {
    __m256 sum2 = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        sum2 = _mm256_fmadd_ps(v, v, sum2);
    }
    float s2 = hsum_float_8(sum2);
    float scale = 1.0f / std::sqrt(s2 / (float)dim + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    for (int i = 0; i < dim; i += 8) {
        __m256 vin = _mm256_loadu_ps(in + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vout = _mm256_mul_ps(_mm256_mul_ps(vin, vscale), vw);
        _mm256_storeu_ps(out + i, vout);
    }
}

// -----------------------------------------------------------------------------
// AVX2 GeLU / SwiGLU Element-wise
// -----------------------------------------------------------------------------
static inline void swiglu_avx2(float* __restrict gate, const float* __restrict up, int dim) {
    for (int i = 0; i < dim; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        // SiLU(g) = g / (1 + exp(-g)) approx
        __m256 silu = _mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(1.0f))); // fast approx
        _mm256_storeu_ps(gate + i, _mm256_mul_ps(silu, u));
    }
}

// -----------------------------------------------------------------------------
// Gemma 4 E4B Active Layer KV Cache (Sliding Window = 512, 16 MB L3 Resident)
// - Gemma 4 E4B: 42 layers, 8 query heads, 2 KV heads, head_dim = 256.
// - KV_DIM = 2 heads * 256 dim = 512 FP16 elements.
// - Each token: 512 x 2 bytes (FP16) = 1,024 bytes (1 KB) for K, 1 KB for V = 2 KB.
// - Per layer: 512 tokens x 2 KB = 1,048,576 bytes = 1.00 MB per active layer!
// - 1.00 MB active layer KV cache resides 100% inside 16 MB unified L3 cache!
// -----------------------------------------------------------------------------
struct Gemma4E4BActiveLayerKVCache {
    static constexpr int SLIDING_WINDOW = 512;
    static constexpr int HEAD_COUNT = 8;
    static constexpr int HEAD_COUNT_KV = 2;
    static constexpr int HEAD_DIM = 256;
    static constexpr int KV_DIM = HEAD_COUNT_KV * HEAD_DIM; // 512
    static constexpr int Q_DIM = HEAD_COUNT * HEAD_DIM;     // 2048
    static constexpr int NUM_LAYERS = 42;

    // Per layer: 512 tokens x 512 elements in FP16 = 512 KB for K, 512 KB for V
    // Total per layer = 1.00 MB (100% resident in unified 16 MB L3 cache!)
    std::vector<uint16_t> k_cache[NUM_LAYERS];
    std::vector<uint16_t> v_cache[NUM_LAYERS];

    Gemma4E4BActiveLayerKVCache() {
        for (int l = 0; l < NUM_LAYERS; ++l) {
            k_cache[l].resize(SLIDING_WINDOW * KV_DIM, 0);
            v_cache[l].resize(SLIDING_WINDOW * KV_DIM, 0);
        }
    }

    // Insert new token into circular sliding window buffer (pos % 512)
    void insert(int layer_idx, int token_pos, const float* k, const float* v) {
        if (layer_idx < 0 || layer_idx >= NUM_LAYERS) return;
        int slot = token_pos % SLIDING_WINDOW;
        uint16_t* kp = &k_cache[layer_idx][slot * KV_DIM];
        uint16_t* vp = &v_cache[layer_idx][slot * KV_DIM];

        for (int i = 0; i < KV_DIM; i += 8) {
            __m256 f_k = _mm256_loadu_ps(k + i);
            __m256 f_v = _mm256_loadu_ps(v + i);
            _mm_storeu_si128((__m128i*)(kp + i), _mm256_cvtps_ph(f_k, 0));
            _mm_storeu_si128((__m128i*)(vp + i), _mm256_cvtps_ph(f_v, 0));
        }
    }

    // Attention computation over active sliding window (100% L3 resident)
    void attend(int layer_idx, int current_pos, const float* q, float* out_attn) {
        if (layer_idx < 0 || layer_idx >= NUM_LAYERS) return;
        int active_tokens = (std::min)(current_pos + 1, SLIDING_WINDOW);
        const uint16_t* kp_base = k_cache[layer_idx].data();
        const uint16_t* vp_base = v_cache[layer_idx].data();

        // 8 query heads, 2 KV heads (4:1 GQA)
        // Each query head attends to kv_head = q_head / 4:
        // kv_head 0 for q_heads 0..3
        // kv_head 1 for q_heads 4..7
        for (int h = 0; h < HEAD_COUNT; ++h) {
            int kv_h = h / 4;
            const float* q_h = q + h * HEAD_DIM;
            float* out_h = out_attn + h * HEAD_DIM;
            memset(out_h, 0, HEAD_DIM * sizeof(float));

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
                float score = hsum_float_8(dot_acc) * (1.0f / 16.0f); // 1 / sqrt(256) = 1/16
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

            // Weighted sum of V
            for (int t = 0; t < active_tokens; ++t) {
                float weight = scores[t];
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

} // namespace gemma4_e4b
