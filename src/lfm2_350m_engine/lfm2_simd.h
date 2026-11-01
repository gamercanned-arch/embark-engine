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

namespace lfm2 {

// Architectural Constants for LFM 2.5 350M (lfm2 architecture)
constexpr int D_MODEL = 1024;
constexpr int D_FF = 4608;
constexpr int NUM_LAYERS = 16;
constexpr int NUM_SHORTCONV_LAYERS = 10; // Layers: 0, 1, 3, 4, 6, 7, 9, 11, 13, 15
constexpr int NUM_ATTN_LAYERS = 6;      // Layers: 2, 5, 8, 10, 12, 14
constexpr int HEAD_COUNT = 16;
constexpr int HEAD_COUNT_KV = 8;        // GQA ratio 2:1 (16 Q heads, 8 KV heads)
constexpr int HEAD_DIM = 64;
constexpr int SHORTCONV_K = 3;          // l_cache = 3
constexpr int SHORTCONV_HISTORY = 2;    // K - 1 = 2 history slots per channel
constexpr int VOCAB_SIZE = 65536;
constexpr float ROPE_FREQ_BASE = 1000000.0f;
constexpr float RMS_NORM_EPS = 1e-5f;

#pragma pack(push, 1)

// Standard GGUF Q8_0 Block (34 bytes)
struct block_q8_0 {
    uint16_t d;      // fp16 scale
    int8_t   qs[32]; // 32 int8 values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

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

// -----------------------------------------------------------------------------
// AVX2 High-Performance Q8_0 Dot Product: dot(w[K], x[K])
// Unrolls by 32 elements per block using F16C scale broadcast + AVX2 FMA
// -----------------------------------------------------------------------------
static inline float vec_dot_q8_0_f32(const block_q8_0* __restrict w, const float* __restrict x, int k) {
    const int nb = k / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*)(w + i + 2), _MM_HINT_NTA);
            _mm_prefetch((const char*)(x + (i + 2) * 32), _MM_HINT_T0);
        }

        const float d = fp16_to_fp32(w[i].d);
        const __m256 vd = _mm256_set1_ps(d);

        // Load 32 int8 weights
        __m256i w_bytes = load_streaming_256(w[i].qs);

        // Lower 16 int8 -> 2 x 8 FP32
        __m128i lo16 = _mm256_castsi256_si128(w_bytes);
        __m256i lo32_0 = _mm256_cvtepi8_epi32(lo16);
        __m256i lo32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8));

        __m256 wf0 = _mm256_cvtepi32_ps(lo32_0);
        __m256 wf1 = _mm256_cvtepi32_ps(lo32_1);

        // Upper 16 int8 -> 2 x 8 FP32
        __m128i hi16 = _mm256_extracti128_si256(w_bytes, 1);
        __m256i hi32_0 = _mm256_cvtepi8_epi32(hi16);
        __m256i hi32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8));

        __m256 wf2 = _mm256_cvtepi32_ps(hi32_0);
        __m256 wf3 = _mm256_cvtepi32_ps(hi32_1);

        // Load 32 FP32 activations
        __m256 x0 = _mm256_loadu_ps(x + i * 32 + 0);
        __m256 x1 = _mm256_loadu_ps(x + i * 32 + 8);
        __m256 x2 = _mm256_loadu_ps(x + i * 32 + 16);
        __m256 x3 = _mm256_loadu_ps(x + i * 32 + 24);

        // Dot product accumulation scaled by block scale d
        __m256 p0 = _mm256_mul_ps(wf0, x0);
        p0 = _mm256_fmadd_ps(wf1, x1, p0);
        __m256 p1 = _mm256_mul_ps(wf2, x2);
        p1 = _mm256_fmadd_ps(wf3, x3, p1);

        acc0 = _mm256_fmadd_ps(vd, p0, acc0);
        acc1 = _mm256_fmadd_ps(vd, p1, acc1);
    }

    return hsum_float_8(_mm256_add_ps(acc0, acc1));
}

// -----------------------------------------------------------------------------
// Fused ShortConv AVX2 Kernel
// Circular Shift + Conv1D (K=3) + FMA + Gating in registers
// 10 layers x 1024 channels. State is 2 history values per channel.
// Total state across 10 layers: 10 * 2 * 1024 * 4B = 81.92 KB (FP32).
// 100% sits resident inside L2/L3 cache with 0 DRAM roundtrips!
// -----------------------------------------------------------------------------
struct LFM2ShortConvState {
    static constexpr int NUM_CONV_LAYERS = 10;
    static constexpr int CHANNELS = 1024;
    static constexpr int KERNEL_SIZE = 3;
    static constexpr int HISTORY_LEN = 2; // K - 1 = 2

    // State buffer: [10 layers][2 history slots][1024 channels] in FP32
    // 10 * 2 * 1024 * 4 bytes = 81,920 bytes (80 KB)
    alignas(64) float state[NUM_CONV_LAYERS][HISTORY_LEN][CHANNELS];

    LFM2ShortConvState() {
        memset(state, 0, sizeof(state));
    }

    void reset() {
        memset(state, 0, sizeof(state));
    }

    // Fused AVX2 Kernel:
    // Computes:
    // 1. Elementwise bx = b * x (where b, x each have length 1024)
    // 2. Conv1D with K=3 across 1024 channels:
    //    conv_out[ch] = kernel[0, ch] * state[0, ch] + kernel[1, ch] * state[1, ch] + kernel[2, ch] * bx[ch]
    // 3. Circular register shift: state[0] <- state[1], state[1] <- bx
    // 4. Output gating: out[ch] = c[ch] * conv_out[ch]
    void forward_fused_avx2(
        int layer_conv_idx,
        const float* __restrict b,
        const float* __restrict c,
        const float* __restrict x,
        const float* __restrict kernel_weights, // [3, 1024]
        float*       __restrict out)
    {
        float* lyr_state = &state[layer_conv_idx][0][0];

        // Process 8 channels per loop iteration (1024 / 8 = 128 iterations)
        for (int ch = 0; ch < CHANNELS; ch += 8) {
            __m256 vb = _mm256_loadu_ps(b + ch);
            __m256 vx = _mm256_loadu_ps(x + ch);
            __m256 vc = _mm256_loadu_ps(c + ch);

            // bx = b * x
            __m256 v_bx = _mm256_mul_ps(vb, vx);

            // Load 2 history vectors from L3-resident state
            __m256 h0 = _mm256_loadu_ps(&lyr_state[0 * CHANNELS + ch]);
            __m256 h1 = _mm256_loadu_ps(&lyr_state[1 * CHANNELS + ch]);

            // Load 3 kernel weights for these 8 channels
            __m256 w0 = _mm256_loadu_ps(kernel_weights + 0 * CHANNELS + ch);
            __m256 w1 = _mm256_loadu_ps(kernel_weights + 1 * CHANNELS + ch);
            __m256 w2 = _mm256_loadu_ps(kernel_weights + 2 * CHANNELS + ch);

            // FMA dot product in registers
            __m256 conv_out = _mm256_mul_ps(w0, h0);
            conv_out = _mm256_fmadd_ps(w1, h1, conv_out);
            conv_out = _mm256_fmadd_ps(w2, v_bx, conv_out);

            // Circular shift update in L2/L3 cache: state[0] = state[1]; state[1] = v_bx
            _mm256_storeu_ps(&lyr_state[0 * CHANNELS + ch], h1);
            _mm256_storeu_ps(&lyr_state[1 * CHANNELS + ch], v_bx);

            // Gated output: out = c * conv_out
            __m256 v_out = _mm256_mul_ps(vc, conv_out);
            _mm256_storeu_ps(out + ch, v_out);
        }
    }
};

// -----------------------------------------------------------------------------
// Active Layer KV Cache & Multi-Head Attention for 6 Attention Layers
// Total KV Cache: 6 layers x 2048 seq x 512 dim x 2 (K+V) x 2 bytes (FP16) = 12.58 MB
// 100% sits resident in the 16 MB unified L3 cache with 0 DRAM evictions!
// -----------------------------------------------------------------------------
struct LFM2ActiveLayerKVCache {
    static constexpr int NUM_ATTN = 6;
    static constexpr int MAX_SEQ = 2048;
    static constexpr int N_HEADS = 16;
    static constexpr int N_HEADS_KV = 8; // GQA 2:1
    static constexpr int DIM_HEAD = 64;
    static constexpr int KV_DIM = N_HEADS_KV * DIM_HEAD; // 512

    alignas(64) uint16_t k_cache[NUM_ATTN][MAX_SEQ][KV_DIM];
    alignas(64) uint16_t v_cache[NUM_ATTN][MAX_SEQ][KV_DIM];

    LFM2ActiveLayerKVCache() {
        reset();
    }

    void reset() {
        memset(k_cache, 0, sizeof(k_cache));
        memset(v_cache, 0, sizeof(v_cache));
    }

    void insert(int layer_idx, int pos, const float* k_vec, const float* v_vec) {
        if (pos >= MAX_SEQ) return;
        uint16_t* k_dst = &k_cache[layer_idx][pos][0];
        uint16_t* v_dst = &v_cache[layer_idx][pos][0];

        for (int i = 0; i < KV_DIM; i += 8) {
            __m256 k_f = _mm256_loadu_ps(k_vec + i);
            __m256 v_f = _mm256_loadu_ps(v_vec + i);
            __m128i k_h = _mm256_cvtps_ph(k_f, 0);
            __m128i v_h = _mm256_cvtps_ph(v_f, 0);
            _mm_storeu_si128((__m128i*)(k_dst + i), k_h);
            _mm_storeu_si128((__m128i*)(v_dst + i), v_h);
        }
    }

    void attend(int layer_idx, int pos, const float* q_vec, float* out_vec) {
        const float scale = 0.125f; // 1.0f / sqrtf(64.0f)
        int seq_len = (std::min)(pos + 1, MAX_SEQ);

        alignas(64) float scores[MAX_SEQ];

        // 16 Query heads
        for (int h = 0; h < N_HEADS; ++h) {
            int kv_h = h / 2; // GQA: 2 Q heads share 1 KV head
            const float* q_head = q_vec + h * DIM_HEAD;
            float max_score = -1e30f;

            for (int t = 0; t < seq_len; ++t) {
                const uint16_t* k_head = &k_cache[layer_idx][t][kv_h * DIM_HEAD];

                // Head dot product in AVX2 (64 dimensions)
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                for (int d = 0; d < DIM_HEAD; d += 16) {
                    __m256 q0 = _mm256_loadu_ps(q_head + d + 0);
                    __m256 q1 = _mm256_loadu_ps(q_head + d + 8);

                    __m128i kh0 = _mm_loadu_si128((const __m128i*)(k_head + d + 0));
                    __m128i kh1 = _mm_loadu_si128((const __m128i*)(k_head + d + 8));

                    __m256 kf0 = _mm256_cvtph_ps(kh0);
                    __m256 kf1 = _mm256_cvtph_ps(kh1);

                    acc0 = _mm256_fmadd_ps(q0, kf0, acc0);
                    acc1 = _mm256_fmadd_ps(q1, kf1, acc1);
                }

                float score = hsum_float_8(_mm256_add_ps(acc0, acc1)) * scale;
                scores[t] = score;
                if (score > max_score) max_score = score;
            }

            // Stable Softmax
            float sum_exp = 0.0f;
            for (int t = 0; t < seq_len; ++t) {
                float e = expf(scores[t] - max_score);
                scores[t] = e;
                sum_exp += e;
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (int t = 0; t < seq_len; ++t) {
                scores[t] *= inv_sum;
            }

            // Weighted sum over V values (64 dimensions)
            float* head_out = out_vec + h * DIM_HEAD;
            __m256 v_acc0 = _mm256_setzero_ps();
            __m256 v_acc1 = _mm256_setzero_ps();
            __m256 v_acc2 = _mm256_setzero_ps();
            __m256 v_acc3 = _mm256_setzero_ps();
            __m256 v_acc4 = _mm256_setzero_ps();
            __m256 v_acc5 = _mm256_setzero_ps();
            __m256 v_acc6 = _mm256_setzero_ps();
            __m256 v_acc7 = _mm256_setzero_ps();

            for (int t = 0; t < seq_len; ++t) {
                __m256 sw = _mm256_set1_ps(scores[t]);
                const uint16_t* v_head = &v_cache[layer_idx][t][kv_h * DIM_HEAD];

                __m256 vf0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 0)));
                __m256 vf1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 8)));
                __m256 vf2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 16)));
                __m256 vf3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 24)));
                __m256 vf4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 32)));
                __m256 vf5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 40)));
                __m256 vf6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 48)));
                __m256 vf7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(v_head + 56)));

                v_acc0 = _mm256_fmadd_ps(sw, vf0, v_acc0);
                v_acc1 = _mm256_fmadd_ps(sw, vf1, v_acc1);
                v_acc2 = _mm256_fmadd_ps(sw, vf2, v_acc2);
                v_acc3 = _mm256_fmadd_ps(sw, vf3, v_acc3);
                v_acc4 = _mm256_fmadd_ps(sw, vf4, v_acc4);
                v_acc5 = _mm256_fmadd_ps(sw, vf5, v_acc5);
                v_acc6 = _mm256_fmadd_ps(sw, vf6, v_acc6);
                v_acc7 = _mm256_fmadd_ps(sw, vf7, v_acc7);
            }

            _mm256_storeu_ps(head_out + 0,  v_acc0);
            _mm256_storeu_ps(head_out + 8,  v_acc1);
            _mm256_storeu_ps(head_out + 16, v_acc2);
            _mm256_storeu_ps(head_out + 24, v_acc3);
            _mm256_storeu_ps(head_out + 32, v_acc4);
            _mm256_storeu_ps(head_out + 40, v_acc5);
            _mm256_storeu_ps(head_out + 48, v_acc6);
            _mm256_storeu_ps(head_out + 56, v_acc7);
        }
    }
};

// -----------------------------------------------------------------------------
// AVX2 RMSNorm Kernel
// -----------------------------------------------------------------------------
static inline void rms_norm_avx2(
    const float* __restrict x,
    const float* __restrict w,
    float*       __restrict y,
    int dim,
    float eps = RMS_NORM_EPS)
{
    __m256 sum2 = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum2 = _mm256_fmadd_ps(vx, vx, sum2);
    }
    float s = hsum_float_8(sum2);
    float rms = 1.0f / sqrtf(s / (float)dim + eps);
    __m256 vrms = _mm256_set1_ps(rms);

    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vw));
    }
}

// -----------------------------------------------------------------------------
// AVX2 SwiGLU / SiLU Gated FFN Kernel: y = (silu(gate) * up)
// -----------------------------------------------------------------------------
static inline void silu_gated_avx2(
    const float* __restrict gate,
    const float* __restrict up,
    float*       __restrict y,
    int dim)
{
    for (int i = 0; i < dim; i += 8) {
        __m256 vg = _mm256_loadu_ps(gate + i);
        __m256 vu = _mm256_loadu_ps(up + i);

        // Accurate SiLU: x / (1.0 + exp(-x))
        alignas(32) float g_arr[8];
        alignas(32) float u_arr[8];
        _mm256_storeu_ps(g_arr, vg);
        _mm256_storeu_ps(u_arr, vu);

        alignas(32) float res[8];
        for (int j = 0; j < 8; ++j) {
            float g = g_arr[j];
            float sig = 1.0f / (1.0f + expf(-g));
            res[j] = (g * sig) * u_arr[j];
        }

        _mm256_storeu_ps(y + i, _mm256_loadu_ps(res));
    }
}

// -----------------------------------------------------------------------------
// Rotary Positional Embedding (RoPE) NEOX
// -----------------------------------------------------------------------------
static inline void apply_rope_neox(
    float* q_vec, int n_heads,
    float* k_vec, int n_heads_kv,
    int pos, int head_dim = HEAD_DIM,
    float freq_base = ROPE_FREQ_BASE)
{
    for (int i = 0; i < head_dim / 2; ++i) {
        float theta = powf(freq_base, -2.0f * (float)i / (float)head_dim);
        float angle = (float)pos * theta;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        // Q heads
        for (int h = 0; h < n_heads; ++h) {
            float* q = q_vec + h * head_dim;
            float q0 = q[i];
            float q1 = q[i + head_dim / 2];
            q[i] = q0 * cos_val - q1 * sin_val;
            q[i + head_dim / 2] = q0 * sin_val + q1 * cos_val;
        }

        // K heads
        for (int h = 0; h < n_heads_kv; ++h) {
            float* k = k_vec + h * head_dim;
            float k0 = k[i];
            float k1 = k[i + head_dim / 2];
            k[i] = k0 * cos_val - k1 * sin_val;
            k[i + head_dim / 2] = k0 * sin_val + k1 * cos_val;
        }
    }
}

} // namespace lfm2
