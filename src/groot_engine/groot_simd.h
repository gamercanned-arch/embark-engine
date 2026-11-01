#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>

namespace groot {

constexpr int D_MODEL = 2048;
constexpr int NUM_LAYERS = 41;
constexpr int NUM_GDN_LAYERS = 30;
constexpr int NUM_ATTN_LAYERS = 11;
constexpr int NUM_EXPERTS = 128;
constexpr int EXPERTS_PER_TOKEN = 8;
constexpr int D_FF_EXPERT = 512;
constexpr int D_FF_SHARED = 512;
constexpr int GDN_STATE_SIZE = 128;
constexpr int GDN_GROUP_COUNT = 16;
constexpr int GDN_INNER_SIZE = 4096;
constexpr int GDN_CONV_KERNEL = 4;
constexpr int ATTN_HEADS = 16;
constexpr int ATTN_KV_HEADS = 2;
constexpr int ATTN_HEAD_DIM = 128; // 2048 / 16

// -----------------------------------------------------------------------------
// Vector Helpers
// -----------------------------------------------------------------------------
static inline float hsum_float_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

static inline float fp16_to_fp32(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

static inline __m256 fast_silu_avx2(__m256 x) {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    neg_x = _mm256_max_ps(neg_x, _mm256_set1_ps(-88.0f));
    neg_x = _mm256_min_ps(neg_x, _mm256_set1_ps(88.0f));

    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2f   = _mm256_set1_ps(0.6931471805599453f);
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(neg_x, log2ef), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 y = _mm256_fnmadd_ps(fx, ln2f, neg_x);

    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666666666666666f);
    const __m256 c4 = _mm256_set1_ps(0.041666666666666664f);
    const __m256 c5 = _mm256_set1_ps(0.008333333333333333f);
    __m256 p = _mm256_fmadd_ps(c5, y, c4);
    p = _mm256_fmadd_ps(p, y, c3);
    p = _mm256_fmadd_ps(p, y, c2);
    p = _mm256_fmadd_ps(p, y, c1);
    p = _mm256_fmadd_ps(p, y, c1);

    __m256i emm0 = _mm256_cvtps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 exp_neg = _mm256_mul_ps(p, _mm256_castsi256_ps(emm0));

    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg);
    __m256 sig = _mm256_div_ps(_mm256_set1_ps(1.0f), denom);
    return _mm256_mul_ps(x, sig);
}

static inline void rmsnorm_avx2(const float* __restrict x, const float* __restrict weight, float* __restrict out, int dim, float eps = 1e-6f) {
    __m256 sum2 = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum2 = _mm256_fmadd_ps(v, v, sum2);
    }
    float ms = (hsum_float_8(sum2) / dim) + eps;
    float scale = 1.0f / std::sqrt(ms);
    __m256 s = _mm256_set1_ps(scale);
    for (int i = 0; i < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, s), w));
    }
}

// -----------------------------------------------------------------------------
// Router GEMV with Hardware F16C Vector Conversion
// Evaluates 128 experts from [128, 2048] router weights against token embedding x [2048]
// -----------------------------------------------------------------------------
static inline void router_gemv_f16c_128(const uint16_t* __restrict W, const float* __restrict x, float* __restrict logits) {
    for (int r = 0; r < NUM_EXPERTS; r += 2) {
        const uint16_t* row0 = W + (r + 0) * D_MODEL;
        const uint16_t* row1 = W + (r + 1) * D_MODEL;

        __m256 acc0_0 = _mm256_setzero_ps();
        __m256 acc0_1 = _mm256_setzero_ps();
        __m256 acc1_0 = _mm256_setzero_ps();
        __m256 acc1_1 = _mm256_setzero_ps();

        for (int c = 0; c < D_MODEL; c += 16) {
            __m256 x0 = _mm256_loadu_ps(x + c + 0);
            __m256 x1 = _mm256_loadu_ps(x + c + 8);

            __m256 w0_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + c + 0)));
            __m256 w0_1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row0 + c + 8)));
            acc0_0 = _mm256_fmadd_ps(w0_0, x0, acc0_0);
            acc0_1 = _mm256_fmadd_ps(w0_1, x1, acc0_1);

            __m256 w1_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + c + 0)));
            __m256 w1_1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row1 + c + 8)));
            acc1_0 = _mm256_fmadd_ps(w1_0, x0, acc1_0);
            acc1_1 = _mm256_fmadd_ps(w1_1, x1, acc1_1);
        }

        logits[r + 0] = hsum_float_8(_mm256_add_ps(acc0_0, acc0_1));
        logits[r + 1] = hsum_float_8(_mm256_add_ps(acc1_0, acc1_1));
    }
}

// -----------------------------------------------------------------------------
// Top-8 Expert Selection (with Softmax over Top-8)
// -----------------------------------------------------------------------------
struct Top8Result {
    int idx[EXPERTS_PER_TOKEN];
    float weight[EXPERTS_PER_TOKEN];
};

static inline Top8Result select_top8_router(const float* __restrict logits) {
    Top8Result res;
    // Find top-8 indices via min-heap / selection
    struct Entry {
        float val;
        int idx;
    };
    Entry top[EXPERTS_PER_TOKEN];
    for (int i = 0; i < EXPERTS_PER_TOKEN; ++i) {
        top[i].val = logits[i];
        top[i].idx = i;
    }
    // Sort initial 8 ascending by val
    for (int i = 0; i < EXPERTS_PER_TOKEN - 1; ++i) {
        for (int j = i + 1; j < EXPERTS_PER_TOKEN; ++j) {
            if (top[i].val > top[j].val) {
                std::swap(top[i], top[j]);
            }
        }
    }
    // Scan remaining 120 experts
    for (int i = EXPERTS_PER_TOKEN; i < NUM_EXPERTS; ++i) {
        float v = logits[i];
        if (v > top[0].val) {
            top[0].val = v;
            top[0].idx = i;
            // Bubble up
            for (int k = 0; k < EXPERTS_PER_TOKEN - 1; ++k) {
                if (top[k].val > top[k + 1].val) {
                    std::swap(top[k], top[k + 1]);
                } else {
                    break;
                }
            }
        }
    }

    // Softmax normalization across the top 8
    float max_val = top[EXPERTS_PER_TOKEN - 1].val;
    float sum_exp = 0.0f;
    for (int k = 0; k < EXPERTS_PER_TOKEN; ++k) {
        top[k].val = std::exp(top[k].val - max_val);
        sum_exp += top[k].val;
    }
    float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    for (int k = 0; k < EXPERTS_PER_TOKEN; ++k) {
        res.idx[k] = top[EXPERTS_PER_TOKEN - 1 - k].idx; // Descending
        res.weight[k] = top[EXPERTS_PER_TOKEN - 1 - k].val * inv_sum;
    }
    return res;
}

// -----------------------------------------------------------------------------
// Non-Temporal Vector Streaming Loader (_MM_HINT_NTA) for MoE Experts
// Bypasses Zen 3 L2/L3 cache hierarchy to preserve stationary activations
// and GDN recurrent states while saturating DDR4-3200 line rate.
// -----------------------------------------------------------------------------
static inline uint64_t stream_groot_expert_nta(const void* __restrict ptr, size_t bytes) {
    const __m256i* p = (const __m256i*)ptr;
    size_t n_vec = bytes / sizeof(__m256i);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    size_t v = 0;
    for (; v + 8 <= n_vec; v += 4) {
        _mm_prefetch((const char*)(p + v + 16), _MM_HINT_NTA);
        acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(p + v + 0));
        acc1 = _mm256_add_epi64(acc1, _mm256_loadu_si256(p + v + 1));
        acc2 = _mm256_add_epi64(acc2, _mm256_loadu_si256(p + v + 2));
        acc3 = _mm256_add_epi64(acc3, _mm256_loadu_si256(p + v + 3));
    }
    for (; v < n_vec; ++v) {
        acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(p + v));
    }
    __m256i sum01 = _mm256_add_epi64(acc0, acc1);
    __m256i sum23 = _mm256_add_epi64(acc2, acc3);
    __m256i total = _mm256_add_epi64(sum01, sum23);
    return (uint64_t)_mm256_extract_epi64(total, 0) + (uint64_t)_mm256_extract_epi64(total, 1);
}

static inline uint64_t stream_cold_expert_chunk_movntdqa(const void* __restrict ptr, size_t bytes) {
    return stream_groot_expert_nta(ptr, bytes);
}

// -----------------------------------------------------------------------------
// Fused GDN Recurrence State
// O(1) constant state: 16 groups * 128 * 128 = 262,144 floats = 1,048,576 B (1.00 MB)
// Conv buffer: 4 steps * 8192 = 32,768 floats = 131,072 B (128 KB)
// Total active state per layer: 1.18 MB (fits entirely in Zen 3 L2/L3 cache!)
// -----------------------------------------------------------------------------
struct GDNRecurrentState {
    alignas(64) float state[GDN_GROUP_COUNT * GDN_STATE_SIZE * GDN_STATE_SIZE]; // 1.00 MB
    alignas(64) float conv_buf[GDN_CONV_KERNEL * 8192];                          // 128 KB
    int conv_step = 0;

    void reset() {
        std::fill(std::begin(state), std::end(state), 0.0f);
        std::fill(std::begin(conv_buf), std::end(conv_buf), 0.0f);
        conv_step = 0;
    }
};

// -----------------------------------------------------------------------------
// Fused GDN Recurrence Update Step (Vectorized AVX2)
// Evaluates in Zen 3 Private 512 KB L2 Cache (2 groups per core = 128 KB << 512 KB)
// Zero L3 cache evictions, zero cross-core cache invalidation, zero DDR4 traffic!
// -----------------------------------------------------------------------------
static inline void gdn_recurrent_step_l2_groups(
    GDNRecurrentState& s,
    int start_group,
    int num_groups,
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    const float* __restrict beta,
    const float* __restrict gate,
    float* __restrict out)
{
    constexpr int D = GDN_STATE_SIZE;  // 128

    for (int g = start_group; g < start_group + num_groups; ++g) {
        float* S_g = s.state + g * D * D;
        const float* k_g = k + g * D;
        const float* v_g = v + g * D;
        const float* q_g = q + g * D;
        const float* gate_g = gate + g * D;
        float* out_g = out + g * D;
        float b = beta[g];

        // 1. Vectorized dot products: u_g = S_g * k_g (128 rows)
        alignas(64) float u[D];
        for (int r = 0; r < D; ++r) {
            const float* row = S_g + r * D;
            __m256 acc = _mm256_setzero_ps();
            for (int c = 0; c < D; c += 8) {
                __m256 sv = _mm256_loadu_ps(row + c);
                __m256 kv = _mm256_loadu_ps(k_g + c);
                acc = _mm256_fmadd_ps(sv, kv, acc);
            }
            u[r] = hsum_float_8(acc);
        }

        // 2. Fused state update: S_g = S_g - b * (u (x) k^T) + (v (x) k^T)
        //                       S_g[r, c] += (v[r] - b * u[r]) * k[c]
        for (int r = 0; r < D; ++r) {
            float coeff = v_g[r] - b * u[r];
            __m256 cv = _mm256_set1_ps(coeff);
            float* row = S_g + r * D;
            for (int c = 0; c < D; c += 8) {
                __m256 sv = _mm256_loadu_ps(row + c);
                __m256 kv = _mm256_loadu_ps(k_g + c);
                sv = _mm256_fmadd_ps(cv, kv, sv);
                _mm256_storeu_ps(row + c, sv);
            }
        }

        // 3. Output projection: o_g = S_g * q_g (128 rows) and gating with SiLU
        for (int r = 0; r < D; r += 8) {
            __m256 o_vec = _mm256_setzero_ps();
            alignas(32) float tmp_o[8];
            for (int i = 0; i < 8; ++i) {
                const float* row = S_g + (r + i) * D;
                __m256 acc = _mm256_setzero_ps();
                for (int c = 0; c < D; c += 8) {
                    __m256 sv = _mm256_loadu_ps(row + c);
                    __m256 qv = _mm256_loadu_ps(q_g + c);
                    acc = _mm256_fmadd_ps(sv, qv, acc);
                }
                tmp_o[i] = hsum_float_8(acc);
            }
            o_vec = _mm256_load_ps(tmp_o);
            __m256 gv = _mm256_loadu_ps(gate_g + r);
            __m256 yv = _mm256_mul_ps(o_vec, fast_silu_avx2(gv));
            _mm256_storeu_ps(out_g + r, yv);
        }
    }
}

static inline void gdn_recurrent_step_avx2(
    GDNRecurrentState& s,
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    const float* __restrict beta,
    const float* __restrict gate,
    float* __restrict out)
{
    gdn_recurrent_step_l2_groups(s, 0, GDN_GROUP_COUNT, q, k, v, beta, gate, out);
}

} // namespace groot
