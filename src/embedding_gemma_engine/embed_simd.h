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

namespace embed_gemma {

#ifndef QK_K
#define QK_K 256
#endif

#pragma pack(push, 1)

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
static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
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

// Fast RMSNorm with AVX2
static inline void rms_norm_avx2(
    const float* __restrict x,
    const float* __restrict weight,
    float*       __restrict out,
    int dim,
    float eps = 1e-6f)
{
    __m256 sum2 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum2 = _mm256_fmadd_ps(vx, vx, sum2);
    }
    float s = hsum_float_8(sum2);
    for (; i < dim; ++i) {
        s += x[i] * x[i];
    }

    float scale = 1.0f / sqrtf(s / (float)dim + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    i = 0;
    if (weight) {
        for (; i + 8 <= dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 vw_plus_1 = _mm256_add_ps(vw, _mm256_set1_ps(1.0f));
            __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw_plus_1);
            _mm256_storeu_ps(out + i, vo);
        }
        for (; i < dim; ++i) {
            out[i] = x[i] * scale * (weight[i] + 1.0f);
        }
    } else {
        for (; i + 8 <= dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(vx, vscale));
        }
        for (; i < dim; ++i) {
            out[i] = x[i] * scale;
        }
    }
}

// Fast GeGLU AVX2: out = GELU(gate) * up
static inline void geglu_avx2(
    const float* __restrict gate,
    const float* __restrict up,
    float*       __restrict out,
    int dim)
{
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 sqrt_2_over_pi = _mm256_set1_ps(0.7978845608f);
    const __m256 coef = _mm256_set1_ps(0.044715f);

    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vg = _mm256_loadu_ps(gate + i);
        __m256 vu = _mm256_loadu_ps(up + i);

        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(vg, vg), vg);
        __m256 inner = _mm256_mul_ps(sqrt_2_over_pi, _mm256_fmadd_ps(coef, x3, vg));

        float tmp_in[8];
        float tmp_tanh[8];
        _mm256_storeu_ps(tmp_in, inner);
        for (int k = 0; k < 8; ++k) {
            tmp_tanh[k] = tanhf(tmp_in[k]);
        }
        __m256 v_tanh = _mm256_loadu_ps(tmp_tanh);

        __m256 gelu_val = _mm256_mul_ps(half, _mm256_mul_ps(vg, _mm256_add_ps(one, v_tanh)));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(gelu_val, vu));
    }

    for (; i < dim; ++i) {
        float x = gate[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float gelu_val = 0.5f * x * (1.0f + tanhf(inner));
        out[i] = gelu_val * up[i];
    }
}

// -----------------------------------------------------------------------------
// Activation Quantization to Q8_K
// -----------------------------------------------------------------------------
static inline void quantize_row_q8_k_zen3(
    const float* __restrict src,
    block_q8_K*  __restrict dst_all,
    int k)
{
    const int nb = k / QK_K;

    for (int i = 0; i < nb; ++i) {
        const float* s = src + i * QK_K;
        block_q8_K* dst = dst_all + i;

        __m256 v_max = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 8) {
            __m256 v = _mm256_loadu_ps(s + j);
            __m256 v_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
            v_max = _mm256_max_ps(v_max, v_abs);
        }
        float max_abs = hsum_float_8(v_max);

        if (max_abs == 0.0f) {
            dst->d = 0.0f;
            memset(dst->qs, 0, QK_K);
            memset(dst->bsums, 0, sizeof(dst->bsums));
            continue;
        }

        const float iscale = 127.0f / max_abs;
        dst->d = 1.0f / iscale;
        const __m256 v_iscale = _mm256_set1_ps(iscale);

        for (int j = 0; j < QK_K; j += 8) {
            __m256 v = _mm256_loadu_ps(s + j);
            __m256 v_scaled = _mm256_mul_ps(v, v_iscale);
            __m256i v_int = _mm256_cvtps_epi32(_mm256_round_ps(v_scaled, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));

            __m128i lo = _mm256_castsi256_si128(v_int);
            __m128i hi = _mm256_extracti128_si256(v_int, 1);
            __m128i packed16 = _mm_packs_epi32(lo, hi);
            __m128i packed8 = _mm_packs_epi16(packed16, packed16);

            uint64_t bytes8 = (uint64_t)_mm_cvtsi128_si64(packed8);
            memcpy(&dst->qs[j], &bytes8, sizeof(bytes8));
        }

        for (int b = 0; b < QK_K / 16; ++b) {
            int16_t sum = 0;
            for (int sb = 0; sb < 16; ++sb) {
                sum += dst->qs[b * 16 + sb];
            }
            dst->bsums[b] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// AVX2 1-Token Q6_K Dot Product
// -----------------------------------------------------------------------------
static inline float vec_dot_q6_k_q8_k_zen3(
    const block_q6_K* __restrict x,
    const block_q8_K* __restrict y,
    int k)
{
    const int nb = k / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i s32 = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        const int8_t*  q8 = y[i].qs;

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < 4; ++j) {
            const __m256i ql_bytes = _mm256_loadu_si256((const __m256i*)ql); ql += 32;
            const __m256i qh_bytes = _mm256_set1_epi64x(*(const int64_t*)(qh + j * 8));

            const __m256i ql0 = _mm256_and_si256(ql_bytes, m15);
            const __m256i ql1 = _mm256_and_si256(_mm256_srli_epi16(ql_bytes, 4), m15);

            const __m256i qh0 = _mm256_and_si256(qh_bytes, m3);
            const __m256i qh1 = _mm256_and_si256(_mm256_srli_epi16(qh_bytes, 2), m3);

            const __m256i w0 = _mm256_sub_epi8(_mm256_or_si256(ql0, _mm256_slli_epi16(qh0, 4)), s32);
            const __m256i w1 = _mm256_sub_epi8(_mm256_or_si256(ql1, _mm256_slli_epi16(qh1, 4)), s32);

            const __m256i a0 = _mm256_loadu_si256((const __m256i*)q8);
            const __m256i a1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            q8 += 64;

            const __m256i scale0 = _mm256_set1_epi16(sc[j * 4 + 0]);
            const __m256i scale1 = _mm256_set1_epi16(sc[j * 4 + 1]);

            const __m256i p0 = _mm256_madd_epi16(scale0, _mm256_maddubs_epi16(w0, a0));
            const __m256i p1 = _mm256_madd_epi16(scale1, _mm256_maddubs_epi16(w1, a1));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return hsum_float_8(acc);
}

// -----------------------------------------------------------------------------
// AVX2 4-Token Unrolled Q6_K Dot Product (Shared Weight Unpack in L1 Cache!)
// -----------------------------------------------------------------------------
static inline void vec_dot_q6_k_q8_k_4x(
    const block_q6_K* __restrict x,
    const block_q8_K* __restrict y0,
    const block_q8_K* __restrict y1,
    const block_q8_K* __restrict y2,
    const block_q8_K* __restrict y3,
    int k,
    float* res0, float* res1, float* res2, float* res3)
{
    const int nb = k / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i s32 = _mm256_set1_epi8(32);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d_w = fp16_to_fp32(x[i].d);
        const float d0 = y0[i].d * d_w;
        const float d1 = y1[i].d * d_w;
        const float d2 = y2[i].d * d_w;
        const float d3 = y3[i].d * d_w;

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;

        const int8_t*  q8_0 = y0[i].qs;
        const int8_t*  q8_1 = y1[i].qs;
        const int8_t*  q8_2 = y2[i].qs;
        const int8_t*  q8_3 = y3[i].qs;

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        __m256i sumi3 = _mm256_setzero_si256();

        for (int j = 0; j < 4; ++j) {
            const __m256i ql_bytes = _mm256_loadu_si256((const __m256i*)ql); ql += 32;
            const __m256i qh_bytes = _mm256_set1_epi64x(*(const int64_t*)(qh + j * 8));

            const __m256i ql0 = _mm256_and_si256(ql_bytes, m15);
            const __m256i ql1 = _mm256_and_si256(_mm256_srli_epi16(ql_bytes, 4), m15);

            const __m256i qh0 = _mm256_and_si256(qh_bytes, m3);
            const __m256i qh1 = _mm256_and_si256(_mm256_srli_epi16(qh_bytes, 2), m3);

            // Weights unpacked ONCE for all 4 tokens!
            const __m256i w0 = _mm256_sub_epi8(_mm256_or_si256(ql0, _mm256_slli_epi16(qh0, 4)), s32);
            const __m256i w1 = _mm256_sub_epi8(_mm256_or_si256(ql1, _mm256_slli_epi16(qh1, 4)), s32);

            const __m256i scale0 = _mm256_set1_epi16(sc[j * 4 + 0]);
            const __m256i scale1 = _mm256_set1_epi16(sc[j * 4 + 1]);

            // Token 0
            const __m256i a0_0 = _mm256_loadu_si256((const __m256i*)q8_0);
            const __m256i a1_0 = _mm256_loadu_si256((const __m256i*)(q8_0 + 32));
            q8_0 += 64;
            const __m256i p0_0 = _mm256_madd_epi16(scale0, _mm256_maddubs_epi16(w0, a0_0));
            const __m256i p1_0 = _mm256_madd_epi16(scale1, _mm256_maddubs_epi16(w1, a1_0));
            sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p0_0, p1_0));

            // Token 1
            const __m256i a0_1 = _mm256_loadu_si256((const __m256i*)q8_1);
            const __m256i a1_1 = _mm256_loadu_si256((const __m256i*)(q8_1 + 32));
            q8_1 += 64;
            const __m256i p0_1 = _mm256_madd_epi16(scale0, _mm256_maddubs_epi16(w0, a0_1));
            const __m256i p1_1 = _mm256_madd_epi16(scale1, _mm256_maddubs_epi16(w1, a1_1));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p0_1, p1_1));

            // Token 2
            const __m256i a0_2 = _mm256_loadu_si256((const __m256i*)q8_2);
            const __m256i a1_2 = _mm256_loadu_si256((const __m256i*)(q8_2 + 32));
            q8_2 += 64;
            const __m256i p0_2 = _mm256_madd_epi16(scale0, _mm256_maddubs_epi16(w0, a0_2));
            const __m256i p1_2 = _mm256_madd_epi16(scale1, _mm256_maddubs_epi16(w1, a1_2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p0_2, p1_2));

            // Token 3
            const __m256i a0_3 = _mm256_loadu_si256((const __m256i*)q8_3);
            const __m256i a1_3 = _mm256_loadu_si256((const __m256i*)(q8_3 + 32));
            q8_3 += 64;
            const __m256i p0_3 = _mm256_madd_epi16(scale0, _mm256_maddubs_epi16(w0, a0_3));
            const __m256i p1_3 = _mm256_madd_epi16(scale1, _mm256_maddubs_epi16(w1, a1_3));
            sumi3 = _mm256_add_epi32(sumi3, _mm256_add_epi32(p0_3, p1_3));
        }

        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d0), _mm256_cvtepi32_ps(sumi0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d1), _mm256_cvtepi32_ps(sumi1), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d2), _mm256_cvtepi32_ps(sumi2), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d3), _mm256_cvtepi32_ps(sumi3), acc3);
    }

    *res0 = hsum_float_8(acc0);
    *res1 = hsum_float_8(acc1);
    *res2 = hsum_float_8(acc2);
    *res3 = hsum_float_8(acc3);
}

// -----------------------------------------------------------------------------
// AVX2 Q8_0 Dot Product with FP32 Activations
// -----------------------------------------------------------------------------
static inline float vec_dot_q8_0_f32_zen3(
    const block_q8_0* __restrict x,
    const float*      __restrict y,
    int k)
{
    const int nb = k / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        float d = fp16_to_fp32(x[i].d);
        __m256 vd = _mm256_set1_ps(d);

        __m128i q_lo = _mm_loadu_si128((const __m128i*)x[i].qs);
        __m128i q_hi = _mm_loadu_si128((const __m128i*)(x[i].qs + 16));

        __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
        __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));
        __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
        __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));

        __m256 y0 = _mm256_loadu_ps(y + i * 32 + 0);
        __m256 y1 = _mm256_loadu_ps(y + i * 32 + 8);
        __m256 y2 = _mm256_loadu_ps(y + i * 32 + 16);
        __m256 y3 = _mm256_loadu_ps(y + i * 32 + 24);

        __m256 p0 = _mm256_mul_ps(_mm256_mul_ps(w0, y0), vd);
        __m256 p1 = _mm256_mul_ps(_mm256_mul_ps(w1, y1), vd);
        __m256 p2 = _mm256_mul_ps(_mm256_mul_ps(w2, y2), vd);
        __m256 p3 = _mm256_mul_ps(_mm256_mul_ps(w3, y3), vd);

        acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3)));
    }

    return hsum_float_8(acc);
}

// -----------------------------------------------------------------------------
// AVX2 BF16 Dot Product with FP32 Activations (for per_layer_model_proj)
// -----------------------------------------------------------------------------
static inline float vec_dot_bf16_f32_zen3(
    const uint16_t* __restrict x,
    const float*    __restrict y,
    int k)
{
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= k; i += 8) {
        __m128i bx = _mm_loadu_si128((const __m128i*)(x + i));
        __m256i bx32 = _mm256_cvtepu16_epi32(bx);
        __m256i f32_bits = _mm256_slli_epi32(bx32, 16);
        __m256 fx = _mm256_castsi256_ps(f32_bits);

        __m256 vy = _mm256_loadu_ps(y + i);
        acc = _mm256_fmadd_ps(fx, vy, acc);
    }
    float s = hsum_float_8(acc);
    for (; i < k; ++i) {
        s += bf16_to_fp32(x[i]) * y[i];
    }
    return s;
}

// -----------------------------------------------------------------------------
// Tiled 4-Token Unrolled GEMM Kernels (Peak L1 Cache Residency & Zero Allocations)
// -----------------------------------------------------------------------------
static inline void gemm_chunk_q6_k_q8_k_zen3(
    const block_q6_K* __restrict weight, // [N_rows, K/QK_K]
    const block_q8_K* __restrict act,    // [M_tokens, K/QK_K]
    float*            __restrict out,    // [M_tokens, N_rows]
    int M_tokens,
    int N_rows,
    int K,
    int r_start,
    int r_end)
{
    const int nb = K / QK_K;
    for (int r = r_start; r < r_end; ++r) {
        const block_q6_K* wr = weight + r * nb;
        int t = 0;
        for (; t + 4 <= M_tokens; t += 4) {
            float r0, r1, r2, r3;
            vec_dot_q6_k_q8_k_4x(wr, 
                act + (t + 0) * nb,
                act + (t + 1) * nb,
                act + (t + 2) * nb,
                act + (t + 3) * nb,
                K, &r0, &r1, &r2, &r3);
            out[(t + 0) * N_rows + r] = r0;
            out[(t + 1) * N_rows + r] = r1;
            out[(t + 2) * N_rows + r] = r2;
            out[(t + 3) * N_rows + r] = r3;
        }
        for (; t < M_tokens; ++t) {
            out[t * N_rows + r] = vec_dot_q6_k_q8_k_zen3(wr, act + t * nb, K);
        }
    }
}

static inline void gemm_chunk_q8_0_f32_zen3(
    const block_q8_0* __restrict weight, // [N_rows, K/32]
    const float*      __restrict act,    // [M_tokens, K]
    float*            __restrict out,    // [M_tokens, N_rows]
    int M_tokens,
    int N_rows,
    int K,
    int r_start,
    int r_end)
{
    const int nb = K / 32;
    for (int r = r_start; r < r_end; ++r) {
        const block_q8_0* wr = weight + r * nb;
        for (int t = 0; t < M_tokens; ++t) {
            out[t * N_rows + r] = vec_dot_q8_0_f32_zen3(wr, act + t * K, K);
        }
    }
}

static inline void gemm_chunk_bf16_f32_zen3(
    const uint16_t* __restrict weight, // [N_rows, K]
    const float*    __restrict act,    // [M_tokens, K]
    float*          __restrict out,    // [M_tokens, N_rows]
    int M_tokens,
    int N_rows,
    int K,
    int r_start,
    int r_end)
{
    for (int r = r_start; r < r_end; ++r) {
        const uint16_t* wr = weight + r * K;
        for (int t = 0; t < M_tokens; ++t) {
            out[t * N_rows + r] = vec_dot_bf16_f32_zen3(wr, act + t * K, K);
        }
    }
}

} // namespace embed_gemma
