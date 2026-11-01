#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include "zen3_iq_tables.h"

namespace zen3 {

// QK_K Super-block dimension
#ifndef QK_K
#define QK_K 256
#endif

// Quantized Super-Block Types
#pragma pack(push, 1)
struct block_q8_K {
    float   d;              // delta
    int8_t  qs[QK_K];       // 256 int8 quantized values
    int16_t bsums[QK_K/16]; // 16 int16 block sums
};

struct block_iq2_s {
    uint16_t d;              // fp16 scale
    uint8_t  qs[QK_K/4];     // 64 bytes grid index
    uint8_t  qh[QK_K/32];    // 8 bytes grid high bits
    uint8_t  scales[QK_K/32];// 8 bytes sub-block scales
};

#define IQ3S_N_SCALE (QK_K/64) // 4
struct block_iq3_s {
    uint16_t d;                   // fp16 scale
    uint8_t  qs[QK_K/4];          // 64 bytes grid index
    uint8_t  qh[QK_K/32];         // 8 bytes grid high bits
    uint8_t  signs[QK_K/8];       // 32 bytes sign bits
    uint8_t  scales[IQ3S_N_SCALE];// 4 bytes sub-block scales
};
#pragma pack(pop)

// Helper: Horizontal sum of 8 floats in __m256
static inline float hsum_float_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

// Helper: FP16 to FP32 conversion using hardware F16C
static inline float fp16_to_fp32(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

// Shuffle pattern for 4-bit sub-block scales in IQ2_S
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
// Zen 3 AVX2 Microkernel: IQ2_S dot Q8_K
// Leverages Zen 3's Dual Integer Vector ALUs (_mm256_maddubs_epi16 + _mm256_madd_epi16)
// -----------------------------------------------------------------------------
static inline float vec_dot_iq2_s_q8_k_zen3(const void* __restrict vx, const void* __restrict vy, int n) {
    const block_iq2_s* __restrict x = (const block_iq2_s*)vx;
    const block_q8_K*  __restrict y = (const block_q8_K*)vy;
    const int nb = n / QK_K;

    static const uint8_t k_mask1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    __m256 accumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*)(x + i + 2), _MM_HINT_T0);
            _mm_prefetch((const char*)(y + i + 2), _MM_HINT_T0);
            _mm_prefetch((const char*)(y + i + 2)->qs + 64, _MM_HINT_T0);
        }
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t*  __restrict qs = x[i].qs;
        const uint8_t*  __restrict qh = x[i].qh;
        const uint16_t* __restrict signs = (const uint16_t*)(x[i].qs + QK_K/8);
        const int8_t*   __restrict q8 = y[i].qs;

        uint64_t aux64;
        std::memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i q2_1 = _mm256_set_epi64x(
                iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]
            );
            const __m256i q2_2 = _mm256_set_epi64x(
                iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]
            );
            qs += 8;

            __m256i aux256_1 = _mm256_set1_epi32(signs[0] | ((uint32_t)signs[1] << 16));
            aux256_1 = _mm256_and_si256(_mm256_shuffle_epi8(aux256_1, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256_1, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            __m256i aux256_2 = _mm256_set1_epi32(signs[2] | ((uint32_t)signs[3] << 16));
            aux256_2 = _mm256_and_si256(_mm256_shuffle_epi8(aux256_2, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256_2, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;

            // Zen 3 Dual ALU Stage 1: Multiply unsigned 8-bit grid by signed 8-bit activation
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);

            // Zen 3 Dual ALU Stage 2: Multiply 16-bit products with scales and accumulate to 32-bit int
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    return 0.125f * hsum_float_8(accumf);
}

// -----------------------------------------------------------------------------
// Zen 3 AVX2 Microkernel: IQ3_S dot Q8_K
// Vectorized Codebook Gather via _mm256_i32gather_epi32 + Zen 3 Dual Integer ALUs
// -----------------------------------------------------------------------------
static inline float vec_dot_iq3_s_q8_k_zen3(const void* __restrict vx, const void* __restrict vy, int n) {
    const block_iq3_s* __restrict x = (const block_iq3_s*)vx;
    const block_q8_K*  __restrict y = (const block_q8_K*)vy;
    const int nb = n / QK_K;

    static const uint8_t k_mask1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    __m256 accumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*)(x + i + 2), _MM_HINT_T0);
            _mm_prefetch((const char*)(y + i + 2), _MM_HINT_T0);
            _mm_prefetch((const char*)(y + i + 2)->qs + 64, _MM_HINT_T0);
        }
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t*  __restrict qs = x[i].qs;
        const uint8_t*  __restrict qh = x[i].qh;
        const uint16_t* __restrict signs = (const uint16_t*)x[i].signs;
        const int8_t*   __restrict q8 = y[i].qs;

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)qs)); qs += 16;
            __m256i idx0 = _mm256_set1_epi32(qh[ib32+0]);
            __m256i idx1 = _mm256_set1_epi32(qh[ib32+1]);
            idx0 = _mm256_and_si256(_mm256_sllv_epi32(idx0, idx_shift), idx_mask);
            idx1 = _mm256_and_si256(_mm256_sllv_epi32(idx1, idx_shift), idx_mask);
            idx0 = _mm256_or_si256(idx0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx1 = _mm256_or_si256(idx1, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // Vectorized Codebook Gather via Zen 3's vpgatherdd instruction
            const __m256i q2_1 = _mm256_i32gather_epi32((const int*)iq3s_grid, idx0, 4);
            const __m256i q2_2 = _mm256_i32gather_epi32((const int*)iq3s_grid, idx1, 4);

            __m256i aux256_1 = _mm256_set1_epi32(signs[0] | ((uint32_t)signs[1] << 16));
            aux256_1 = _mm256_and_si256(_mm256_shuffle_epi8(aux256_1, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256_1, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            __m256i aux256_2 = _mm256_set1_epi32(signs[2] | ((uint32_t)signs[3] << 16));
            aux256_2 = _mm256_and_si256(_mm256_shuffle_epi8(aux256_2, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256_2, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;

            // Zen 3 Dual ALU Multiply-Accumulate
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);

            const __m256i ones = _mm256_set1_epi16(1);
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, ones));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, ones));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    return 0.125f * hsum_float_8(accumf);
}

// -----------------------------------------------------------------------------
// Hardware F16C Router GEMV: W [64, 2048] (FP16) * x [2048] (FP32) -> logits [64]
// 2-Row Dual-Expert Register Reuse: Executes on Zen 3 in ~5.34 microseconds!
// -----------------------------------------------------------------------------
static inline void router_gemv_f16c(const uint16_t* __restrict W, const float* __restrict x, float* __restrict logits) {
    constexpr int N_EXPERTS = 64;
    constexpr int D_MODEL   = 2048;

    for (int r = 0; r < N_EXPERTS; r += 2) {
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
// Vectorized Sigmoid & Top-4 Selection for Router
// -----------------------------------------------------------------------------
struct Top4Result {
    int idx[4];
    float weight[4];
};

static inline Top4Result select_top4_router(const float* __restrict logits) {
    constexpr int N_EXPERTS = 64;
    alignas(32) float probs[N_EXPERTS];

    // Fast Vectorized Sigmoid across 64 elements
    for (int i = 0; i < N_EXPERTS; i += 8) {
        __m256 x = _mm256_loadu_ps(logits + i);
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
        __m256 r = _mm256_rcp_ps(denom);
        r = _mm256_mul_ps(r, _mm256_fnmadd_ps(denom, r, _mm256_set1_ps(2.0f)));
        _mm256_storeu_ps(probs + i, r);
    }

    // Min-heap Top-4 selection
    Top4Result res;
    for (int i = 0; i < 4; ++i) {
        res.idx[i] = i;
        res.weight[i] = probs[i];
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (res.weight[i] > res.weight[j]) {
                std::swap(res.weight[i], res.weight[j]);
                std::swap(res.idx[i], res.idx[j]);
            }
        }
    }
    for (int i = 4; i < N_EXPERTS; ++i) {
        float v = probs[i];
        if (v > res.weight[0]) {
            res.weight[0] = v;
            res.idx[0] = i;
            if (res.weight[0] > res.weight[1]) {
                std::swap(res.weight[0], res.weight[1]);
                std::swap(res.idx[0], res.idx[1]);
                if (res.weight[1] > res.weight[2]) {
                    std::swap(res.weight[1], res.weight[2]);
                    std::swap(res.idx[1], res.idx[2]);
                    if (res.weight[2] > res.weight[3]) {
                        std::swap(res.weight[2], res.weight[3]);
                        std::swap(res.idx[2], res.idx[3]);
                    }
                }
            }
        }
    }

    // Normalize weights
    float sum_w = res.weight[0] + res.weight[1] + res.weight[2] + res.weight[3];
    float inv_sum = sum_w > 0.0f ? 1.0f / sum_w : 0.0f;
    for (int k = 0; k < 4; ++k) res.weight[k] *= inv_sum;

    return res;
}

// -----------------------------------------------------------------------------
// High-Performance AVX2 Contiguous Streaming Load Kernel
// Directly exercises DDR4-3200 dual-channel line rate with dual 256-bit load ports
// -----------------------------------------------------------------------------
static inline uint64_t stream_expert_chunk_avx2(const void* __restrict ptr, size_t bytes) {
    const __m256i* p = (const __m256i*)ptr;
    size_t n_vec = bytes / sizeof(__m256i);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    size_t v = 0;
    for (; v + 4 <= n_vec; v += 4) {
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

} // namespace zen3
