#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdint>

#define QK_K 256

#pragma pack(push, 1)
struct block_q6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
};
#pragma pack(pop)

static inline float fp16_to_fp32(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    return _mm_cvtss_f32(_mm_cvtph_ps(v));
}

static inline float hsum_float_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

// 4-token unrolled Q6_K dot product
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
            // Weights loaded from L1 cache
            const __m256i ql_bytes = _mm256_loadu_si256((const __m256i*)ql); ql += 32;
            const __m256i qh_bytes = _mm256_set1_epi64x(*(const int64_t*)(qh + j * 8));

            const __m256i ql0 = _mm256_and_si256(ql_bytes, m15);
            const __m256i ql1 = _mm256_and_si256(_mm256_srli_epi16(ql_bytes, 4), m15);

            const __m256i qh0 = _mm256_and_si256(qh_bytes, m3);
            const __m256i qh1 = _mm256_and_si256(_mm256_srli_epi16(qh_bytes, 2), m3);

            // Unpacked weights (shared across all 4 tokens!)
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

int main() {
    int K = 512;
    int N_rows = 2048;
    int M_tokens = 512;
    int nb = K / QK_K;

    std::vector<block_q6_K> weights(N_rows * nb);
    std::vector<block_q8_K> acts(M_tokens * nb);
    std::vector<float> out(M_tokens * N_rows, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 20; ++rep) {
        for (int r = 0; r < N_rows; ++r) {
            const block_q6_K* wr = weights.data() + r * nb;
            for (int t = 0; t < M_tokens; t += 4) {
                float r0, r1, r2, r3;
                vec_dot_q6_k_q8_k_4x(wr, 
                    &acts[(t+0)*nb], &acts[(t+1)*nb], &acts[(t+2)*nb], &acts[(t+3)*nb], 
                    K, &r0, &r1, &r2, &r3);
                out[(t+0)*N_rows + r] = r0;
                out[(t+1)*N_rows + r] = r1;
                out[(t+2)*N_rows + r] = r2;
                out[(t+3)*N_rows + r] = r3;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0;
    std::cout << "4x Unrolled GEMM single-thread time: " << ms << " ms for [512, 2048] x 512" << std::endl;
    return 0;
}
