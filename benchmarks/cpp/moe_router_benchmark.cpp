#include <windows.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>

// Architectural Constants for LFM 2.5 24B A2B (lfm2moe)
constexpr int D_MODEL = 2048;
constexpr int N_EXPERTS = 64;
constexpr int TOP_K = 4;
constexpr int N_MOE_LAYERS = 38;
constexpr int EXPERT_FFN_DIM = 1536;

// Expert sizes in bytes (from parsed GGUF UD-IQ3_XXS)
constexpr size_t GATE_EXP_BYTES = 1007616;  // IQ2_S [2048, 1536]
constexpr size_t UP_EXP_BYTES   = 1007616;  // IQ2_S [2048, 1536]
constexpr size_t DOWN_EXP_BYTES = 1351680;  // IQ3_S [1536, 2048]
constexpr size_t ONE_EXPERT_BYTES = GATE_EXP_BYTES + UP_EXP_BYTES + DOWN_EXP_BYTES; // 3,366,912 bytes
constexpr size_t LAYER_ALL_EXPERTS_BYTES = 64 * ONE_EXPERT_BYTES; // 215,482,368 bytes (~205.5 MB)
constexpr size_t ACTIVE_4_EXPERTS_BYTES = 4 * ONE_EXPERT_BYTES;   // 13,467,648 bytes (~12.84 MB)

// Q8_0 definitions
#define QK8_0 32
struct block_q8_0 {
    uint16_t d;        // fp16 scale
    int8_t qs[QK8_0];  // 32 int8 weights
};

// FP16 helper: convert float to uint16_t fp16 using F16C
static inline uint16_t float_to_fp16(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, 0);
    return (uint16_t)_mm_cvtsi128_si32(h);
}

// Timing utilities
struct Timer {
    LARGE_INTEGER freq;
    LARGE_INTEGER start_time;

    Timer() {
        QueryPerformanceFrequency(&freq);
    }

    void start() {
        QueryPerformanceCounter(&start_time);
    }

    double stop_us() const {
        LARGE_INTEGER end_time;
        QueryPerformanceCounter(&end_time);
        return (double)(end_time.QuadPart - start_time.QuadPart) * 1e6 / freq.QuadPart;
    }

    double stop_ns() const {
        LARGE_INTEGER end_time;
        QueryPerformanceCounter(&end_time);
        return (double)(end_time.QuadPart - start_time.QuadPart) * 1e9 / freq.QuadPart;
    }
};

static inline uint64_t rdtsc_fence() {
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

// Latency statistics helper
struct BenchmarkStats {
    double min_us = 1e9;
    double max_us = 0.0;
    double mean_us = 0.0;
    double median_us = 0.0;
    double p90_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double stddev_us = 0.0;
    double cycles_mean = 0.0;

    static BenchmarkStats compute(std::vector<double>& times_us, const std::vector<uint64_t>& cycles) {
        BenchmarkStats s;
        if (times_us.empty()) return s;
        std::sort(times_us.begin(), times_us.end());
        s.min_us = times_us.front();
        s.max_us = times_us.back();
        double sum = 0.0;
        for (double t : times_us) sum += t;
        s.mean_us = sum / times_us.size();
        s.median_us = times_us[times_us.size() / 2];
        s.p90_us = times_us[(size_t)(times_us.size() * 0.90)];
        s.p95_us = times_us[(size_t)(times_us.size() * 0.95)];
        s.p99_us = times_us[(size_t)(times_us.size() * 0.99)];

        double var = 0.0;
        for (double t : times_us) {
            double diff = t - s.mean_us;
            var += diff * diff;
        }
        s.stddev_us = std::sqrt(var / times_us.size());

        if (!cycles.empty()) {
            double c_sum = 0.0;
            for (uint64_t c : cycles) c_sum += (double)c;
            s.cycles_mean = c_sum / cycles.size();
        }
        return s;
    }

    void print(const char* label, double bytes_transferred = 0.0) const {
        std::cout << std::left << std::setw(32) << label << ": "
                  << "Mean: " << std::right << std::setw(7) << std::fixed << std::setprecision(2) << mean_us << " us "
                  << "(Median: " << std::setw(7) << median_us << " us, "
                  << "P99: " << std::setw(7) << p99_us << " us, "
                  << "Min: " << std::setw(7) << min_us << " us) "
                  << " | " << std::setw(8) << (int)cycles_mean << " cyc";
        if (bytes_transferred > 0.0 && mean_us > 0.0) {
            double gb_s = (bytes_transferred / (mean_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
            std::cout << " | " << std::setw(6) << std::setprecision(1) << gb_s << " GB/s";
        }
        std::cout << "\n";
    }
};

// -------------------------------------------------------------
// AVX2 Vector Helpers
// -------------------------------------------------------------
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

// Fast AVX2 exponential approximation
static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    
    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2f   = _mm256_set1_ps(0.6931471805599453f);
    
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2ef), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 y = _mm256_fnmadd_ps(fx, ln2f, x);
    
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
    __m256 pow2n = _mm256_castsi256_ps(emm0);
    
    return _mm256_mul_ps(p, pow2n);
}

// Vectorized Sigmoid: 1.0f / (1.0f + exp(-x))
static inline __m256 sigmoid256_ps(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = exp256_ps(neg_x);
    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg);
    // Reciprocal with 1 Newton-Raphson iteration
    __m256 r = _mm256_rcp_ps(denom);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(denom, r, _mm256_set1_ps(2.0f)));
    return r;
}

// Multiply int8_t pairs and accumulate as float
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);

    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);

    const __m128i dot_l = _mm_maddubs_epi16(axl, syl);
    const __m128i dot_h = _mm_maddubs_epi16(axh, syh);

    const __m128i ones = _mm_set1_epi16(1);
    const __m128i sum32_l = _mm_madd_epi16(dot_l, ones);
    const __m128i sum32_h = _mm_madd_epi16(dot_h, ones);

    __m256i sum32 = _mm256_set_m128i(sum32_h, sum32_l);
    return _mm256_cvtepi32_ps(sum32);
}

// -------------------------------------------------------------
// GEMV Kernels (64 x 2048)
// -------------------------------------------------------------

// FP32 GEMV: W [64, 2048] * x [2048] -> logits [64]
void gemv_fp32(const float* __restrict W, const float* __restrict x, float* __restrict logits) {
    for (int r = 0; r < N_EXPERTS; ++r) {
        const float* row = W + r * D_MODEL;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int c = 0; c < D_MODEL; c += 32) {
            __m256 w0 = _mm256_loadu_ps(row + c + 0);
            __m256 x0 = _mm256_loadu_ps(x + c + 0);
            acc0 = _mm256_fmadd_ps(w0, x0, acc0);

            __m256 w1 = _mm256_loadu_ps(row + c + 8);
            __m256 x1 = _mm256_loadu_ps(x + c + 8);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);

            __m256 w2 = _mm256_loadu_ps(row + c + 16);
            __m256 x2 = _mm256_loadu_ps(x + c + 16);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);

            __m256 w3 = _mm256_loadu_ps(row + c + 24);
            __m256 x3 = _mm256_loadu_ps(x + c + 24);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }
        __m256 sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        logits[r] = hsum256_ps(sum);
    }
}

// FP16 GEMV: W [64, 2048] (uint16_t) * x [2048] (float) -> logits [64]
void gemv_fp16(const uint16_t* __restrict W, const float* __restrict x, float* __restrict logits) {
    for (int r = 0; r < N_EXPERTS; ++r) {
        const uint16_t* row = W + r * D_MODEL;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int c = 0; c < D_MODEL; c += 32) {
            __m128i h0 = _mm_loadu_si128((const __m128i*)(row + c + 0));
            __m256 w0 = _mm256_cvtph_ps(h0);
            __m256 x0 = _mm256_loadu_ps(x + c + 0);
            acc0 = _mm256_fmadd_ps(w0, x0, acc0);

            __m128i h1 = _mm_loadu_si128((const __m128i*)(row + c + 8));
            __m256 w1 = _mm256_cvtph_ps(h1);
            __m256 x1 = _mm256_loadu_ps(x + c + 8);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);

            __m128i h2 = _mm_loadu_si128((const __m128i*)(row + c + 16));
            __m256 w2 = _mm256_cvtph_ps(h2);
            __m256 x2 = _mm256_loadu_ps(x + c + 16);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);

            __m128i h3 = _mm_loadu_si128((const __m128i*)(row + c + 24));
            __m256 w3 = _mm256_cvtph_ps(h3);
            __m256 x3 = _mm256_loadu_ps(x + c + 24);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }
        __m256 sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        logits[r] = hsum256_ps(sum);
    }
}

// Q8_0 quantize input vector x (2048 elements)
void quantize_row_q8_0(const float* __restrict x, block_q8_0* __restrict y, int k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float v = std::abs(x[i * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = float_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) {
            float v0 = x[i * QK8_0 + j] * id;
            y[i].qs[j] = (int8_t)std::round(v0);
        }
    }
}

// Q8_0 dot Q8_0 GEMV (quantized x dot quantized weights)
void gemv_q8_0_q8_0(const block_q8_0* __restrict W, const block_q8_0* __restrict qx, float* __restrict logits) {
    constexpr int nb = D_MODEL / QK8_0; // 64 blocks
    for (int r = 0; r < N_EXPERTS; ++r) {
        const block_q8_0* row = W + r * nb;
        __m256 acc = _mm256_setzero_ps();

        for (int ib = 0; ib < nb; ++ib) {
            float scale_w = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(row[ib].d)));
            float scale_x = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(qx[ib].d)));
            __m256 d = _mm256_set1_ps(scale_w * scale_x);

            __m256i qw = _mm256_loadu_si256((const __m256i*)row[ib].qs);
            __m256i qx_vec = _mm256_loadu_si256((const __m256i*)qx[ib].qs);

            __m256 q = mul_sum_i8_pairs_float(qw, qx_vec);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        logits[r] = hsum256_ps(acc);
    }
}

// -------------------------------------------------------------
// Vectorized Sigmoid and Top-4 Selection
// -------------------------------------------------------------

// Vectorized Sigmoid across 64 elements
void sigmoid_64_avx2(const float* __restrict logits, float* __restrict probs) {
    for (int i = 0; i < N_EXPERTS; i += 8) {
        __m256 x = _mm256_loadu_ps(logits + i);
        __m256 p = sigmoid256_ps(x);
        _mm256_storeu_ps(probs + i, p);
    }
}

// Scalar Sigmoid for baseline
void sigmoid_64_scalar(const float* __restrict logits, float* __restrict probs) {
    for (int i = 0; i < N_EXPERTS; ++i) {
        probs[i] = 1.0f / (1.0f + std::exp(-logits[i]));
    }
}

// Bias addition: selection_probs = probs + exp_probs_b
static inline void add_bias_64_avx2(const float* __restrict probs, const float* __restrict bias, float* __restrict sel_probs) {
    for (int i = 0; i < N_EXPERTS; i += 8) {
        __m256 p = _mm256_loadu_ps(probs + i);
        __m256 b = _mm256_loadu_ps(bias + i);
        _mm256_storeu_ps(sel_probs + i, _mm256_add_ps(p, b));
    }
}

struct Top4Result {
    int idx[TOP_K];
    float weight[TOP_K];
};

// Fast 4-element insertion min-heap for Top-4 out of 64
Top4Result select_top4_heap(const float* __restrict sel_probs, const float* __restrict raw_probs) {
    Top4Result res;
    for (int i = 0; i < TOP_K; ++i) {
        res.idx[i] = i;
        res.weight[i] = sel_probs[i];
    }
    // Sort 4 initial elements ascending
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (res.weight[i] > res.weight[j]) {
                std::swap(res.weight[i], res.weight[j]);
                std::swap(res.idx[i], res.idx[j]);
            }
        }
    }
    // Scan remaining 60 elements
    for (int i = 4; i < N_EXPERTS; ++i) {
        float v = sel_probs[i];
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

    // Retrieve raw probs and normalize
    float sum_w = 0.0f;
    for (int k = 0; k < TOP_K; ++k) {
        res.weight[k] = raw_probs[res.idx[k]];
        sum_w += res.weight[k];
    }
    float inv_sum = sum_w > 0.0f ? 1.0f / sum_w : 0.0f;
    for (int k = 0; k < TOP_K; ++k) {
        res.weight[k] *= inv_sum;
    }
    return res;
}

// std::partial_sort for comparison
Top4Result select_top4_std(const float* __restrict sel_probs, const float* __restrict raw_probs) {
    int indices[64];
    for (int i = 0; i < 64; ++i) indices[i] = i;
    std::partial_sort(indices, indices + 4, indices + 64, [&](int a, int b) {
        return sel_probs[a] > sel_probs[b];
    });
    Top4Result res;
    float sum_w = 0.0f;
    for (int k = 0; k < 4; ++k) {
        res.idx[k] = indices[k];
        res.weight[k] = raw_probs[indices[k]];
        sum_w += res.weight[k];
    }
    float inv_sum = sum_w > 0.0f ? 1.0f / sum_w : 0.0f;
    for (int k = 0; k < 4; ++k) res.weight[k] *= inv_sum;
    return res;
}

// L3 Cache Eviction Helper (streams 64 MB dummy buffer to flush 16 MB L3)
void flush_l3_cache(const uint8_t* dummy, size_t size, volatile uint64_t* sink) {
    uint64_t s = 0;
    const uint64_t* p = (const uint64_t*)dummy;
    size_t count = size / sizeof(uint64_t);
    for (size_t i = 0; i < count; i += 8) { // 64-byte cache line stride
        s += p[i];
    }
    *sink += s;
}

// -------------------------------------------------------------
// Main Benchmark Runner
// -------------------------------------------------------------
int main() {
    std::cout << "================================================================================\n";
    std::cout << "  LFM 2.5 24B A2B (lfm2moe) MoE Router Microbenchmark & DDR4 Analysis\n";
    std::cout << "  Hardware: AMD Zen 3 (Ryzen 7 7730U, 8C/16T, 16MB L3, Dual DDR4-3200)\n";
    std::cout << "================================================================================\n\n";

    constexpr int WARMUP_ITERS = 500;
    constexpr int BENCH_ITERS = 10000;

    // Allocate aligned memory for vectors
    alignas(64) float x[D_MODEL];
    alignas(64) float logits[N_EXPERTS];
    alignas(64) float probs[N_EXPERTS];
    alignas(64) float sel_probs[N_EXPERTS];
    alignas(64) float bias[N_EXPERTS];
    alignas(64) block_q8_0 qx[D_MODEL / QK8_0];

    // Dummy eviction buffer: 64 MB (4x L3 cache)
    size_t dummy_size = 64 * 1024 * 1024;
    uint8_t* dummy_evict = (uint8_t*)_aligned_malloc(dummy_size, 64);
    std::memset(dummy_evict, 0x5A, dummy_size);
    volatile uint64_t sink = 0;

    // Initialize x and bias
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < D_MODEL; ++i) x[i] = dist(rng) * 0.05f;
    for (int i = 0; i < N_EXPERTS; ++i) bias[i] = dist(rng) * 0.1f;
    quantize_row_q8_0(x, qx, D_MODEL);

    // -------------------------------------------------------------
    // Part 1: Component Isolated Microbenchmarks (10,000 iters)
    // -------------------------------------------------------------
    std::cout << "[Part 1] Isolated Component Microbenchmarks (10,000 iterations each)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Allocate 1-layer router weights
    size_t fp32_1layer_bytes = N_EXPERTS * D_MODEL * sizeof(float); // 512 KB
    size_t fp16_1layer_bytes = N_EXPERTS * D_MODEL * sizeof(uint16_t); // 256 KB
    size_t q8_0_1layer_bytes = N_EXPERTS * (D_MODEL / QK8_0) * sizeof(block_q8_0); // 139,264 B

    float* w_fp32 = (float*)_aligned_malloc(fp32_1layer_bytes, 64);
    uint16_t* w_fp16 = (uint16_t*)_aligned_malloc(fp16_1layer_bytes, 64);
    block_q8_0* w_q8 = (block_q8_0*)_aligned_malloc(q8_0_1layer_bytes, 64);

    for (int i = 0; i < N_EXPERTS * D_MODEL; ++i) {
        float f = dist(rng) * 0.02f;
        w_fp32[i] = f;
        w_fp16[i] = float_to_fp16(f);
    }
    for (int r = 0; r < N_EXPERTS; ++r) {
        quantize_row_q8_0(w_fp32 + r * D_MODEL, w_q8 + r * (D_MODEL / QK8_0), D_MODEL);
    }

    Timer timer;
    std::vector<double> times;
    std::vector<uint64_t> cycles;
    times.reserve(BENCH_ITERS);
    cycles.reserve(BENCH_ITERS);

    // 1a. Sigmoid AVX2 vs Scalar
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        sigmoid_64_avx2(logits, probs);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats::compute(times, cycles).print("Sigmoid AVX2 (64 elem)");

    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        sigmoid_64_scalar(logits, probs);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats::compute(times, cycles).print("Sigmoid Scalar (64 elem)");

    // 1b. Top-4 Selection
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        Top4Result r = select_top4_heap(sel_probs, probs);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
        sink += r.idx[0];
    }
    BenchmarkStats::compute(times, cycles).print("Top-4 MinHeap (64 elem)");

    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        Top4Result r = select_top4_std(sel_probs, probs);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
        sink += r.idx[0];
    }
    BenchmarkStats::compute(times, cycles).print("Top-4 std::partial_sort");

    // 1c. Single-Layer Router GEMV (Warm L2 cache residency)
    std::cout << "\n[Part 2] Single-Layer Router Gate GEMV (64 x 2048) in Warm L2 Cache (10,000 iters)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Warm L2 FP32 GEMV
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        gemv_fp32(w_fp32, x, logits);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats::compute(times, cycles).print("FP32 GEMV (Warm L2)", (double)fp32_1layer_bytes);

    // Warm L2 FP16 GEMV
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        gemv_fp16(w_fp16, x, logits);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats::compute(times, cycles).print("FP16 GEMV (Warm L2)", (double)fp16_1layer_bytes);

    // Warm L2 Q8_0 GEMV
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        gemv_q8_0_q8_0(w_q8, qx, logits);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats::compute(times, cycles).print("Q8_0 GEMV (Warm L2)", (double)q8_0_1layer_bytes);

    // -------------------------------------------------------------
    // Part 3: 38-Layer Token Routing Across Cache Tiers (10,000 iters)
    // -------------------------------------------------------------
    std::cout << "\n[Part 3] Full 38-Layer Token Routing: Cold DDR4 vs Warm L3 vs Warm L2 (10,000 iters)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Allocate all 38 layers for FP32, FP16, and Q8_0
    size_t fp32_38layers_bytes = N_MOE_LAYERS * fp32_1layer_bytes; // 19.46 MB
    size_t fp16_38layers_bytes = N_MOE_LAYERS * fp16_1layer_bytes; // 9.73 MB
    size_t q8_0_38layers_bytes = N_MOE_LAYERS * q8_0_1layer_bytes; // 5.29 MB

    float* all_w_fp32 = (float*)_aligned_malloc(fp32_38layers_bytes, 64);
    uint16_t* all_w_fp16 = (uint16_t*)_aligned_malloc(fp16_38layers_bytes, 64);
    block_q8_0* all_w_q8 = (block_q8_0*)_aligned_malloc(q8_0_38layers_bytes, 64);

    for (size_t i = 0; i < N_MOE_LAYERS * N_EXPERTS * D_MODEL; ++i) {
        float f = dist(rng) * 0.02f;
        all_w_fp32[i] = f;
        all_w_fp16[i] = float_to_fp16(f);
    }
    for (int l = 0; l < N_MOE_LAYERS; ++l) {
        for (int r = 0; r < N_EXPERTS; ++r) {
            quantize_row_q8_0(all_w_fp32 + (l * N_EXPERTS + r) * D_MODEL,
                              all_w_q8 + (l * N_EXPERTS + r) * (D_MODEL / QK8_0),
                              D_MODEL);
        }
    }

    // Benchmark 3a: Warm L3 Residency (Token Pass across all 38 layers)
    // In each iteration, execute 38 layers sequentially: GEMV + Sigmoid + Bias + Top4
    // FP16: 9.7 MB fits in 16 MB L3.
    // Q8_0: 5.3 MB fits in 16 MB L3.
    // FP32: 19.5 MB spills slightly from 16 MB L3.

    // Warm L3 FP16 (38 layers)
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        for (int l = 0; l < N_MOE_LAYERS; ++l) {
            const uint16_t* layer_w = all_w_fp16 + l * (N_EXPERTS * D_MODEL);
            gemv_fp16(layer_w, x, logits);
            sigmoid_64_avx2(logits, probs);
            add_bias_64_avx2(probs, bias, sel_probs);
            Top4Result res = select_top4_heap(sel_probs, probs);
            sink += res.idx[0];
        }
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats fp16_l3_stats = BenchmarkStats::compute(times, cycles);
    fp16_l3_stats.print("FP16 38-Layer Token (Warm L3)", (double)fp16_38layers_bytes);

    // Warm L3 Q8_0 (38 layers)
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        for (int l = 0; l < N_MOE_LAYERS; ++l) {
            const block_q8_0* layer_w = all_w_q8 + l * (N_EXPERTS * (D_MODEL / QK8_0));
            gemv_q8_0_q8_0(layer_w, qx, logits);
            sigmoid_64_avx2(logits, probs);
            add_bias_64_avx2(probs, bias, sel_probs);
            Top4Result res = select_top4_heap(sel_probs, probs);
            sink += res.idx[0];
        }
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats q8_l3_stats = BenchmarkStats::compute(times, cycles);
    q8_l3_stats.print("Q8_0 38-Layer Token (Warm L3)", (double)q8_0_38layers_bytes);

    // FP32 38 layers (spills L3)
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        for (int l = 0; l < N_MOE_LAYERS; ++l) {
            const float* layer_w = all_w_fp32 + l * (N_EXPERTS * D_MODEL);
            gemv_fp32(layer_w, x, logits);
            sigmoid_64_avx2(logits, probs);
            add_bias_64_avx2(probs, bias, sel_probs);
            Top4Result res = select_top4_heap(sel_probs, probs);
            sink += res.idx[0];
        }
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats fp32_38_stats = BenchmarkStats::compute(times, cycles);
    fp32_38_stats.print("FP32 38-Layer Token (19.5 MB)", (double)fp32_38layers_bytes);

    // Benchmark 3b: Cold DDR4 RAM Streaming
    // Flush L3 cache before token pass (or single layer) to force DRAM read
    constexpr int COLD_ITERS = 2000; // Cold test with L3 flush
    times.clear(); cycles.clear();
    for (int i = 0; i < COLD_ITERS; ++i) {
        flush_l3_cache(dummy_evict, dummy_size, &sink);
        uint64_t c0 = rdtsc_fence();
        timer.start();
        for (int l = 0; l < N_MOE_LAYERS; ++l) {
            const uint16_t* layer_w = all_w_fp16 + l * (N_EXPERTS * D_MODEL);
            gemv_fp16(layer_w, x, logits);
            sigmoid_64_avx2(logits, probs);
            add_bias_64_avx2(probs, bias, sel_probs);
            Top4Result res = select_top4_heap(sel_probs, probs);
            sink += res.idx[0];
        }
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats fp16_cold_stats = BenchmarkStats::compute(times, cycles);
    fp16_cold_stats.print("FP16 38-Layer Token (Cold DDR4)", (double)fp16_38layers_bytes);

    // Cold Single Layer FP16 vs Warm Single Layer FP16
    times.clear(); cycles.clear();
    for (int i = 0; i < COLD_ITERS; ++i) {
        flush_l3_cache(dummy_evict, dummy_size, &sink);
        uint64_t c0 = rdtsc_fence();
        timer.start();
        gemv_fp16(w_fp16, x, logits);
        sigmoid_64_avx2(logits, probs);
        add_bias_64_avx2(probs, bias, sel_probs);
        Top4Result res = select_top4_heap(sel_probs, probs);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
        sink += res.idx[0];
    }
    BenchmarkStats::compute(times, cycles).print("FP16 1-Layer (Cold DDR4)", (double)fp16_1layer_bytes);

    // Warm Single Layer FP16 (in L2)
    times.clear(); cycles.clear();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        uint64_t c0 = rdtsc_fence();
        timer.start();
        gemv_fp16(w_fp16, x, logits);
        sigmoid_64_avx2(logits, probs);
        add_bias_64_avx2(probs, bias, sel_probs);
        Top4Result res = select_top4_heap(sel_probs, probs);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
        sink += res.idx[0];
    }
    BenchmarkStats::compute(times, cycles).print("FP16 1-Layer (Warm L2)", (double)fp16_1layer_bytes);

    // -------------------------------------------------------------
    // Part 4: Expert Dispatch & Memory Hopping Penalty (Task 4)
    // -------------------------------------------------------------
    std::cout << "\n[Part 4] Expert Dispatch & Memory Hopping Penalty (DDR4-3200 Empirical)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Allocate 1 full layer pool of 64 experts: 215.5 MB
    uint8_t* layer_expert_pool = (uint8_t*)_aligned_malloc(LAYER_ALL_EXPERTS_BYTES, 64);
    std::memset(layer_expert_pool, 0x3C, LAYER_ALL_EXPERTS_BYTES);

    std::cout << "Layer Expert Buffer: 64 experts x " << (ONE_EXPERT_BYTES / 1024.0 / 1024.0) << " MB = "
              << (LAYER_ALL_EXPERTS_BYTES / 1024.0 / 1024.0) << " MB (Evicts L3 entirely on access)\n";
    std::cout << "Streaming 4 active experts: 4 x " << (ONE_EXPERT_BYTES / 1024.0 / 1024.0) << " MB = "
              << (ACTIVE_4_EXPERTS_BYTES / 1024.0 / 1024.0) << " MB per layer\n\n";

    // Subtest 4a: Contiguous Streaming of 4 Experts (13.47 MB contiguous block)
    constexpr int HOP_ITERS = 1000;
    times.clear(); cycles.clear();
    for (int i = 0; i < HOP_ITERS; ++i) {
        // Choose contiguous offset (expert k, k+1, k+2, k+3)
        size_t start_expert = (i % 61);
        const uint64_t* ptr = (const uint64_t*)(layer_expert_pool + start_expert * ONE_EXPERT_BYTES);
        size_t num_u64 = ACTIVE_4_EXPERTS_BYTES / sizeof(uint64_t);

        uint64_t local_sum = 0;
        uint64_t c0 = rdtsc_fence();
        timer.start();
        // AVX2 streaming load loop
        const __m256i* vptr = (const __m256i*)ptr;
        size_t num_v256 = ACTIVE_4_EXPERTS_BYTES / sizeof(__m256i);
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        for (size_t v = 0; v < num_v256; v += 2) {
            acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(vptr + v));
            acc1 = _mm256_add_epi64(acc1, _mm256_loadu_si256(vptr + v + 1));
        }
        sink += _mm256_extract_epi64(acc0, 0) + _mm256_extract_epi64(acc1, 0);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats contig_stats = BenchmarkStats::compute(times, cycles);
    contig_stats.print("Contiguous 4 Experts (13.47 MB)", (double)ACTIVE_4_EXPERTS_BYTES);

    // Subtest 4b: Scattered 4 Experts (Random 4 indices out of 64)
    // Simulates dynamic token-level routing where 4 arbitrary experts are selected
    times.clear(); cycles.clear();
    std::uniform_int_distribution<int> exp_dist(0, 63);
    for (int i = 0; i < HOP_ITERS; ++i) {
        int e[4];
        e[0] = exp_dist(rng);
        e[1] = exp_dist(rng);
        e[2] = exp_dist(rng);
        e[3] = exp_dist(rng);

        uint64_t c0 = rdtsc_fence();
        timer.start();
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        for (int k = 0; k < 4; ++k) {
            const __m256i* vptr = (const __m256i*)(layer_expert_pool + e[k] * ONE_EXPERT_BYTES);
            size_t num_v256 = ONE_EXPERT_BYTES / sizeof(__m256i);
            for (size_t v = 0; v < num_v256; v += 2) {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(vptr + v));
                acc1 = _mm256_add_epi64(acc1, _mm256_loadu_si256(vptr + v + 1));
            }
        }
        sink += _mm256_extract_epi64(acc0, 0) + _mm256_extract_epi64(acc1, 0);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats scattered_stats = BenchmarkStats::compute(times, cycles);
    scattered_stats.print("Scattered 4 Experts (Random 4/64)", (double)ACTIVE_4_EXPERTS_BYTES);

    // Subtest 4c: Projection Hopping (Gate -> Up -> Down interleaved)
    // As executed in real MoE runtime: Gate for 4 exps, Up for 4 exps, Down for 4 exps
    // In GGUF, Gate, Up, Down are 3 distinct tensors in memory!
    size_t gate_pool_bytes = 64 * GATE_EXP_BYTES; // 64.49 MB
    size_t up_pool_bytes   = 64 * UP_EXP_BYTES;   // 64.49 MB
    size_t down_pool_bytes = 64 * DOWN_EXP_BYTES; // 86.51 MB
    uint8_t* gate_pool = layer_expert_pool;
    uint8_t* up_pool = (uint8_t*)_aligned_malloc(up_pool_bytes, 64);
    uint8_t* down_pool = (uint8_t*)_aligned_malloc(down_pool_bytes, 64);
    std::memset(up_pool, 0x4B, up_pool_bytes);
    std::memset(down_pool, 0x5C, down_pool_bytes);

    times.clear(); cycles.clear();
    for (int i = 0; i < HOP_ITERS; ++i) {
        int e[4];
        e[0] = exp_dist(rng);
        e[1] = exp_dist(rng);
        e[2] = exp_dist(rng);
        e[3] = exp_dist(rng);

        uint64_t c0 = rdtsc_fence();
        timer.start();
        __m256i acc0 = _mm256_setzero_si256();

        // 1. Stream Gate projections for 4 experts
        for (int k = 0; k < 4; ++k) {
            const __m256i* vptr = (const __m256i*)(gate_pool + e[k] * GATE_EXP_BYTES);
            size_t num_v256 = GATE_EXP_BYTES / sizeof(__m256i);
            for (size_t v = 0; v < num_v256; ++v) {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(vptr + v));
            }
        }
        // 2. Stream Up projections for 4 experts
        for (int k = 0; k < 4; ++k) {
            const __m256i* vptr = (const __m256i*)(up_pool + e[k] * UP_EXP_BYTES);
            size_t num_v256 = UP_EXP_BYTES / sizeof(__m256i);
            for (size_t v = 0; v < num_v256; ++v) {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(vptr + v));
            }
        }
        // 3. Stream Down projections for 4 experts
        for (int k = 0; k < 4; ++k) {
            const __m256i* vptr = (const __m256i*)(down_pool + e[k] * DOWN_EXP_BYTES);
            size_t num_v256 = DOWN_EXP_BYTES / sizeof(__m256i);
            for (size_t v = 0; v < num_v256; ++v) {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(vptr + v));
            }
        }
        sink += _mm256_extract_epi64(acc0, 0);
        double t = timer.stop_us();
        uint64_t c1 = rdtsc_fence();
        times.push_back(t);
        cycles.push_back(c1 - c0);
    }
    BenchmarkStats proj_hop_stats = BenchmarkStats::compute(times, cycles);
    proj_hop_stats.print("Proj-Hopping 4 Exps (Gate/Up/Down)", (double)ACTIVE_4_EXPERTS_BYTES);

    // Calculate bandwidth penalty
    double contig_bw = (ACTIVE_4_EXPERTS_BYTES / (contig_stats.mean_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    double scattered_bw = (ACTIVE_4_EXPERTS_BYTES / (scattered_stats.mean_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    double proj_hop_bw = (ACTIVE_4_EXPERTS_BYTES / (proj_hop_stats.mean_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    double penalty_pct = (1.0 - scattered_bw / contig_bw) * 100.0;
    double proj_penalty_pct = (1.0 - proj_hop_bw / contig_bw) * 100.0;

    std::cout << "\n>>> Hopping Penalty Summary:\n";
    std::cout << "    Contiguous 4-Expert BW: " << std::fixed << std::setprecision(2) << contig_bw << " GB/s\n";
    std::cout << "    Scattered 4-Expert BW:  " << scattered_bw << " GB/s (Penalty: -" << penalty_pct << "%)\n";
    std::cout << "    Projection-Hopping BW:  " << proj_hop_bw << " GB/s (Penalty: -" << proj_penalty_pct << "%)\n";

    // Cleanup
    _aligned_free(w_fp32);
    _aligned_free(w_fp16);
    _aligned_free(w_q8);
    _aligned_free(all_w_fp32);
    _aligned_free(all_w_fp16);
    _aligned_free(all_w_q8);
    _aligned_free(dummy_evict);
    _aligned_free(layer_expert_pool);
    _aligned_free(up_pool);
    _aligned_free(down_pool);

    std::cout << "\n================================================================================\n";
    std::cout << "  Benchmark Complete (sink=" << sink << ")\n";
    std::cout << "================================================================================\n";

    return 0;
}
