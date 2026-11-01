#include <windows.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cstring>
#include <iomanip>
#include <random>

constexpr size_t GATE_EXP_BYTES = 1007616;
constexpr size_t UP_EXP_BYTES   = 1007616;
constexpr size_t DOWN_EXP_BYTES = 1351680;
constexpr size_t ONE_EXPERT_BYTES = GATE_EXP_BYTES + UP_EXP_BYTES + DOWN_EXP_BYTES; // 3,366,912
constexpr size_t LAYER_ALL_EXPERTS_BYTES = 64 * ONE_EXPERT_BYTES; // ~205.5 MB
constexpr size_t ACTIVE_4_EXPERTS_BYTES = 4 * ONE_EXPERT_BYTES;   // ~12.84 MB

struct Timer {
    LARGE_INTEGER freq, start_time;
    Timer() { QueryPerformanceFrequency(&freq); }
    void start() { QueryPerformanceCounter(&start_time); }
    double stop_us() const {
        LARGE_INTEGER end_time;
        QueryPerformanceCounter(&end_time);
        return (double)(end_time.QuadPart - start_time.QuadPart) * 1e6 / freq.QuadPart;
    }
};

int main() {
    std::cout << "Testing Multi-Threaded Expert Streaming (OpenMP 8 Threads) on DDR4-3200...\n";
    uint8_t* pool = (uint8_t*)_aligned_malloc(LAYER_ALL_EXPERTS_BYTES, 64);
    std::memset(pool, 0xAB, LAYER_ALL_EXPERTS_BYTES);
    volatile uint64_t sink = 0;
    Timer timer;

    constexpr int ITERS = 1000;

    // 1. Single-Thread Contiguous
    double t_contig_1t = 0;
    for (int i = 0; i < ITERS; ++i) {
        size_t offset = (i % 61) * ONE_EXPERT_BYTES;
        const __m256i* ptr = (const __m256i*)(pool + offset);
        size_t n_vec = ACTIVE_4_EXPERTS_BYTES / sizeof(__m256i);
        timer.start();
        __m256i acc0 = _mm256_setzero_si256();
        for (size_t v = 0; v < n_vec; ++v) {
            acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(ptr + v));
        }
        sink += _mm256_extract_epi64(acc0, 0);
        t_contig_1t += timer.stop_us();
    }
    t_contig_1t /= ITERS;
    double bw_contig_1t = (ACTIVE_4_EXPERTS_BYTES / (t_contig_1t * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "1 Thread  Contiguous 4-Expert BW: " << std::fixed << std::setprecision(2) << bw_contig_1t << " GB/s (" << t_contig_1t << " us)\n";

    // 2. 8 Threads Contiguous (Each thread streams chunk of the 4 active experts)
    double t_contig_8t = 0;
    for (int i = 0; i < ITERS; ++i) {
        size_t offset = (i % 61) * ONE_EXPERT_BYTES;
        const __m256i* ptr = (const __m256i*)(pool + offset);
        size_t n_vec = ACTIVE_4_EXPERTS_BYTES / sizeof(__m256i);
        timer.start();
        uint64_t thread_sink = 0;
        #pragma omp parallel num_threads(8) reduction(+:thread_sink)
        {
            int tid = omp_get_thread_num();
            size_t chunk = n_vec / 8;
            size_t start = tid * chunk;
            size_t end = (tid == 7) ? n_vec : start + chunk;
            __m256i acc = _mm256_setzero_si256();
            for (size_t v = start; v < end; ++v) {
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256(ptr + v));
            }
            thread_sink += _mm256_extract_epi64(acc, 0);
        }
        sink += thread_sink;
        t_contig_8t += timer.stop_us();
    }
    t_contig_8t /= ITERS;
    double bw_contig_8t = (ACTIVE_4_EXPERTS_BYTES / (t_contig_8t * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "8 Threads Contiguous 4-Expert BW: " << bw_contig_8t << " GB/s (" << t_contig_8t << " us)\n";

    // 3. 8 Threads Scattered (Each of the 4 active experts streamed in parallel across threads)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> exp_dist(0, 63);
    double t_scat_8t = 0;
    for (int i = 0; i < ITERS; ++i) {
        int e[4];
        for (int k = 0; k < 4; ++k) e[k] = exp_dist(rng);

        timer.start();
        uint64_t thread_sink = 0;
        #pragma omp parallel num_threads(8) reduction(+:thread_sink)
        {
            int tid = omp_get_thread_num();
            // 8 threads, 4 experts -> 2 threads per expert
            int exp_idx = tid / 2;
            int part = tid % 2;
            const __m256i* ptr = (const __m256i*)(pool + e[exp_idx] * ONE_EXPERT_BYTES);
            size_t n_vec = ONE_EXPERT_BYTES / sizeof(__m256i);
            size_t chunk = n_vec / 2;
            size_t start = part * chunk;
            size_t end = (part == 1) ? n_vec : start + chunk;
            __m256i acc = _mm256_setzero_si256();
            for (size_t v = start; v < end; ++v) {
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256(ptr + v));
            }
            thread_sink += _mm256_extract_epi64(acc, 0);
        }
        sink += thread_sink;
        t_scat_8t += timer.stop_us();
    }
    t_scat_8t /= ITERS;
    double bw_scat_8t = (ACTIVE_4_EXPERTS_BYTES / (t_scat_8t * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "8 Threads Scattered 4-Expert BW:  " << bw_scat_8t << " GB/s (" << t_scat_8t << " us)\n";
    std::cout << "Scattered vs Contiguous 8T penalty: " << (1.0 - bw_scat_8t / bw_contig_8t) * 100.0 << "%\n";

    _aligned_free(pool);
    return 0;
}
