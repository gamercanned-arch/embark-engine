#include <windows.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include <cstring>

struct Timer {
    LARGE_INTEGER freq, start_time;
    Timer() { QueryPerformanceFrequency(&freq); }
    void start() { QueryPerformanceCounter(&start_time); }
    double stop_ns() const {
        LARGE_INTEGER end_time;
        QueryPerformanceCounter(&end_time);
        return (double)(end_time.QuadPart - start_time.QuadPart) * 1e9 / freq.QuadPart;
    }
};

void test_large_pages() {
    std::cout << "[1] Checking Windows Large Page (2MB HugePages) Support:\n";
    SIZE_T min_lp = GetLargePageMinimum();
    std::cout << "    GetLargePageMinimum: " << min_lp << " bytes (" << (min_lp / (1024 * 1024)) << " MB)\n";
    
    // Try VirtualAlloc with MEM_LARGE_PAGES
    void* lp_buf = VirtualAlloc(NULL, min_lp, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (lp_buf) {
        std::cout << "    >>> SUCCESS: SeLockMemoryPrivilege is active! MEM_LARGE_PAGES allocated directly at: " << lp_buf << "\n";
        VirtualFree(lp_buf, 0, MEM_RELEASE);
    } else {
        DWORD err = GetLastError();
        std::cout << "    >>> NOTE: MEM_LARGE_PAGES requires SeLockMemoryPrivilege (Error " << err << ").\n";
        std::cout << "    >>> Using 64KB-aligned VirtualAlloc with pre-faulted OS committed pages as fallback.\n";
    }
}

void test_cache_hierarchy_latency() {
    std::cout << "\n[2] Memory Hierarchy Latency & Bandwidth Profile (Pointer Chasing & Streaming):\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    
    struct Level {
        const char* name;
        size_t size;
        int iters;
    };
    
    Level levels[] = {
        {"L1 Data Cache (16 KB)  ", 16 * 1024, 200000},
        {"L2 Cache (256 KB)      ", 256 * 1024, 100000},
        {"L3 Cache (8 MB)        ", 8 * 1024 * 1024, 5000},
        {"DDR4 DRAM (64 MB)      ", 64 * 1024 * 1024, 500},
        {"DDR4 MoE Pool (256 MB) ", 256 * 1024 * 1024, 100}
    };
    
    Timer timer;
    volatile uint64_t sink = 0;
    
    for (const auto& lvl : levels) {
        uint8_t* buf = (uint8_t*)_aligned_malloc(lvl.size, 64);
        std::memset(buf, 0x5A, lvl.size);
        
        // Sequential Streaming Bandwidth (AVX2 256-bit reads)
        timer.start();
        size_t n_vec = lvl.size / sizeof(__m256i);
        const __m256i* vptr = (const __m256i*)buf;
        __m256i acc = _mm256_setzero_si256();
        for (int it = 0; it < lvl.iters; ++it) {
            for (size_t i = 0; i < n_vec; ++i) {
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256(vptr + i));
            }
        }
        sink += _mm256_extract_epi64(acc, 0);
        double seq_ns = timer.stop_ns();
        double seq_bw = ((double)lvl.size * lvl.iters / (seq_ns * 1e-9)) / (1024.0 * 1024.0 * 1024.0);
        
        // Stride Random Latency (Pointer Chasing with 64-byte hops)
        size_t n_nodes = lvl.size / 64;
        std::vector<size_t> p(n_nodes);
        std::iota(p.begin(), p.end(), 0);
        // Pseudo-random shuffle for pointer chasing
        for (size_t i = n_nodes - 1; i > 0; --i) {
            size_t j = (i * 1103515245 + 12345) % (i + 1);
            std::swap(p[i], p[j]);
        }
        uint64_t** nodes = (uint64_t**)buf;
        for (size_t i = 0; i < n_nodes; ++i) {
            nodes[p[i] * 8] = (uint64_t*)&nodes[p[(i + 1) % n_nodes] * 8];
        }
        
        uint64_t* curr = (uint64_t*)buf;
        int chase_steps = (lvl.size <= 256 * 1024) ? 500000 : 100000;
        timer.start();
        for (int s = 0; s < chase_steps; ++s) {
            curr = (uint64_t*)*curr;
        }
        sink += (uint64_t)curr;
        double chase_ns = timer.stop_ns() / chase_steps;
        
        std::cout << lvl.name 
                  << " | Streaming BW: " << std::setw(7) << std::fixed << std::setprecision(2) << seq_bw << " GB/s"
                  << " | Random Latency: " << std::setw(6) << std::fixed << std::setprecision(2) << chase_ns << " ns\n";
        
        _aligned_free(buf);
    }
}

void test_simd_throughput() {
    std::cout << "\n[3] Zen 3 SIMD Micro-Kernel Raw Instruction Throughput (1 Core):\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    
    constexpr int ITERS = 10000000; // 10M loops
    Timer timer;
    volatile uint64_t sink = 0;
    
    // 1. FP32 FMA Dual-Issue (Port 0 & Port 1)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 v = _mm256_set1_ps(1.0001f);
        timer.start();
        for (int i = 0; i < ITERS; ++i) {
            acc0 = _mm256_fmadd_ps(acc0, v, v);
            acc1 = _mm256_fmadd_ps(acc1, v, v);
            acc2 = _mm256_fmadd_ps(acc2, v, v);
            acc3 = _mm256_fmadd_ps(acc3, v, v);
        }
        double ns = timer.stop_ns();
        double ops = (double)ITERS * 4 * 16; // 4 FMAs x 16 FLOPs
        double gflops = ops / ns;
        std::cout << "FP32 Dual-Issue FMA (Port 0 & 1)  : " << std::setw(7) << std::fixed << std::setprecision(2) << gflops << " GFLOPs/s (" << (ns / (ITERS * 4)) << " ns / FMA)\n";
        sink += (uint64_t)_mm256_cvtss_f32(acc0);
    }
    
    // 2. INT8 Integer Dot Product (vpmaddubsw + vpmaddwd)
    {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i v_u8 = _mm256_set1_epi8(2);
        __m256i v_s8 = _mm256_set1_epi8(3);
        __m256i one  = _mm256_set1_epi16(1);
        timer.start();
        for (int i = 0; i < ITERS; ++i) {
            __m256i p0 = _mm256_maddubs_epi16(v_u8, v_s8);
            __m256i p1 = _mm256_maddubs_epi16(v_u8, v_s8);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, one));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, one));
        }
        double ns = timer.stop_ns();
        double macs = (double)ITERS * 2 * 32; // 2 instructions x 32 INT8 MACs
        double gmacs = macs / ns;
        std::cout << "INT8 Vector Dot Product (vpmadd)  : " << std::setw(7) << std::fixed << std::setprecision(2) << gmacs << " GMACs/s  (" << (ns / (ITERS * 2)) << " ns / pair)\n";
        sink += _mm256_extract_epi64(acc0, 0);
    }
    
    // 3. VPSHUFB Lookup Table (Port 1 & 5 Vector Shuffles)
    {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i table = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m256i idx0  = _mm256_set1_epi8(0x07);
        __m256i idx1  = _mm256_set1_epi8(0x03);
        timer.start();
        for (int i = 0; i < ITERS; ++i) {
            acc0 = _mm256_shuffle_epi8(table, idx0);
            acc1 = _mm256_shuffle_epi8(table, idx1);
        }
        double ns = timer.stop_ns();
        double lookups = (double)ITERS * 2 * 32; // 2 shuffles x 32 lookups
        double glookups = lookups / ns;
        std::cout << "VPSHUFB Parallel Lookups (2 Shuf) : " << std::setw(7) << std::fixed << std::setprecision(2) << glookups << " GLookups/s (" << (ns / (ITERS * 2)) << " ns / shuf)\n";
        sink += _mm256_extract_epi64(acc0, 0);
    }

    // 4. F16C Hardware Decompress (_mm256_cvtph_ps)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m128i h0 = _mm_set1_epi16(0x3C00); // 1.0f in fp16
        __m128i h1 = _mm_set1_epi16(0x4000); // 2.0f in fp16
        timer.start();
        for (int i = 0; i < ITERS; ++i) {
            acc0 = _mm256_add_ps(acc0, _mm256_cvtph_ps(h0));
            acc1 = _mm256_add_ps(acc1, _mm256_cvtph_ps(h1));
        }
        double ns = timer.stop_ns();
        double f16c_ops = (double)ITERS * 2 * 8; // 2 convs x 8 fp16->fp32
        double g_f16c = f16c_ops / ns;
        std::cout << "F16C Hardware Decompress (F16->F32): " << std::setw(7) << std::fixed << std::setprecision(2) << g_f16c << " GDecodes/s (" << (ns / (ITERS * 2)) << " ns / conv)\n";
        sink += (uint64_t)_mm256_cvtss_f32(acc0);
    }
}

int main() {
    test_large_pages();
    test_cache_hierarchy_latency();
    test_simd_throughput();
    return 0;
}
