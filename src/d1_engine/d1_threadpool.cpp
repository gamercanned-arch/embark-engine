#include "d1_threadpool.h"
#include <immintrin.h>
#include <iostream>

namespace d1 {

D1ThreadPool::D1ThreadPool() {
    for (int i = 0; i < NUM_D1_WORKERS; ++i) {
        m_worker_sinks[i] = 0;
        m_worker_done[i].store(0, std::memory_order_relaxed);
    }

    std::atomic<int> ready_count{0};

    for (int i = 0; i < NUM_D1_WORKERS; ++i) {
        m_workers.emplace_back([this, i, &ready_count]() {
            // Pin strictly to physical cores 0, 2, 4, 6, 8, 10, 12, 14
            // (Avoiding SMT hyperthreads 1, 3, 5, 7, 9, 11, 13, 15 to eliminate vector ALU contention)
            int logical_cpu = i * 2;
            DWORD_PTR affinity_mask = (DWORD_PTR)(1ULL << logical_cpu);

            SetThreadAffinityMask(GetCurrentThread(), affinity_mask);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

            ready_count.fetch_add(1, std::memory_order_release);
            worker_main(i);
        });
    }

    // Spin-wait until all 8 workers have completed thread affinity and priority configuration
    while (ready_count.load(std::memory_order_acquire) < NUM_D1_WORKERS) {
        std::this_thread::yield();
    }
}

D1ThreadPool::~D1ThreadPool() {
    m_terminate.store(true, std::memory_order_release);
    m_generation.fetch_add(1, std::memory_order_release);

    for (auto& w : m_workers) {
        if (w.joinable()) {
            w.join();
        }
    }
}

void D1ThreadPool::worker_main(int worker_id) {
    int last_gen = 0;

    while (!m_terminate.load(std::memory_order_relaxed)) {
        // Spin-wait for new task generation (< 200 ns barrier latency)
        while (m_generation.load(std::memory_order_acquire) == last_gen) {
            if (m_terminate.load(std::memory_order_relaxed)) return;
            _mm_pause();
        }

        last_gen = m_generation.load(std::memory_order_relaxed);
        if (m_terminate.load(std::memory_order_relaxed)) return;

        if (m_active_task) {
            m_active_task(worker_id);
        }

        m_completed.fetch_add(1, std::memory_order_release);
    }
}

void D1ThreadPool::dispatch(const std::function<void(int worker_id)>& task_func) {
    m_active_task = task_func;
    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    // Spin-wait barrier until all 8 physical cores complete work
    while (m_completed.load(std::memory_order_acquire) < NUM_D1_WORKERS) {
        _mm_pause();
    }
}

void D1ThreadPool::stream_contiguous_block(const void* ptr, size_t total_bytes) {
    const uint8_t* byte_ptr = (const uint8_t*)ptr;
    size_t chunk_per_core = (total_bytes / NUM_D1_WORKERS + 63ULL) & ~63ULL;

    dispatch([this, byte_ptr, total_bytes, chunk_per_core](int tid) {
        size_t start = tid * chunk_per_core;
        size_t end = (tid == NUM_D1_WORKERS - 1) ? total_bytes : (std::min)(start + chunk_per_core, total_bytes);

        if (start < total_bytes) {
            const __m256i* p = (const __m256i*)(byte_ptr + start);
            size_t n_vec = (end - start) / sizeof(__m256i);

            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            size_t v = 0;
            // Saturated non-temporal streaming reads using MOVNTDQA to bypass L3 pollution
            for (; v + 4 <= n_vec; v += 4) {
                acc0 = _mm256_add_epi64(acc0, _mm256_stream_load_si256(p + v + 0));
                acc1 = _mm256_add_epi64(acc1, _mm256_stream_load_si256(p + v + 1));
                acc2 = _mm256_add_epi64(acc2, _mm256_stream_load_si256(p + v + 2));
                acc3 = _mm256_add_epi64(acc3, _mm256_stream_load_si256(p + v + 3));
            }

            for (; v < n_vec; ++v) {
                acc0 = _mm256_add_epi64(acc0, _mm256_stream_load_si256(p + v));
            }

            __m256i sum_all = _mm256_add_epi64(_mm256_add_epi64(acc0, acc1), _mm256_add_epi64(acc2, acc3));
            __m128i lo = _mm256_castsi256_si128(sum_all);
            __m128i hi = _mm256_extracti128_si256(sum_all, 1);
            __m128i res = _mm_add_epi64(lo, hi);
            uint64_t val = (uint64_t)_mm_cvtsi128_si64(res) + (uint64_t)_mm_extract_epi64(res, 1);

            m_worker_sinks[tid] += val;
        }
    });
}

uint64_t D1ThreadPool::get_total_sink() const {
    uint64_t total = 0;
    for (int i = 0; i < NUM_D1_WORKERS; ++i) {
        total += m_worker_sinks[i];
    }
    return total;
}

} // namespace d1
