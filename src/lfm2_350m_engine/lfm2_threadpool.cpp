#include "lfm2_threadpool.h"
#include <immintrin.h>
#include <iostream>
#include <algorithm>

namespace lfm2 {

LFM2ThreadPool::LFM2ThreadPool() {
    for (int i = 0; i < NUM_LFM2_WORKERS; ++i) {
        m_worker_sinks[i] = 0;
        m_worker_done[i].store(0, std::memory_order_relaxed);
    }

    std::atomic<int> ready_count{0};

    for (int i = 0; i < NUM_LFM2_WORKERS; ++i) {
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
    while (ready_count.load(std::memory_order_acquire) < NUM_LFM2_WORKERS) {
        std::this_thread::yield();
    }
}

LFM2ThreadPool::~LFM2ThreadPool() {
    m_terminate.store(true, std::memory_order_release);
    m_generation.fetch_add(1, std::memory_order_release);

    for (auto& w : m_workers) {
        if (w.joinable()) {
            w.join();
        }
    }
}

void LFM2ThreadPool::worker_main(int worker_id) {
    int last_gen = 0;

    while (!m_terminate.load(std::memory_order_relaxed)) {
        // Spin-wait for new task generation (< 150 ns barrier latency)
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

void LFM2ThreadPool::dispatch(const std::function<void(int worker_id)>& task_func) {
    m_active_task = task_func;
    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    // Spin-wait barrier until all 8 physical cores complete work
    while (m_completed.load(std::memory_order_acquire) < NUM_LFM2_WORKERS) {
        _mm_pause();
    }
}

void LFM2ThreadPool::stream_contiguous_block(const void* ptr, size_t total_bytes) {
    const uint8_t* byte_ptr = (const uint8_t*)ptr;
    size_t chunk_per_core = (total_bytes / NUM_LFM2_WORKERS + 63ULL) & ~63ULL;

    dispatch([this, byte_ptr, total_bytes, chunk_per_core](int tid) {
        size_t start = tid * chunk_per_core;
        size_t end = (tid == NUM_LFM2_WORKERS - 1) ? total_bytes : (std::min)(start + chunk_per_core, total_bytes);

        if (start < total_bytes) {
            const __m256i* p = (const __m256i*)(byte_ptr + start);
            size_t n_vec = (end - start) / sizeof(__m256i);

            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256();
            __m256i acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256();
            __m256i acc7 = _mm256_setzero_si256();

            size_t v = 0;
            // 8-way unrolled non-temporal streaming reads (256 bytes / 4 cache lines per iteration)
            for (; v + 8 <= n_vec; v += 8) {
                _mm_prefetch((const char*)(p + v + 16), _MM_HINT_NTA);
                _mm_prefetch((const char*)(p + v + 20), _MM_HINT_NTA);

                acc0 = _mm256_add_epi64(acc0, _mm256_stream_load_si256(p + v + 0));
                acc1 = _mm256_add_epi64(acc1, _mm256_stream_load_si256(p + v + 1));
                acc2 = _mm256_add_epi64(acc2, _mm256_stream_load_si256(p + v + 2));
                acc3 = _mm256_add_epi64(acc3, _mm256_stream_load_si256(p + v + 3));
                acc4 = _mm256_add_epi64(acc4, _mm256_stream_load_si256(p + v + 4));
                acc5 = _mm256_add_epi64(acc5, _mm256_stream_load_si256(p + v + 5));
                acc6 = _mm256_add_epi64(acc6, _mm256_stream_load_si256(p + v + 6));
                acc7 = _mm256_add_epi64(acc7, _mm256_stream_load_si256(p + v + 7));
            }

            for (; v < n_vec; ++v) {
                acc0 = _mm256_add_epi64(acc0, _mm256_stream_load_si256(p + v));
            }

            __m256i tot0 = _mm256_add_epi64(_mm256_add_epi64(acc0, acc1), _mm256_add_epi64(acc2, acc3));
            __m256i tot1 = _mm256_add_epi64(_mm256_add_epi64(acc4, acc5), _mm256_add_epi64(acc6, acc7));
            __m256i tot = _mm256_add_epi64(tot0, tot1);

            uint64_t val = (uint64_t)_mm256_extract_epi64(tot, 0) + (uint64_t)_mm256_extract_epi64(tot, 1)
                         + (uint64_t)_mm256_extract_epi64(tot, 2) + (uint64_t)_mm256_extract_epi64(tot, 3);
            m_worker_sinks[tid] += val;
        }
    });
}

uint64_t LFM2ThreadPool::get_total_sink() const {
    uint64_t sum = 0;
    for (int i = 0; i < NUM_LFM2_WORKERS; ++i) {
        sum += m_worker_sinks[i];
    }
    return sum;
}

} // namespace lfm2
