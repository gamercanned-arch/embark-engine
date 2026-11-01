// flux2_threadpool.cpp
#include "flux2_threadpool.h"
#include <iostream>
#include <immintrin.h>

namespace flux2 {

Flux2ThreadPool::Flux2ThreadPool() {
    for (int i = 0; i < NUM_WORKERS; i++) {
        m_worker_sinks[i] = 0;
    }

    m_workers.reserve(NUM_WORKERS);

    for (int i = 0; i < NUM_WORKERS; i++) {
        m_workers.emplace_back([this, i]() {
            // Pin strictly to physical cores (0, 2, 4, 6, 8, 10, 12, 14)
            DWORD_PTR mask = (DWORD_PTR)(1ULL << (i * 2));
            HANDLE hThread = GetCurrentThread();
            SetThreadAffinityMask(hThread, mask);
            SetThreadPriority(hThread, THREAD_PRIORITY_HIGHEST);

            int local_gen = 0;

            while (!m_terminate.load(std::memory_order_relaxed)) {
                // Low-power spin wait
                while (m_generation.load(std::memory_order_acquire) == local_gen) {
                    if (m_terminate.load(std::memory_order_relaxed)) return;
                    _mm_pause();
                }

                local_gen = m_generation.load(std::memory_order_relaxed);

                int mode = m_mode.load(std::memory_order_relaxed);
                if (mode == 0) {
                    // Custom parallel task
                    if (m_custom_task) {
                        m_custom_task(i);
                    }
                } else if (mode == 1) {
                    // Contiguous memory stream / touch
                    if (m_stream_ptr && m_stream_bytes > 0) {
                        size_t chunk_per_core = m_stream_bytes / NUM_WORKERS;
                        size_t start_byte = i * chunk_per_core;
                        size_t len = (i == NUM_WORKERS - 1) ? (m_stream_bytes - start_byte) : chunk_per_core;

                        const uint8_t* p = (const uint8_t*)m_stream_ptr + start_byte;
                        uint64_t sum = 0;
                        size_t n_vec = len / 32;
                        const __m256i* vp = (const __m256i*)p;
                        __m256i acc0 = _mm256_setzero_si256();
                        __m256i acc1 = _mm256_setzero_si256();

                        for (size_t v = 0; v + 1 < n_vec; v += 2) {
                            acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(vp + v + 0));
                            acc1 = _mm256_add_epi64(acc1, _mm256_loadu_si256(vp + v + 1));
                        }
                        __m256i total = _mm256_add_epi64(acc0, acc1);
                        sum = (uint64_t)_mm256_extract_epi64(total, 0) + (uint64_t)_mm256_extract_epi64(total, 1);
                        m_worker_sinks[i] += sum;
                    }
                }

                m_completed.fetch_add(1, std::memory_order_release);
            }
        });
    }
}

Flux2ThreadPool::~Flux2ThreadPool() {
    m_terminate.store(true, std::memory_order_release);
    m_generation.fetch_add(1, std::memory_order_release);
    for (auto& w : m_workers) {
        if (w.joinable()) w.join();
    }
}

void Flux2ThreadPool::dispatch(const std::function<void(int worker_id)>& task_func) {
    m_custom_task = task_func;
    m_mode.store(0, std::memory_order_relaxed);
    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    while (m_completed.load(std::memory_order_acquire) < NUM_WORKERS) {
        _mm_pause();
    }
}

void Flux2ThreadPool::stream_contiguous(const void* ptr, size_t total_bytes) {
    m_stream_ptr = ptr;
    m_stream_bytes = total_bytes;
    m_mode.store(1, std::memory_order_relaxed);
    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    while (m_completed.load(std::memory_order_acquire) < NUM_WORKERS) {
        _mm_pause();
    }
}

uint64_t Flux2ThreadPool::get_total_sink() const {
    uint64_t s = 0;
    for (int i = 0; i < NUM_WORKERS; i++) s += m_worker_sinks[i];
    return s;
}

} // namespace flux2
