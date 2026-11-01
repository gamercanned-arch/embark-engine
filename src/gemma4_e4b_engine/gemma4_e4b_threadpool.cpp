#include "gemma4_e4b_threadpool.h"
#include <iostream>
#include <immintrin.h>

namespace gemma4_e4b {

// AMD Ryzen 7 7730U: 8 physical Zen 3 cores mapped to even logical IDs:
// Physical Cores: 0, 2, 4, 6, 8, 10, 12, 14
static const int PHYSICAL_CORE_MAP[NUM_GEMMA4_WORKERS] = {0, 2, 4, 6, 8, 10, 12, 14};

Gemma4E4BThreadPool::Gemma4E4BThreadPool() {
    for (int i = 0; i < NUM_GEMMA4_WORKERS; ++i) {
        m_worker_sinks[i] = 0;
        m_worker_done[i].store(0, std::memory_order_relaxed);
    }

    m_workers.reserve(NUM_GEMMA4_WORKERS);
    for (int i = 0; i < NUM_GEMMA4_WORKERS; ++i) {
        m_workers.emplace_back(&Gemma4E4BThreadPool::worker_main, this, i);
    }
}

Gemma4E4BThreadPool::~Gemma4E4BThreadPool() {
    m_terminate.store(true, std::memory_order_release);
    m_generation.fetch_add(1, std::memory_order_release);

    for (auto& th : m_workers) {
        if (th.joinable()) th.join();
    }
}

void Gemma4E4BThreadPool::worker_main(int worker_id) {
    // Pin worker thread to its dedicated physical Zen 3 core
    int core_id = PHYSICAL_CORE_MAP[worker_id];
    DWORD_PTR affinity_mask = (DWORD_PTR)(1ULL << core_id);
    SetThreadAffinityMask(GetCurrentThread(), affinity_mask);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    int last_gen = 0;

    while (true) {
        // High-performance lock-free spin-wait (< 200 ns latency)
        while (m_generation.load(std::memory_order_acquire) == last_gen) {
            if (m_terminate.load(std::memory_order_acquire)) return;
            _mm_pause();
        }

        last_gen = m_generation.load(std::memory_order_relaxed);
        if (m_terminate.load(std::memory_order_acquire)) return;

        if (m_active_task) {
            m_active_task(worker_id);
        }

        m_worker_done[worker_id].store(last_gen, std::memory_order_release);
    }
}

void Gemma4E4BThreadPool::dispatch(const std::function<void(int)>& task_func) {
    m_active_task = task_func;
    int next_gen = m_generation.load(std::memory_order_relaxed) + 1;

    // Wake up workers
    m_generation.store(next_gen, std::memory_order_release);

    // Wait for all 8 workers to complete the current generation
    for (int i = 0; i < NUM_GEMMA4_WORKERS; ++i) {
        while (m_worker_done[i].load(std::memory_order_acquire) != next_gen) {
            _mm_pause();
        }
    }
}

void Gemma4E4BThreadPool::stream_contiguous_block(const void* ptr, size_t total_bytes) {
    const uint8_t* base = (const uint8_t*)ptr;
    size_t chunk_per_worker = total_bytes / NUM_GEMMA4_WORKERS;
    // Align chunk size to 64 bytes
    chunk_per_worker = (chunk_per_worker + 63ULL) & ~63ULL;

    dispatch([&](int wid) {
        size_t start = wid * chunk_per_worker;
        size_t end = (wid == NUM_GEMMA4_WORKERS - 1) ? total_bytes : (start + chunk_per_worker);
        if (start >= total_bytes) return;
        if (end > total_bytes) end = total_bytes;

        const uint8_t* p = base + start;
        size_t bytes = end - start;
        size_t n_vecs = bytes / 32;

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        // Non-temporal streaming loads (MOVNTDQA) to bypass L3 cache pollution
        for (size_t i = 0; i < n_vecs; i += 2) {
            if (i + 16 < n_vecs) {
                _mm_prefetch((const char*)(p + (i + 16) * 32), _MM_HINT_NTA);
            }
            __m256i v0 = _mm256_stream_load_si256((const __m256i*)(p + i * 32));
            __m256i v1 = _mm256_stream_load_si256((const __m256i*)(p + (i + 1) * 32));
            acc0 = _mm256_xor_si256(acc0, v0);
            acc1 = _mm256_add_epi64(acc1, v1);
        }

        // Handle remainder
        size_t rem_start = n_vecs * 32;
        uint64_t rem_val = 0;
        for (size_t i = rem_start; i < bytes; ++i) {
            rem_val += p[i];
        }

        __m256i acc = _mm256_xor_si256(acc0, acc1);
        uint64_t v[4];
        _mm256_storeu_si256((__m256i*)v, acc);
        uint64_t local_sink = v[0] ^ v[1] ^ v[2] ^ v[3] ^ rem_val;

        add_sink(wid, local_sink);
    });
}

uint64_t Gemma4E4BThreadPool::get_total_sink() const {
    uint64_t total = 0;
    for (int i = 0; i < NUM_GEMMA4_WORKERS; ++i) {
        total += m_worker_sinks[i];
    }
    return total;
}

} // namespace gemma4_e4b
