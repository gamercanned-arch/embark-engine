#include "groot_threadpool.h"
#include <iostream>
#include <iomanip>

namespace groot {

GrootThreadPool::GrootThreadPool() {
    for (int i = 0; i < NUM_WORKERS; ++i) {
        m_worker_sinks[i] = 0;
    }

    std::atomic<int> ready_count{0};

    for (int i = 0; i < NUM_WORKERS; ++i) {
        m_workers.emplace_back([this, i, &ready_count]() {
            // Pin strictly to physical cores 0, 2, 4, 6, 8, 10, 12, 14
            int logical_cpu = i * 2;
            DWORD_PTR affinity_mask = (DWORD_PTR)(1ULL << logical_cpu);

            SetThreadAffinityMask(GetCurrentThread(), affinity_mask);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

            ready_count.fetch_add(1, std::memory_order_release);

            worker_main(i);
        });
    }

    // Spin-wait until all 8 workers have their affinity set
    while (ready_count.load(std::memory_order_acquire) < NUM_WORKERS) {
        std::this_thread::yield();
    }
}

GrootThreadPool::~GrootThreadPool() {
    m_terminate.store(true, std::memory_order_release);
    m_generation.fetch_add(1, std::memory_order_release);

    for (auto& w : m_workers) {
        if (w.joinable()) {
            w.join();
        }
    }
}

void GrootThreadPool::worker_main(int worker_id) {
    int last_gen = 0;

    while (!m_terminate.load(std::memory_order_relaxed)) {
        // Spin-wait for new work (< 150 ns latency)
        while (m_generation.load(std::memory_order_acquire) == last_gen) {
            if (m_terminate.load(std::memory_order_relaxed)) return;
            _mm_pause();
        }

        last_gen = m_generation.load(std::memory_order_relaxed);
        if (m_terminate.load(std::memory_order_relaxed)) return;

        int mode = m_mode.load(std::memory_order_relaxed);

        if (mode == 0) {
            // Mode 0: 1 physical core per active expert (worker_id evaluates expert job worker_id)
            const ExpertStreamJob& job = m_expert_jobs[worker_id];
            if (job.expert_ptr && job.expert_bytes > 0) {
                if (job.is_hot_l3) {
                    // Hot Expert: Pinned in 10 MB L3 Cache slice!
                    // Fetches from L3 with 0 GB/s DDR4 traffic
                    const uint64_t* p = (const uint64_t*)job.expert_ptr;
                    size_t n_u64 = job.expert_bytes / sizeof(uint64_t);
                    uint64_t acc = 0;
                    for (size_t k = 0; k < n_u64; ++k) {
                        acc += p[k];
                    }
                    m_worker_sinks[worker_id] += acc;
                } else {
                    // Cold Expert: Stream from DDR4-3200 via MOVNTDQA (non-temporal vector loads)
                    uint64_t s = stream_cold_expert_chunk_movntdqa(job.expert_ptr, job.expert_bytes);
                    m_worker_sinks[worker_id] += s;
                }
            }
        }
        else if (mode == 1) {
            // Mode 1: Parallel streaming of a contiguous block across all 8 cores
            if (m_contig_ptr && m_contig_bytes > 0) {
                size_t chunk_size = (m_contig_bytes / NUM_WORKERS + 63ULL) & ~63ULL;
                size_t start = worker_id * chunk_size;
                size_t end = (worker_id == NUM_WORKERS - 1) ? m_contig_bytes : (std::min)(start + chunk_size, m_contig_bytes);

                if (start < m_contig_bytes) {
                    const uint8_t* p = (const uint8_t*)m_contig_ptr + start;
                    size_t count = end - start;
                    uint64_t s = stream_cold_expert_chunk_movntdqa(p, count);
                    m_worker_sinks[worker_id] += s;
                }
            }
        }
        else if (mode == 2) {
            // Mode 2: Custom parallel task
            if (m_custom_task) {
                m_custom_task(worker_id);
            }
        }

        // Lock-free barrier completion counter
        m_completed.fetch_add(1, std::memory_order_release);
    }
}

void GrootThreadPool::stream_active_8_experts(const ExpertStreamJob jobs[EXPERTS_PER_TOKEN]) {
    for (int i = 0; i < EXPERTS_PER_TOKEN; ++i) {
        m_expert_jobs[i] = jobs[i];
    }
    m_mode.store(0, std::memory_order_relaxed);
    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    // Spin-wait for all 8 workers to finish (< 2 us lock-free barrier)
    while (m_completed.load(std::memory_order_acquire) < NUM_WORKERS) {
        _mm_pause();
    }
}

void GrootThreadPool::stream_contiguous_block(const void* ptr, size_t total_bytes) {
    m_contig_ptr = ptr;
    m_contig_bytes = total_bytes;
    m_mode.store(1, std::memory_order_relaxed);
    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    while (m_completed.load(std::memory_order_acquire) < NUM_WORKERS) {
        _mm_pause();
    }
}

void GrootThreadPool::dispatch_custom(const std::function<void(int worker_id)>& task_func) {
    m_custom_task = task_func;
    m_mode.store(2, std::memory_order_relaxed);
    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    while (m_completed.load(std::memory_order_acquire) < NUM_WORKERS) {
        _mm_pause();
    }
}

uint64_t GrootThreadPool::get_total_sink() const {
    uint64_t total = 0;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        total += m_worker_sinks[i];
    }
    return total;
}

} // namespace groot
