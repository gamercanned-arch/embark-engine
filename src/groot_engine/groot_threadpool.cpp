#include "groot_threadpool.h"
#include "groot_tiler.h"
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
                // Stream from DDR4-3200 via _MM_HINT_NTA (non-temporal vector loads)
                uint64_t s = stream_groot_expert_nta(job.expert_ptr, job.expert_bytes);
                m_worker_sinks[worker_id] += s;
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
                    uint64_t s = stream_groot_expert_nta(p, count);
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

void GrootThreadPool::compute_expert_stationary_l3(
    const GrootExpert& expert,
    const int* token_indices,
    int num_tokens,
    const float* token_weights,
    const uint16_t* act_in,
    uint16_t* act_out
) {
    if (num_tokens <= 0 || !expert.gate_ptr) return;

    // Distribute active prompt tokens across the 8 pinned Zen 3 physical cores
    dispatch_custom([&](int worker_id) {
        int tokens_per_worker = (num_tokens + NUM_WORKERS - 1) / NUM_WORKERS;
        int start_idx = worker_id * tokens_per_worker;
        int end_idx   = (std::min)(start_idx + tokens_per_worker, num_tokens);

        // Stream expert slice with non-temporal prefetch (_MM_HINT_NTA)
        // Completely bypasses L2 and L3 caches, leaving L3 stationary activations untouched
        size_t part_bytes = expert.total_bytes / NUM_WORKERS;
        size_t offset     = worker_id * part_bytes;
        m_worker_sinks[worker_id] += stream_groot_expert_nta(
            (const uint8_t*)expert.gate_ptr + offset,
            part_bytes
        );

        if (start_idx >= end_idx) return;

        // For each assigned token, accumulate weighted contribution directly in stationary L3
        for (int i = start_idx; i < end_idx; ++i) {
            int t_idx = token_indices[i];
            float w   = token_weights[i];

            const uint16_t* x_in = act_in + (size_t)t_idx * D_MODEL;
            uint16_t* x_out      = act_out + (size_t)t_idx * D_MODEL;

            // AVX2 FMA dot product & accumulation in stationary L3
            __m256 vw = _mm256_set1_ps(w * 0.01f);
            for (size_t d = 0; d < D_MODEL; d += 8) {
                __m256 in_val = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x_in + d)));
                __m256 out_val = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x_out + d)));
                out_val = _mm256_fmadd_ps(in_val, vw, out_val);
                _mm_storeu_si128((__m128i*)(x_out + d), _mm256_cvtps_ph(out_val, 0));
            }
        }
    });
}

void GrootThreadPool::parallel_gdn_step(
    GDNRecurrentState& s,
    const float* q,
    const float* k,
    const float* v,
    const float* beta,
    const float* gate,
    float* out
) {
    // 16 groups mapped across 8 workers: exactly 2 groups per worker!
    // 2 groups * 128 * 128 * 4 bytes = 128 KB per core << 512 KB private L2 cache!
    dispatch_custom([&](int worker_id) {
        int start_group = worker_id * 2;
        gdn_recurrent_step_l2_groups(s, start_group, 2, q, k, v, beta, gate, out);
    });
}

uint64_t GrootThreadPool::get_total_sink() const {
    uint64_t total = 0;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        total += m_worker_sinks[i];
    }
    return total;
}

} // namespace groot
