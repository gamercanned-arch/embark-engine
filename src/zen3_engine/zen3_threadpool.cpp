#include "zen3_threadpool.h"
#include "zen3_iq_simd.h"
#include <iostream>
#include <iomanip>

namespace zen3 {

Zen3ThreadPool::Zen3ThreadPool() {
    for (int i = 0; i < NUM_ZEN3_WORKERS; ++i) {
        m_worker_sinks[i] = 0;
    }

    std::atomic<int> ready_count{0};

    for (int i = 0; i < NUM_ZEN3_WORKERS; ++i) {
        m_workers.emplace_back([this, i, &ready_count]() {
            // Pin strictly to physical cores 0, 2, 4, 6, 8, 10, 12, 14
            int physical_core = i;
            int logical_cpu   = i * 2;
            DWORD_PTR affinity_mask = (DWORD_PTR)(1ULL << logical_cpu);

            SetThreadAffinityMask(GetCurrentThread(), affinity_mask);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

            ready_count.fetch_add(1, std::memory_order_release);

            worker_main(i);
        });
    }

    // Wait for all 8 workers to finish affinity configuration
    while (ready_count.load(std::memory_order_acquire) < NUM_ZEN3_WORKERS) {
        std::this_thread::yield();
    }
}

Zen3ThreadPool::~Zen3ThreadPool() {
    m_terminate.store(true, std::memory_order_release);
    m_generation.fetch_add(1, std::memory_order_release);

    for (auto& w : m_workers) {
        if (w.joinable()) {
            w.join();
        }
    }
}

void Zen3ThreadPool::worker_main(int worker_id) {
    int last_gen = 0;

    while (!m_terminate.load(std::memory_order_relaxed)) {
        // Spin-wait for new task generation (< 200 ns latency)
        while (m_generation.load(std::memory_order_acquire) == last_gen) {
            if (m_terminate.load(std::memory_order_relaxed)) return;
            _mm_pause();
        }

        last_gen = m_generation.load(std::memory_order_relaxed);
        if (m_terminate.load(std::memory_order_relaxed)) return;

        int mode = m_mode.load(std::memory_order_relaxed);

        if (mode == 0) {
            // Assign 2 physical cores per active expert across the 4 selected experts:
            // Worker 0, 1 -> Expert 0 (part 0, 1)
            // Worker 2, 3 -> Expert 1 (part 0, 1)
            // Worker 4, 5 -> Expert 2 (part 0, 1)
            // Worker 6, 7 -> Expert 3 (part 0, 1)
            int exp_slot = worker_id / 2;
            int part     = worker_id % 2;

            const ExpertJob& job = m_jobs[exp_slot];
            if (job.expert_ptr && job.expert_bytes > 0) {
                // Split contiguous expert block into two halves aligned to 64 bytes
                size_t half_bytes = (job.expert_bytes / 2 + 63ULL) & ~63ULL;
                size_t start_offset = part * half_bytes;
                size_t end_offset   = (part == 1) ? job.expert_bytes : (std::min)(start_offset + half_bytes, job.expert_bytes);

                if (start_offset < job.expert_bytes) {
                    size_t bytes_to_stream = end_offset - start_offset;
                    m_worker_sinks[worker_id] += stream_expert_chunk_avx2(
                        job.expert_ptr + start_offset,
                        bytes_to_stream
                    );
                }
            }
        } else if (mode == 1) {
            if (m_custom_task) {
                m_custom_task(worker_id);
            }
        }

        m_completed.fetch_add(1, std::memory_order_release);
    }
}

void Zen3ThreadPool::stream_active_experts(const PackedExpert* active_experts[TOP_K]) {
    m_mode.store(0, std::memory_order_relaxed);
    for (int k = 0; k < TOP_K; ++k) {
        m_jobs[k].expert_ptr   = active_experts[k]->expert_base;
        m_jobs[k].expert_bytes = active_experts[k]->total_bytes;
        m_jobs[k].expert_idx   = k;
    }

    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    // Spin-wait for all 8 physical cores to finish streaming
    while (m_completed.load(std::memory_order_acquire) < NUM_ZEN3_WORKERS) {
        _mm_pause();
    }
}

void Zen3ThreadPool::compute_expert_stationary_l3(
    const PackedExpert& expert,
    const int* token_indices,
    int num_tokens,
    const float* token_weights,
    const uint16_t* act_in,
    uint16_t* act_out
) {
    if (num_tokens <= 0 || !expert.expert_base) return;

    // Distribute active tokens across the 8 pinned Zen 3 physical cores
    dispatch_custom([&](int worker_id) {
        int tokens_per_worker = (num_tokens + NUM_ZEN3_WORKERS - 1) / NUM_ZEN3_WORKERS;
        int start_idx = worker_id * tokens_per_worker;
        int end_idx   = (std::min)(start_idx + tokens_per_worker, num_tokens);

        // Stream expert slice with non-temporal prefetch to prevent evicting L3 activations
        size_t part_bytes = expert.total_bytes / NUM_ZEN3_WORKERS;
        size_t offset     = worker_id * part_bytes;
        m_worker_sinks[worker_id] += stream_expert_chunk_nta(
            expert.expert_base + offset,
            part_bytes
        );

        if (start_idx >= end_idx) return;

        // For each assigned token, compute dot-product with weights and accumulate into L3
        for (int i = start_idx; i < end_idx; ++i) {
            int t_idx = token_indices[i];
            float w   = token_weights[i];

            const uint16_t* x_in = act_in + (size_t)t_idx * D_MODEL;
            uint16_t* x_out      = act_out + (size_t)t_idx * D_MODEL;

            // AVX2 FMA dot product & accumulation in L3:
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

void Zen3ThreadPool::dispatch_custom(const std::function<void(int worker_id)>& task_func) {
    m_mode.store(1, std::memory_order_relaxed);
    m_custom_task = task_func;

    m_completed.store(0, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);

    while (m_completed.load(std::memory_order_acquire) < NUM_ZEN3_WORKERS) {
        _mm_pause();
    }
}

uint64_t Zen3ThreadPool::get_total_sink() const {
    uint64_t total = 0;
    for (int i = 0; i < NUM_ZEN3_WORKERS; ++i) {
        total += m_worker_sinks[i];
    }
    return total;
}

} // namespace zen3
