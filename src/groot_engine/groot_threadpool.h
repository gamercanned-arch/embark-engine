#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include "groot_simd.h"

namespace groot {

constexpr int NUM_WORKERS = 8;

struct GrootExpert; // Forward decl

struct ExpertStreamJob {
    const uint8_t* expert_ptr = nullptr;
    size_t expert_bytes = 0;
    int expert_idx = 0;
};

class GrootThreadPool {
public:
    GrootThreadPool();
    ~GrootThreadPool();

    GrootThreadPool(const GrootThreadPool&) = delete;
    GrootThreadPool& operator=(const GrootThreadPool&) = delete;

    // Dispatches the 8 active experts across the 8 pinned physical cores
    void stream_active_8_experts(const ExpertStreamJob jobs[EXPERTS_PER_TOKEN]);

    // Dispatch custom parallel lambda across 8 pinned physical workers
    void dispatch_custom(const std::function<void(int worker_id)>& task_func);

    // Contiguous block streaming across all 8 cores
    void stream_contiguous_block(const void* ptr, size_t total_bytes);

    // Stationary L3 Prefill MoE Kernel: distributes active prompt tokens for expert e across 8 cores
    void compute_expert_stationary_l3(
        const GrootExpert& expert,
        const int* token_indices,
        int num_tokens,
        const float* token_weights,
        const uint16_t* act_in,
        uint16_t* act_out
    );

    // Parallel GDN Recurrent Step: assigns 2 heads per core (128 KB/core in private L2)
    void parallel_gdn_step(
        GDNRecurrentState& s,
        const float* q,
        const float* k,
        const float* v,
        const float* beta,
        const float* gate,
        float* out
    );

    uint64_t get_total_sink() const;

private:
    std::vector<std::thread> m_workers;
    std::atomic<int> m_generation{0};
    std::atomic<int> m_completed{0};
    std::atomic<bool> m_terminate{false};

    // Mode: 0 = 1 core per active expert (8 experts), 1 = contiguous chunk across 8 cores, 2 = custom lambda
    std::atomic<int> m_mode{0};
    ExpertStreamJob m_expert_jobs[EXPERTS_PER_TOKEN];
    const void* m_contig_ptr = nullptr;
    size_t m_contig_bytes = 0;
    std::function<void(int)> m_custom_task;

    alignas(64) uint64_t m_worker_sinks[NUM_WORKERS];

    void worker_main(int worker_id);
};

} // namespace groot
