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
#include "lfm2_tiler.h"

namespace zen3 {

constexpr int NUM_ZEN3_WORKERS = 8;

struct ExpertJob {
    const uint8_t* expert_ptr = nullptr;
    size_t expert_bytes = 0;
    int expert_idx = 0;
};

class Zen3ThreadPool {
public:
    Zen3ThreadPool();
    ~Zen3ThreadPool();

    // Prevent copying
    Zen3ThreadPool(const Zen3ThreadPool&) = delete;
    Zen3ThreadPool& operator=(const Zen3ThreadPool&) = delete;

    // Dispatches the 4 active experts across the 8 pinned physical cores
    // (2 physical cores assigned per active expert)
    void stream_active_experts(const PackedExpert* active_experts[TOP_K]);

    // Arbitrary work dispatch across the 8 pinned workers
    void dispatch_custom(const std::function<void(int worker_id)>& task_func);

    // Cumulative sink to prevent compiler DCE
    uint64_t get_total_sink() const;

private:
    std::vector<std::thread> m_workers;
    std::atomic<int> m_generation{0};
    std::atomic<int> m_completed{0};
    std::atomic<bool> m_terminate{false};

    // Mode: 0 = 2 cores per expert, 1 = 8 cores contiguous stream, 2 = custom function
    std::atomic<int> m_mode{0};
    ExpertJob m_jobs[NUM_ZEN3_WORKERS];
    std::function<void(int)> m_custom_task;

    alignas(64) uint64_t m_worker_sinks[NUM_ZEN3_WORKERS];
    alignas(64) std::atomic<int> m_worker_done[NUM_ZEN3_WORKERS];

    void worker_main(int worker_id);
public:
    // Streams a contiguous 4-expert block with all 8 physical cores (reaching full DDR4-3200 line rate >= 30 GB/s)
    void stream_contiguous_block(const void* ptr, size_t total_bytes);
};

} // namespace zen3
