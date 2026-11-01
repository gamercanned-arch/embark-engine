#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <cstdint>
#include <cstddef>

namespace lfm2 {

constexpr int NUM_LFM2_WORKERS = 8;

class LFM2ThreadPool {
public:
    LFM2ThreadPool();
    ~LFM2ThreadPool();

    LFM2ThreadPool(const LFM2ThreadPool&) = delete;
    LFM2ThreadPool& operator=(const LFM2ThreadPool&) = delete;

    // Dispatch a parallel task across all 8 pinned physical cores with < 150 ns spin barrier
    void dispatch(const std::function<void(int worker_id)>& task_func);

    // Parallel saturated streaming weight load using MOVNTDQA non-temporal loads
    void stream_contiguous_block(const void* ptr, size_t total_bytes);

    uint64_t get_total_sink() const;

private:
    std::vector<std::thread> m_workers;
    std::atomic<bool> m_terminate{false};
    std::atomic<int>  m_generation{0};
    std::atomic<int>  m_completed{0};
    std::function<void(int worker_id)> m_active_task;

    alignas(64) uint64_t m_worker_sinks[NUM_LFM2_WORKERS];
    alignas(64) std::atomic<int> m_worker_done[NUM_LFM2_WORKERS];

    void worker_main(int worker_id);
};

} // namespace lfm2
