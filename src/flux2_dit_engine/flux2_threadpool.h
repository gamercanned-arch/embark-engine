// flux2_threadpool.h
// Pinned Zen 3 Worker Threadpool for FLUX.2 Klein 4B DiT Orchestration
// Pins 8 worker threads to physical cores: 0, 2, 4, 6, 8, 10, 12, 14
// Bypasses SMT sibling interference and achieves sub-150ns barrier latency.

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

namespace flux2 {

constexpr int NUM_WORKERS = 8;

class Flux2ThreadPool {
public:
    Flux2ThreadPool();
    ~Flux2ThreadPool();

    Flux2ThreadPool(const Flux2ThreadPool&) = delete;
    Flux2ThreadPool& operator=(const Flux2ThreadPool&) = delete;

    // Dispatches a parallel task function across the 8 physical Zen 3 cores
    void dispatch(const std::function<void(int worker_id)>& task_func);

    // Contiguous memory stream / pre-faulting across all 8 cores
    void stream_contiguous(const void* ptr, size_t total_bytes);

    uint64_t get_total_sink() const;

private:
    std::vector<std::thread> m_workers;
    std::atomic<int> m_generation{0};
    std::atomic<int> m_completed{0};
    std::atomic<bool> m_terminate{false};

    // Task state
    std::atomic<int> m_mode{0}; // 0 = custom lambda, 1 = memory streaming
    std::function<void(int)> m_custom_task;
    const void* m_stream_ptr = nullptr;
    size_t m_stream_bytes = 0;

    alignas(64) uint64_t m_worker_sinks[NUM_WORKERS];
};

} // namespace flux2
