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
#include <cstdint>

namespace embed_gemma {

constexpr int NUM_ZEN3_CORES = 8;

class EmbedThreadPool {
public:
    EmbedThreadPool();
    ~EmbedThreadPool();

    EmbedThreadPool(const EmbedThreadPool&) = delete;
    EmbedThreadPool& operator=(const EmbedThreadPool&) = delete;

    // Dispatches a lambda/task across all 8 pinned physical cores (0, 2, 4, 6, 8, 10, 12, 14)
    // with sub-microsecond spin-wait barrier sync (< 200 ns).
    void dispatch(const std::function<void(int worker_id)>& task_func);

    // Parallel streaming over contiguous memory buffer across 8 pinned workers using MOVNTDQA
    void stream_contiguous_block(const void* ptr, size_t total_bytes);

    // Cumulative sink value across all workers to prevent dead-code elimination
    uint64_t get_total_sink() const;
    void add_sink(int worker_id, uint64_t val) { m_worker_sinks[worker_id] += val; }

private:
    std::vector<std::thread> m_workers;
    std::atomic<int> m_generation{0};
    std::atomic<int> m_completed{0};
    std::atomic<bool> m_terminate{false};

    std::function<void(int)> m_active_task;

    alignas(64) uint64_t m_worker_sinks[NUM_ZEN3_CORES];
    alignas(64) std::atomic<int> m_worker_done[NUM_ZEN3_CORES];

    void worker_main(int worker_id);
};

} // namespace embed_gemma
