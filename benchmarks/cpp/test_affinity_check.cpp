#include <windows.h>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    constexpr int NUM_THREADS = 8;
    std::vector<std::thread> workers;
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        int core_id = i * 2; // Cores 0, 2, 4, 6, 8, 10, 12, 14
        DWORD_PTR mask = (DWORD_PTR)(1ULL << core_id);
        
        workers.emplace_back([i, core_id, mask]() {
            DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            std::cout << "Thread " << i << " pinned to Physical Core " << (core_id / 2)
                      << " (Logical CPU " << core_id << ", mask 0x" << std::hex << mask << std::dec << ")\n";
        });
    }
    
    for (auto& w : workers) w.join();
    return 0;
}
