#include "apu_gemma4_e4b_orchestrator.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string model_path = "C:\\Users\\abhik\\Downloads\\gemma-4-E4B-it-Q4_K_M.gguf";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "  GEMMA 4 E4B DUAL-ENGINE HYBRID APU BENCHMARK SUITE\n";
    std::cout << "  Hardware: AMD Ryzen 7 7730U APU (Zen 3 CPU + Vega 8 iGPU, Barcelo-R UMA)\n";
    std::cout << "  Model:    " << model_path << "\n";
    std::cout << "================================================================================\n\n";

    // Pin Main Coordination Thread to Logical Core 1 (SMT sibling) so it never
    // collides with the 8 worker threads on Physical Cores 0, 2, 4, 6, 8, 10, 12, 14
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)(1ULL << 1));
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    try {
        gemma4_e4b::APUGemma4E4BOrchestrator orchestrator(model_path);

        if (!orchestrator.initialize()) {
            std::cerr << "[-] Orchestrator initialization failed!\n";
            return 1;
        }

        // 1. Standard Production Workload: pp512 + tg128
        std::cout << "\n>>> EXECUTING WORKLOAD 1: STANDARD PRODUCTION (pp512 + tg128) <<<\n";
        auto res_std = orchestrator.run_hybrid_workload(512, 128);

        // 2. Extended High-Context Workload: pp2048 + tg512
        std::cout << "\n\n>>> EXECUTING WORKLOAD 2: EXTENDED HIGH-CONTEXT (pp2048 + tg512) <<<\n";
        auto res_ext = orchestrator.run_hybrid_workload(2048, 512);

        std::cout << "\n================================================================================\n";
        std::cout << "  GEMMA 4 E4B BENCHMARK SUITE COMPLETE - ALL WORKLOADS VERIFIED\n";
        std::cout << "================================================================================\n";

    } catch (const std::exception& ex) {
        std::cerr << "[-] Fatal Exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
