#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "apu_d1_orchestrator.h"

int main(int argc, char** argv) {
    std::string gguf_path = "C:\\Users\\abhik\\Downloads\\d1-2B-Q4_K_M.gguf";
    std::string mmproj_path = "C:\\Users\\abhik\\Downloads\\mmproj-d1-2B-F16.gguf";
    bool run_wl1 = true;
    bool run_wl2 = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (arg == "--wl1") {
            run_wl1 = true;
            run_wl2 = false;
        } else if (arg == "--wl2") {
            run_wl1 = false;
            run_wl2 = true;
        } else if (arg == "--all") {
            run_wl1 = true;
            run_wl2 = true;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "  d1-2B Tuned Dual-Engine APU Megakernel Benchmark\n";
    std::cout << "  Target: AMD Ryzen 7 7730U (Zen 3 8C/16T, 16MB L3, Vega 8 gfx90c, DDR4-3200)\n";
    std::cout << "  Model:  " << gguf_path << "\n";
    std::cout << "  Vision: " << mmproj_path << "\n";
    std::cout << "================================================================================\n\n";

    d1::APUD1Orchestrator engine(gguf_path, mmproj_path);
    if (!engine.initialize()) {
        std::cerr << "[-] Failed to initialize d1-2B APU Orchestrator\n";
        return 1;
    }

    std::cout << "\n[+] APU Dual-Engine Initialization Complete. Commencing Benchmarks...\n";

    d1::D1HybridRunResult res1;
    d1::D1HybridRunResult res2;

    if (run_wl1) {
        // Workload 1: Production pp512 + tg128
        res1 = engine.run_hybrid_workload(512, 128);
    }

    if (run_wl2) {
        // Workload 2: Extended Context pp2048 + tg512
        res2 = engine.run_hybrid_workload(2048, 512);
    }

    std::cout << "\n================================================================================\n";
    std::cout << "  FINAL SCORECARD: Tuned Dual-Engine APU Megakernel on d1-2B\n";
    std::cout << "================================================================================\n";
    std::cout << "| Metric                           | Stock Baseline | Tuned APU Engine | Improvement    |\n";
    std::cout << "| :------------------------------- | :------------- | :--------------- | :------------- |\n";
    std::cout << "| Backend Engine                   | CPU Fallback   | Dual Vega8+Zen3  | Native APU Act |\n";
    if (run_wl1) {
        std::cout << "| WL1 Prefill Speed (pp512)        | 57.78 tok/s    | " 
                  << std::fixed << std::setprecision(2) << res1.prefill.prefill_tokens_per_sec << " tok/s   | +"
                  << ((res1.prefill.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res1.prefill.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL1 Decode Speed (tg128)         | 11.75 tok/s    | "
                  << std::fixed << std::setprecision(2) << res1.decode.tokens_per_sec << " tok/s   | +"
                  << ((res1.decode.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res1.decode.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL1 Peak Burst Decode Speed      | ~12.50 tok/s   | "
                  << std::fixed << std::setprecision(2) << res1.decode.peak_tokens_per_sec << " tok/s   | Peak Burst     |\n";
        std::cout << "| WL1 Effective Memory Read        | ~12.80 GB/s    | "
                  << std::fixed << std::setprecision(2) << res1.decode.effective_bandwidth_gb_s << " GB/s   | Bus Saturation |\n";
    }
    if (run_wl2) {
        std::cout << "| WL2 Prefill Speed (pp2048)       | 64.05 tok/s    | "
                  << std::fixed << std::setprecision(2) << res2.prefill.prefill_tokens_per_sec << " tok/s   | +"
                  << ((res2.prefill.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res2.prefill.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL2 Decode Speed (tg512)         | 14.39 tok/s    | "
                  << std::fixed << std::setprecision(2) << res2.decode.tokens_per_sec << " tok/s   | +"
                  << ((res2.decode.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res2.decode.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL2 Effective Memory Read        | ~15.70 GB/s    | "
                  << std::fixed << std::setprecision(2) << res2.decode.effective_bandwidth_gb_s << " GB/s   | Bus Saturation |\n";
    }
    std::cout << "================================================================================\n";

    return 0;
}
