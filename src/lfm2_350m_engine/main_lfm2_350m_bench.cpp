#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "apu_lfm2_orchestrator.h"

int main(int argc, char** argv) {
    std::string gguf_path = "C:\\Users\\abhik\\Downloads\\LFM2.5-350M-Q8_0.gguf";
    bool run_wl1 = true;
    bool run_wl2 = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            gguf_path = argv[++i];
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
    std::cout << "  LFM 2.5 350M Tuned Dual-Engine APU Megakernel Benchmark\n";
    std::cout << "  Target: AMD Ryzen 7 7730U (Zen 3 8C/16T, 16MB L3, Vega 8 gfx90c, DDR4-3200)\n";
    std::cout << "  Model:  " << gguf_path << "\n";
    std::cout << "================================================================================\n\n";

    lfm2::APULFM2Orchestrator engine(gguf_path);
    if (!engine.initialize()) {
        std::cerr << "[-] Failed to initialize LFM 2.5 350M APU Orchestrator\n";
        return 1;
    }

    std::cout << "\n[+] APU Dual-Engine Initialization Complete. Commencing Benchmarks...\n";

    lfm2::LFM2HybridRunResult res1;
    lfm2::LFM2HybridRunResult res2;

    if (run_wl1) {
        // Workload 1: Production pp512 + tg128
        res1 = engine.run_hybrid_workload(512, 128);
    }

    if (run_wl2) {
        // Workload 2: Extended Context pp2048 + tg512
        res2 = engine.run_hybrid_workload(2048, 512);
    }

    std::cout << "\n================================================================================\n";
    std::cout << "  FINAL SCORECARD: Tuned Dual-Engine APU Megakernel on LFM 2.5 350M\n";
    std::cout << "================================================================================\n";
    std::cout << "| Metric                           | Stock Baseline | Tuned APU Engine | Improvement    |\n";
    std::cout << "| :------------------------------- | :------------- | :--------------- | :------------- |\n";
    std::cout << "| Backend Engine                   | CPU Fallback   | Dual Vega8+Zen3  | Native APU Act |\n";
    if (run_wl1) {
        std::cout << "| WL1 Prefill Speed (pp512)        | 155.97 tok/s   | " 
                  << std::fixed << std::setprecision(2) << res1.prefill.prefill_tokens_per_sec << " tok/s  | +"
                  << ((res1.prefill.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res1.prefill.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL1 Decode Speed (tg128)         | 40.26 tok/s    | "
                  << std::fixed << std::setprecision(2) << res1.decode.tokens_per_sec << " tok/s   | +"
                  << ((res1.decode.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res1.decode.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL1 Peak Burst Decode Speed      | ~43.25 tok/s   | "
                  << std::fixed << std::setprecision(2) << res1.decode.peak_tokens_per_sec << " tok/s   | Peak Burst     |\n";
        std::cout << "| WL1 Effective Memory Read        | ~14.47 GB/s    | "
                  << std::fixed << std::setprecision(2) << res1.decode.effective_bandwidth_gb_s << " GB/s   | Bus Saturation |\n";
    }
    if (run_wl2) {
        std::cout << "| WL2 Prefill Speed (pp2048)       | 151.91 tok/s   | "
                  << std::fixed << std::setprecision(2) << res2.prefill.prefill_tokens_per_sec << " tok/s  | +"
                  << ((res2.prefill.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res2.prefill.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL2 Decode Speed (tg512)         | 45.03 tok/s    | "
                  << std::fixed << std::setprecision(2) << res2.decode.tokens_per_sec << " tok/s   | +"
                  << ((res2.decode.speedup_vs_baseline - 1.0) * 100.0) << "% (" << res2.decode.speedup_vs_baseline << "x) |\n";
        std::cout << "| WL2 Effective Memory Read        | ~16.18 GB/s    | "
                  << std::fixed << std::setprecision(2) << res2.decode.effective_bandwidth_gb_s << " GB/s   | Bus Saturation |\n";
    }
    std::cout << "================================================================================\n";

    return 0;
}
