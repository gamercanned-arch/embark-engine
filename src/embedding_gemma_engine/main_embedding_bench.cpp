#include "apu_embed_orchestrator.h"
#include <iostream>
#include <iomanip>
#include <string>

using namespace embed_gemma;

int main(int argc, char** argv) {
    std::string text_path = "C:\\Users\\abhik\\Downloads\\embeddinggemma-2-UD-Q6_K_XL.gguf";
    std::string mmproj_path = "C:\\Users\\abhik\\Downloads\\mmproj-F16 (1).gguf";

    bool run_p512 = true;
    bool run_p2048 = true;
    bool run_vision = true;
    bool run_multimodal = true;
    int iters = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            text_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--p512") {
            run_p512 = true; run_p2048 = false; run_vision = false; run_multimodal = false;
        } else if (arg == "--p2048") {
            run_p512 = false; run_p2048 = true; run_vision = false; run_multimodal = false;
        } else if (arg == "--vision") {
            run_p512 = false; run_p2048 = false; run_vision = true; run_multimodal = false;
        } else if (arg == "--multimodal") {
            run_p512 = false; run_p2048 = false; run_vision = false; run_multimodal = true;
        } else if (arg == "--all") {
            run_p512 = true; run_p2048 = true; run_vision = true; run_multimodal = true;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "  EmbeddingGemma-2 APU Megakernel Benchmark Suite\n";
    std::cout << "  Platform: AMD Ryzen 7 7730U APU (Zen 3 8C/16T, Vega 8 gfx90c iGPU, DDR4-3200)\n";
    std::cout << "  Backbone: " << text_path << "\n";
    std::cout << "  Vision:   " << mmproj_path << "\n";
    std::cout << "================================================================================\n\n";

    APUEmbedOrchestrator orchestrator(text_path, mmproj_path);
    if (!orchestrator.initialize()) {
        std::cerr << "[-] Failed to initialize EmbeddingGemma APU engine!" << std::endl;
        return 1;
    }

    TextEncodingResult res512, res2048;
    VisionEncodingResult res_vis;
    MultimodalEmbeddingResult res_mm;

    if (run_p512) {
        std::cout << "\n--------------------------------------------------------------------------------\n";
        std::cout << "  [Workload 1] p512 Bidirectional Encoding (512 tokens -> 768-dim embedding)\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        res512 = orchestrator.benchmark_text_encoding(512, iters);
    }

    if (run_p2048) {
        std::cout << "\n--------------------------------------------------------------------------------\n";
        std::cout << "  [Workload 2] p2048 Extended Context Encoding (2048 tokens -> 768-dim embedding)\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        res2048 = orchestrator.benchmark_text_encoding(2048, iters);
    }

    if (run_vision) {
        std::cout << "\n--------------------------------------------------------------------------------\n";
        std::cout << "  [Workload 3] Multimodal Image Projector (224x224 RGB -> 196 patches @ 512-dim)\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        res_vis = orchestrator.benchmark_vision_encoding(iters);
    }

    if (run_multimodal) {
        std::cout << "\n--------------------------------------------------------------------------------\n";
        std::cout << "  [Workload 4] End-to-End Multimodal Embedding (Image + 512 Text Tokens = 708 Tokens)\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        res_mm = orchestrator.benchmark_multimodal_embedding(512, iters);
    }

    std::cout << "\n================================================================================\n";
    std::cout << "  FINAL EMBEDDINGGEMMA-2 APU PERFORMANCE SUMMARY\n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(28) << "Metric" 
              << std::right << std::setw(18) << "Stock llama.cpp" 
              << std::setw(18) << "Tuned APU Engine" 
              << std::setw(16) << "Speedup" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    if (run_p512) {
        std::cout << std::left << std::setw(28) << "p512 Throughput (t/s)" 
                  << std::right << std::setw(18) << "387.01 t/s" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res512.tokens_per_sec << " t/s"
                  << std::setw(15) << std::fixed << std::setprecision(2) << res512.speedup_vs_baseline << "x\n";
        std::cout << std::left << std::setw(28) << "p512 Latency (ms)" 
                  << std::right << std::setw(18) << "1322.96 ms" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res512.latency_ms << " ms"
                  << std::setw(15) << std::fixed << std::setprecision(2) << res512.speedup_vs_baseline << "x\n";
    }

    if (run_p2048) {
        std::cout << std::left << std::setw(28) << "p2048 Throughput (t/s)" 
                  << std::right << std::setw(18) << "291.67 t/s" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res2048.tokens_per_sec << " t/s"
                  << std::setw(15) << std::fixed << std::setprecision(2) << res2048.speedup_vs_baseline << "x\n";
        std::cout << std::left << std::setw(28) << "p2048 Latency (ms)" 
                  << std::right << std::setw(18) << "7021.63 ms" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res2048.latency_ms << " ms"
                  << std::setw(15) << std::fixed << std::setprecision(2) << res2048.speedup_vs_baseline << "x\n";
    }

    if (run_vision) {
        std::cout << std::left << std::setw(28) << "Vision Latency (ms/img)" 
                  << std::right << std::setw(18) << "N/A (OOM/Crash)" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res_vis.latency_ms << " ms"
                  << std::setw(15) << "Wave64 UMA\n";
        std::cout << std::left << std::setw(28) << "Vision Compute (TFLOPS)" 
                  << std::right << std::setw(18) << "N/A" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res_vis.sustained_tflops << " TFLOPS"
                  << std::setw(15) << "RPM Peak 4.1\n";
    }

    if (run_multimodal) {
        std::cout << std::left << std::setw(28) << "E2E Multimodal TPS" 
                  << std::right << std::setw(18) << "N/A" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res_mm.end_to_end_tokens_per_sec << " t/s"
                  << std::setw(15) << "Dual-Engine\n";
        std::cout << std::left << std::setw(28) << "E2E Latency (ms)" 
                  << std::right << std::setw(18) << "N/A" 
                  << std::setw(16) << std::fixed << std::setprecision(2) << res_mm.total_latency_ms << " ms"
                  << std::setw(15) << "Zen3+Vega8\n";
    }

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Peak Working Set: " << std::fixed << std::setprecision(2) << orchestrator.get_peak_working_set_gb() << " GiB / 16.0 GiB (UMA Zero-Copy)\n";
    std::cout << "================================================================================\n\n";

    return 0;
}
