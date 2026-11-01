// main_apu_bench.cpp
// Unified Verification & Benchmarking CLI for Zen 3 + Vega 8 Dual-Engine MoE
#include "apu_orchestrator.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    zen3::OrchestratorConfig config;
    int prompt_len = 512;
    int gen_tokens = 64;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.gguf_path = argv[++i];
        } else if (arg == "--prompt-len" && i + 1 < argc) {
            prompt_len = std::stoi(argv[++i]);
        } else if (arg == "--gen-tokens" && i + 1 < argc) {
            gen_tokens = std::stoi(argv[++i]);
        }
    }

    try {
        zen3::APUOrchestrator orchestrator(config);

        if (!orchestrator.initialize()) {
            std::cerr << "[-] Orchestrator initialization failed!\n";
            return 1;
        }

        orchestrator.run_hybrid_inference_benchmark(prompt_len, gen_tokens);

    } catch (const std::exception& ex) {
        std::cerr << "[-] Fatal Exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
