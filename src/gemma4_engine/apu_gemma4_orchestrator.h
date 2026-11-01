#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "gemma4_tiler.h"
#include "gemma4_threadpool.h"
#include "gemma4_simd.h"
#include "opencl/vega8_context.h"

namespace gemma4 {

struct Gemma4PrefillResult {
    int seq_len = 0;
    double gemm_latency_ms = 0.0;
    double sustained_tflops = 0.0;
    double full_model_prefill_ms = 0.0;
    double prefill_tokens_per_sec = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct Gemma4DecodeResult {
    int num_tokens = 0;
    double mean_latency_ms = 0.0;
    double median_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    double p90_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double tokens_per_sec = 0.0;
    double peak_tokens_per_sec = 0.0;
    double effective_bandwidth_gb_s = 0.0;
    double peak_working_set_gb = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct Gemma4HybridRunResult {
    int prompt_len = 0;
    int gen_tokens = 0;
    Gemma4PrefillResult prefill;
    Gemma4DecodeResult decode;
    double total_latency_sec = 0.0;
    double end_to_end_tokens_per_sec = 0.0;
};

class APUGemma4Orchestrator {
public:
    APUGemma4Orchestrator(const std::string& gguf_path);
    ~APUGemma4Orchestrator();

    APUGemma4Orchestrator(const APUGemma4Orchestrator&) = delete;
    APUGemma4Orchestrator& operator=(const APUGemma4Orchestrator&) = delete;

    bool initialize();

    // Engine B: Vega 8 GPU Prompt Prefill (M = 512, 2048)
    Gemma4PrefillResult benchmark_prefill(int seq_len = 512, int iters = 5);

    // Engine A: Zen 3 AVX2 Megakernel Decode (M = 1, N tokens)
    Gemma4DecodeResult benchmark_decode(int num_tokens = 128);

    // End-to-End Hybrid APU Execution: Prefill on Vega 8 -> Zero-Copy UMA Handoff -> Decode on Zen 3
    Gemma4HybridRunResult run_hybrid_workload(int prompt_len, int gen_tokens);

    double get_peak_working_set_gb() const;

private:
    std::string m_gguf_path;
    Gemma4Tiler m_tiler;
    std::unique_ptr<Gemma4ThreadPool> m_threadpool;
    std::unique_ptr<Vega8Context> m_vega8;
    std::unique_ptr<Gemma4KVCache> m_kv_cache;

    cl_program m_gpu_program = nullptr;
    cl_kernel  m_kernel_gemm_fp16 = nullptr;
    cl_kernel  m_kernel_gemm_fp32 = nullptr;

    bool m_initialized = false;
    double m_peak_ws_gb = 0.0;

    void update_peak_memory();
    bool setup_opencl();
};

} // namespace gemma4
