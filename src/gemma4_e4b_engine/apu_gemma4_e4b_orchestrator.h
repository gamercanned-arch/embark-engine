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
#include "gemma4_e4b_tiler.h"
#include "gemma4_e4b_threadpool.h"
#include "gemma4_e4b_simd.h"
#include "opencl/vega8_context.h"

namespace gemma4_e4b {

struct Gemma4E4BPrefillResult {
    int seq_len = 0;
    double gemm_latency_ms = 0.0;
    double sustained_tflops = 0.0;
    double full_model_prefill_ms = 0.0;
    double prefill_tokens_per_sec = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct Gemma4E4BDecodeResult {
    int num_tokens = 0;
    double mean_latency_ms = 0.0;
    double median_latency_ms = 0.0;
    double min_latency_ms = 0.0; // Peak burst
    double p90_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double tokens_per_sec = 0.0;
    double peak_tokens_per_sec = 0.0;
    double effective_bandwidth_gb_s = 0.0;
    double peak_working_set_gb = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct Gemma4E4BHybridRunResult {
    int prompt_len = 0;
    int gen_tokens = 0;
    Gemma4E4BPrefillResult prefill;
    Gemma4E4BDecodeResult decode;
    double total_latency_sec = 0.0;
    double end_to_end_tokens_per_sec = 0.0;
};

class APUGemma4E4BOrchestrator {
public:
    explicit APUGemma4E4BOrchestrator(const std::string& gguf_path);
    ~APUGemma4E4BOrchestrator();

    APUGemma4E4BOrchestrator(const APUGemma4E4BOrchestrator&) = delete;
    APUGemma4E4BOrchestrator& operator=(const APUGemma4E4BOrchestrator&) = delete;

    bool initialize();

    // Engine B: Vega 8 OpenCL Wave64 LDS double-buffering prefill
    Gemma4E4BPrefillResult benchmark_prefill(int seq_len, int iters = 5);

    // Engine A: Zen 3 Megakernel AVX2 non-temporal streaming decode
    Gemma4E4BDecodeResult benchmark_decode(int num_tokens);

    // End-to-End Hybrid APU Workload (Engine B prefill + UMA Zero-Copy + Engine A decode)
    Gemma4E4BHybridRunResult run_hybrid_workload(int prompt_len, int gen_tokens);

    double get_peak_working_set_gb() const;

private:
    std::string m_gguf_path;
    bool m_initialized = false;

    Gemma4E4BTiler m_tiler;
    std::unique_ptr<Gemma4E4BThreadPool> m_threadpool;
    std::unique_ptr<Gemma4E4BActiveLayerKVCache> m_kv_cache;

    std::unique_ptr<Vega8Context> m_vega8;
    cl_program m_gpu_program = nullptr;
    cl_kernel  m_kernel_gemm_fp16 = nullptr;
    cl_kernel  m_kernel_gemm_fp32 = nullptr;

    double m_peak_ws_gb = 0.0;

    bool setup_opencl();
    void update_peak_memory();
};

} // namespace gemma4_e4b
