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
#include "spark_tiler.h"
#include "spark_threadpool.h"
#include "spark_simd.h"
#include "opencl/vega8_context.h"

namespace spark {

struct SparkPrefillResult {
    int seq_len = 0;
    double gemm_latency_ms = 0.0;
    double sustained_tflops = 0.0;
    double full_model_prefill_ms = 0.0;
    double prefill_tokens_per_sec = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct SparkDecodeResult {
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

struct SparkHybridRunResult {
    int prompt_len = 0;
    int gen_tokens = 0;
    SparkPrefillResult prefill;
    SparkDecodeResult decode;
    double total_latency_sec = 0.0;
    double end_to_end_tokens_per_sec = 0.0;
};

class APUSparkOrchestrator {
public:
    explicit APUSparkOrchestrator(const std::string& gguf_path);
    ~APUSparkOrchestrator();

    APUSparkOrchestrator(const APUSparkOrchestrator&) = delete;
    APUSparkOrchestrator& operator=(const APUSparkOrchestrator&) = delete;

    bool initialize();

    // Engine B: Vega 8 OpenCL Wave64 LDS double-buffering prefill
    SparkPrefillResult benchmark_prefill(int seq_len, int iters = 5);

    // Engine A: Zen 3 Megakernel AVX2 non-temporal streaming decode
    SparkDecodeResult benchmark_decode(int num_tokens);

    // End-to-End Hybrid APU Workload (Engine B prefill + UMA Zero-Copy + Engine A decode)
    SparkHybridRunResult run_hybrid_workload(int prompt_len, int gen_tokens);

    double get_peak_working_set_gb() const;

private:
    std::string m_gguf_path;
    bool m_initialized = false;

    SparkTiler m_tiler;
    std::unique_ptr<SparkThreadPool> m_threadpool;
    std::unique_ptr<SparkActiveLayerKVCache> m_kv_cache;

    std::unique_ptr<Vega8Context> m_vega8;
    cl_program m_gpu_program = nullptr;
    cl_kernel  m_kernel_gemm_fp16 = nullptr;
    cl_kernel  m_kernel_gemm_fp32 = nullptr;

    double m_peak_ws_gb = 0.0;

    bool setup_opencl();
    void update_peak_memory();
};

} // namespace spark
