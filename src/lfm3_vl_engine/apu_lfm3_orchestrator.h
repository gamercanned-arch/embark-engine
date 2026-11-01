#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <string>
#include <vector>
#include <memory>
#include "lfm3_tiler.h"
#include "lfm3_threadpool.h"
#include "opencl/vega8_context.h"

namespace lfm3 {

struct LFM3PrefillResult {
    int seq_len = 0;
    double gemm_latency_ms = 0.0;
    double sustained_tflops = 0.0;
    double shortconv_prefill_ms = 0.0;
    double full_model_prefill_ms = 0.0;
    double prefill_tokens_per_sec = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct LFM3DecodeResult {
    int num_tokens = 0;
    double median_latency_ms = 0.0;
    double mean_latency_ms = 0.0;
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

struct LFM3VisionResult {
    double vision_encode_ms = 0.0;
    double sustained_tflops = 0.0;
    double projector_tps = 0.0;
};

struct LFM3HybridRunResult {
    int prompt_len = 0;
    int gen_tokens = 0;
    LFM3PrefillResult prefill;
    LFM3DecodeResult  decode;
    LFM3VisionResult  vision;
    double total_latency_sec = 0.0;
    double end_to_end_tokens_per_sec = 0.0;
};

class APULFM3Orchestrator {
public:
    APULFM3Orchestrator(const std::string& gguf_path, const std::string& mmproj_path = "");
    ~APULFM3Orchestrator();

    bool initialize();

    // Benchmarking
    LFM3PrefillResult benchmark_prefill(int seq_len, int iters = 5);
    LFM3DecodeResult  benchmark_decode(int num_tokens);
    LFM3VisionResult  benchmark_vision_projector(int iters = 5);
    LFM3HybridRunResult run_hybrid_workload(int prompt_len, int gen_tokens);

    double get_peak_working_set_gb() const;

private:
    std::string m_gguf_path;
    std::string m_mmproj_path;
    bool m_initialized = false;
    double m_peak_ws_gb = 0.0;

    // Subsystems
    LFM3Tiler m_tiler;
    std::unique_ptr<LFM3ThreadPool> m_threadpool;
    std::unique_ptr<LFM3ShortConvState> m_shortconv_state;
    std::unique_ptr<LFM3ActiveLayerKVCache> m_kv_cache;
    std::unique_ptr<Vega8Context> m_vega8;

    // OpenCL Handles
    cl_program m_gpu_program = nullptr;
    cl_kernel  m_kernel_gemm_fp16 = nullptr;
    cl_kernel  m_kernel_gemm_fp32 = nullptr;
    cl_kernel  m_kernel_shortconv = nullptr;

    void update_peak_memory();
    bool setup_opencl();
};

} // namespace lfm3
