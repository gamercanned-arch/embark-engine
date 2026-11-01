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
#include "groot_tiler.h"
#include "groot_threadpool.h"
#include "opencl/vega8_context.h"

namespace groot {

struct GrootPrefillResult {
    int seq_len = 0;
    double gemm_latency_ms = 0.0;
    double sustained_tflops = 0.0;
    double full_model_prefill_ms = 0.0;
    double prefill_tokens_per_sec = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct GrootDecodeResult {
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
    double hot_expert_hit_rate = 0.0;
    double ddr4_traffic_saved_pct = 0.0;
    double peak_working_set_gb = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct GrootVisionResult {
    double vision_encode_ms = 0.0;
    double sustained_tflops = 0.0;
    double projector_tps = 0.0;
};

struct GrootHybridRunResult {
    int prompt_len = 0;
    int gen_tokens = 0;
    GrootPrefillResult prefill;
    GrootDecodeResult  decode;
    GrootVisionResult  vision;
    double total_latency_sec = 0.0;
    double end_to_end_tokens_per_sec = 0.0;
};

class APUGrootOrchestrator {
public:
    APUGrootOrchestrator(const std::string& gguf_path, const std::string& mmproj_path = "");
    ~APUGrootOrchestrator();

    bool initialize();

    // Benchmarking
    GrootPrefillResult benchmark_pipelined_prefill(int seq_len);
    GrootPrefillResult benchmark_prefill(int seq_len, int iters = 5);
    GrootDecodeResult  benchmark_decode(int num_tokens);
    GrootVisionResult  benchmark_vision_projector(int iters = 5);
    GrootHybridRunResult run_hybrid_workload(int prompt_len, int gen_tokens);

    double get_peak_working_set_gb() const;

private:
    std::string m_gguf_path;
    std::string m_mmproj_path;
    bool m_initialized = false;
    double m_peak_ws_gb = 0.0;

    // Subsystems
    GrootTiler m_tiler;
    std::unique_ptr<GrootThreadPool> m_threadpool;
    std::vector<GDNRecurrentState> m_gdn_states;
    std::unique_ptr<Vega8Context> m_vega8;

    // OpenCL Handles
    cl_program m_gpu_program = nullptr;
    cl_kernel  m_kernel_gemm_half2 = nullptr;
    cl_kernel  m_kernel_vision_fp16 = nullptr;
    cl_mem     m_gpu_pool_buffer = nullptr;

    // Stationary L3 Zero-Copy UMA Buffers
    cl_mem     m_gpu_act_in = nullptr;
    cl_mem     m_gpu_act_out = nullptr;
    cl_mem     m_gpu_shexp_out = nullptr;
    cl_mem     m_gpu_shexp_w = nullptr;
    cl_mem     m_gpu_attn_w = nullptr;

    void update_peak_memory();
    bool setup_opencl();
};

} // namespace groot
