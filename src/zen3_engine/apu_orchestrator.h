// apu_orchestrator.h
// Unified Dual-Engine APU Orchestrator for AMD Ryzen 7 7730U (Zen 3 CPU + Vega 8 iGPU)
// Coordinates Zero-Copy UMA Handoff between GPU Prefill (M >= 32) and CPU Decode (M = 1)
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

#include "lfm2_tiler.h"
#include "zen3_threadpool.h"
#include "zen3_iq_simd.h"
#include "opencl/vega8_context.h"

namespace zen3 {

struct OrchestratorConfig {
    std::string gguf_path = "C:\\Users\\abhik\\Downloads\\LFM2.5-24B-A2B-UD-IQ3_XXS.gguf";
    int prefill_batch_threshold = 32; // Use Vega 8 for M >= 32, Zen 3 for M < 32
    bool enable_gpu_prefill = true;
    bool enable_cpu_decode = true;
};

struct PrefillBenchmarkResult {
    int seq_len = 0;
    double gemm_latency_ms = 0.0;
    double sustained_tflops = 0.0;
    double gemm_tokens_per_sec = 0.0; // Single isolated projection throughput
    double full_model_tflops_workload = 0.0; // Full 38-layer transformer forward pass FLOPs
    double full_model_prefill_ms = 0.0; // True full-model prefill latency
    double full_model_tokens_per_sec = 0.0; // True full-model prefill throughput
    double speedup_vs_baseline = 0.0;
};

struct DecodeBenchmarkResult {
    int num_tokens = 0;
    double mean_latency_ms = 0.0;
    double median_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double tokens_per_sec = 0.0;
    double peak_tokens_per_sec = 0.0;
    double effective_moe_bandwidth_gb_s = 0.0;
    double peak_working_set_gb = 0.0;
};

class APUOrchestrator {
public:
    APUOrchestrator(const OrchestratorConfig& config = OrchestratorConfig());
    ~APUOrchestrator();

    // Prevent copy
    APUOrchestrator(const APUOrchestrator&) = delete;
    APUOrchestrator& operator=(const APUOrchestrator&) = delete;

    // Initialize both engines:
    // 1. Chunked Direct-I/O staging into 7.81 GB contiguous pool
    // 2. Vega 8 OpenCL context and Zero-Copy UMA mapping
    // 3. Zen 3 pinned 8-core threadpool
    bool initialize();

    // Engine B: Vega 8 GPU Prompt Prefill (M >= 32)
    PrefillBenchmarkResult benchmark_prefill(int seq_len = 512, int num_iters = 10);

    // Engine A: Zen 3 AVX2 Contiguous MoE Decode (M = 1)
    DecodeBenchmarkResult benchmark_decode(int num_tokens = 64);

    // End-to-End Hybrid APU Execution:
    // Prefill prompt on GPU -> Handoff via UMA RAM -> Decode tokens on CPU
    void run_hybrid_inference_benchmark(int prompt_len = 512, int gen_tokens = 64);

    // System Telemetry
    double get_current_working_set_gb() const;
    double get_peak_working_set_gb() const;

private:
    OrchestratorConfig m_config;
    LFM2Tiler m_tiler;
    std::unique_ptr<Zen3ThreadPool> m_threadpool;
    std::unique_ptr<zen3_engine::Vega8Context> m_vega8;

    cl_mem m_gpu_expert_pool = nullptr;
    cl_program m_gpu_program = nullptr;
    cl_kernel m_kernel_prefill = nullptr;

    bool m_initialized = false;
    double m_peak_ws_gb = 0.0;

    void update_peak_memory();
    bool setup_opencl_kernels();
};

} // namespace zen3
