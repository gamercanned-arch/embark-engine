// apu_flux2_orchestrator.h
// Complete Dual-Engine APU Orchestrator for FLUX.2 Klein 4B DiT on AMD Ryzen 7 7730U
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
#include "flux2_tiler.h"
#include "flux2_threadpool.h"
#include "flux2_rope_lut.h"
#include "opencl/vega8_context.h"

namespace flux2 {

struct GenerationResult {
    double text_latency_ms = 0.0;
    double rope_init_ms = 0.0;
    double dit_latency_ms = 0.0;
    double dit_step_latency_ms[4] = {0.0, 0.0, 0.0, 0.0};
    double vae_latency_ms = 0.0;
    double total_wallclock_sec = 0.0;
    double speedup_vs_baseline = 0.0;
    double peak_working_set_gb = 0.0;
    double memory_bandwidth_gb_s = 0.0;
    bool success = false;
};

class APUFlux2Orchestrator {
public:
    APUFlux2Orchestrator(
        const std::string& dit_path,
        const std::string& text_enc_path,
        const std::string& vae_path
    );
    ~APUFlux2Orchestrator();

    bool initialize();

    // Benchmarking phases
    double benchmark_text_conditioning(int iters = 3);
    double benchmark_dit_denoising(int steps = 4, int iters = 3);
    double benchmark_vae_decode(int iters = 3);

    // End-to-end full generation run
    GenerationResult run_generation(const std::string& prompt, int steps = 4);

    double get_peak_working_set_gb() const;

private:
    std::string m_dit_path;
    std::string m_text_enc_path;
    std::string m_vae_path;
    bool m_initialized = false;
    double m_peak_ws_gb = 0.0;

    // Subsystems
    Flux2Tiler m_tiler;
    std::unique_ptr<Flux2ThreadPool> m_threadpool;
    RoPE3DTable m_rope_lut;
    std::unique_ptr<Vega8Context> m_vega8;

    // OpenCL program & kernels
    cl_program m_gpu_program = nullptr;
    cl_kernel  m_kernel_flash_attn = nullptr;
    cl_kernel  m_kernel_gemm = nullptr;
    cl_kernel  m_kernel_adaln_pre_norm = nullptr;
    cl_kernel  m_kernel_adaln_gate = nullptr;
    cl_kernel  m_kernel_vae_blend = nullptr;

    // Zero-Copy OpenCL Buffers
    cl_mem m_gpu_dit_weights = nullptr;
    cl_mem m_gpu_text_weights = nullptr;
    cl_mem m_gpu_vae_weights = nullptr;
    cl_mem m_gpu_rope_cos = nullptr;
    cl_mem m_gpu_rope_sin = nullptr;
    cl_mem m_gpu_latent_a = nullptr;
    cl_mem m_gpu_latent_b = nullptr;
    cl_mem m_gpu_text_tokens = nullptr;

    // Working activation and attention buffers
    cl_mem m_cl_attn_q = nullptr;
    cl_mem m_cl_attn_k = nullptr;
    cl_mem m_cl_attn_v = nullptr;
    cl_mem m_cl_attn_out = nullptr;
    cl_mem m_cl_gemm_a = nullptr;
    cl_mem m_cl_gemm_b = nullptr;
    cl_mem m_cl_gemm_c = nullptr;
    cl_mem m_cl_tile_rgb = nullptr;
    cl_mem m_cl_canvas_rgb = nullptr;
    cl_mem m_cl_canvas_w = nullptr;

    bool setup_opencl();
    void update_peak_memory();
};

} // namespace flux2
