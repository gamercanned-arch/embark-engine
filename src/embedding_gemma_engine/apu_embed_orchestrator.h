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
#include "embed_tiler.h"
#include "embed_threadpool.h"
#include "opencl/vega8_context.h"

namespace embed_gemma {

struct TextEncodingResult {
    int seq_len = 0;
    double latency_ms = 0.0;
    double tokens_per_sec = 0.0;
    double effective_bandwidth_gb_s = 0.0;
    double peak_working_set_gb = 0.0;
    double speedup_vs_baseline = 0.0;
};

struct VisionEncodingResult {
    int num_patches = 196;
    double latency_ms = 0.0;
    double sustained_tflops = 0.0;
    double images_per_sec = 0.0;
};

struct MultimodalEmbeddingResult {
    int text_tokens = 0;
    int vision_tokens = 196;
    int total_tokens = 0;
    double vision_latency_ms = 0.0;
    double text_latency_ms = 0.0;
    double total_latency_ms = 0.0;
    double end_to_end_tokens_per_sec = 0.0;
    double peak_working_set_gb = 0.0;
};

struct LayerScratchBuffers {
    std::vector<float> normed_inp;
    std::vector<block_q8_K> q8_inp;
    std::vector<float> Q_buf;
    std::vector<float> K_buf;
    std::vector<float> V_buf;
    std::vector<float> attn_ctx;
    std::vector<block_q8_K> q8_ctx;
    std::vector<float> attn_out;
    std::vector<float> normed_ffn;
    std::vector<block_q8_K> q8_ffn;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_up;
    std::vector<float> ffn_act;
    std::vector<block_q8_K> q8_ffn_act;
    std::vector<float> ffn_down;
    std::vector<float> ple_gate;
    std::vector<float> ple_act;
    std::vector<float> ple_proj;
};

class APUEmbedOrchestrator {
public:
    APUEmbedOrchestrator(const std::string& text_gguf_path, const std::string& mmproj_path = "");
    ~APUEmbedOrchestrator();

    APUEmbedOrchestrator(const APUEmbedOrchestrator&) = delete;
    APUEmbedOrchestrator& operator=(const APUEmbedOrchestrator&) = delete;

    bool initialize();

    TextEncodingResult benchmark_text_encoding(int seq_len, int iters = 5);
    VisionEncodingResult benchmark_vision_encoding(int iters = 5);
    MultimodalEmbeddingResult benchmark_multimodal_embedding(int text_tokens = 512, int iters = 5);

    double get_peak_working_set_gb() const;

private:
    std::string m_text_gguf_path;
    std::string m_mmproj_path;
    bool m_initialized = false;
    double m_peak_ws_gb = 0.0;

    // Subsystems
    EmbedTiler m_tiler;
    std::unique_ptr<EmbedThreadPool> m_threadpool;
    std::unique_ptr<Vega8Context> m_vega8;
    LayerScratchBuffers m_scratch;

    // OpenCL Handles
    cl_program m_gpu_program = nullptr;
    cl_kernel  m_kernel_gemm_fp16 = nullptr;
    cl_kernel  m_kernel_gemm_fp32 = nullptr;
    cl_kernel  m_kernel_vision_proj = nullptr;
    cl_kernel  m_kernel_mean_pool = nullptr;

    void update_peak_memory();
    bool setup_opencl();
    void allocate_scratch_buffers(int max_seq_len = 2048);
    void run_encoder_layer_zen3(int layer_idx, const float* inp_tokens, float* out_tokens, int seq_len, const float* inp_per_layer);
};

} // namespace embed_gemma
