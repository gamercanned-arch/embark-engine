// flux2_tiler.h
// High-Throughput Direct-I/O Staging & Persistent Zero-Copy UMA Buffer Manager for FLUX.2 Klein 4B
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
#include <cstdint>
#include "flux2_threadpool.h"

namespace flux2 {

constexpr size_t STATIC_POOL_SIZE = 5200000000ULL; // ~5.2 GB contiguous UMA pool

struct ModelBuffer {
    uint8_t* ptr = nullptr;
    size_t size = 0;
    std::string path;
};

class Flux2Tiler {
public:
    Flux2Tiler();
    ~Flux2Tiler();

    Flux2Tiler(const Flux2Tiler&) = delete;
    Flux2Tiler& operator=(const Flux2Tiler&) = delete;

    // Load and stage all 3 models simultaneously into persistent UMA pool:
    // 1. Qwen3-4B Text Encoder (2.36 GB)
    // 2. FLUX.2 Klein 4B DiT (2.12 GB)
    // 3. FLUX.2 VAE Decoder (0.34 GB)
    bool load_and_stage(
        const std::string& dit_path,
        const std::string& text_enc_path,
        const std::string& vae_path,
        Flux2ThreadPool* threadpool = nullptr
    );

    // Buffer accessors
    const uint8_t* get_dit_pool() const { return m_dit_buf.ptr; }
    size_t get_dit_size() const { return m_dit_buf.size; }

    const uint8_t* get_text_enc_pool() const { return m_text_buf.ptr; }
    size_t get_text_enc_size() const { return m_text_buf.size; }

    const uint8_t* get_vae_pool() const { return m_vae_buf.ptr; }
    size_t get_vae_size() const { return m_vae_buf.size; }

    double get_load_time_sec() const { return m_load_time_sec; }
    double get_peak_ws_gb() const { return m_peak_ws_bytes / (1024.0 * 1024.0 * 1024.0); }
    double get_staging_bandwidth_gb_s() const;

private:
    uint8_t* m_unified_pool = nullptr;
    size_t m_pool_capacity = STATIC_POOL_SIZE;
    size_t m_pool_allocated = 0;

    ModelBuffer m_dit_buf;
    ModelBuffer m_text_buf;
    ModelBuffer m_vae_buf;

    double m_load_time_sec = 0.0;
    size_t m_bytes_loaded = 0;
    size_t m_peak_ws_bytes = 0;

    bool stage_file_direct_io(const std::string& path, ModelBuffer& out_buf);
    void update_peak_memory();
};

} // namespace flux2
