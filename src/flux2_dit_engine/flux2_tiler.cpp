// flux2_tiler.cpp
#include "flux2_tiler.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <psapi.h>

#pragma comment(lib, "psapi.lib")

namespace flux2 {

Flux2Tiler::Flux2Tiler() {
    // Allocate 5.2 GB contiguous page-aligned virtual memory block
    m_unified_pool = (uint8_t*)VirtualAlloc(
        NULL,
        m_pool_capacity,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE
    );
    if (!m_unified_pool) {
        std::cerr << "[-] Failed to allocate 5.2 GB contiguous UMA pool!\\n";
    }
}

Flux2Tiler::~Flux2Tiler() {
    if (m_unified_pool) {
        VirtualFree(m_unified_pool, 0, MEM_RELEASE);
        m_unified_pool = nullptr;
    }
}

void Flux2Tiler::update_peak_memory() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        if (pmc.PeakWorkingSetSize > m_peak_ws_bytes) {
            m_peak_ws_bytes = pmc.PeakWorkingSetSize;
        }
    }
}

bool Flux2Tiler::stage_file_direct_io(const std::string& path, ModelBuffer& out_buf) {
    HANDLE hFile = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[-] Failed to open file: " << path << "\\n";
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        return false;
    }

    size_t file_bytes = (size_t)fileSize.QuadPart;
    // Align to 4096-byte page boundary
    size_t aligned_bytes = (file_bytes + 4095) & ~4095ULL;

    if (m_pool_allocated + aligned_bytes > m_pool_capacity) {
        std::cerr << "[-] Out of UMA memory pool capacity!\\n";
        CloseHandle(hFile);
        return false;
    }

    uint8_t* dst = m_unified_pool + m_pool_allocated;
    m_pool_allocated += aligned_bytes;

    out_buf.ptr = dst;
    out_buf.size = file_bytes;
    out_buf.path = path;

    // Direct chunked read
    constexpr DWORD CHUNK_SIZE = 64 * 1024 * 1024; // 64 MB
    size_t bytes_left = file_bytes;
    uint8_t* cur_dst = dst;

    while (bytes_left > 0) {
        DWORD to_read = (DWORD)(std::min)((size_t)CHUNK_SIZE, bytes_left);
        DWORD bytes_read = 0;
        if (!ReadFile(hFile, cur_dst, to_read, &bytes_read, NULL) || bytes_read == 0) {
            std::cerr << "[-] Direct ReadFile error on: " << path << "\\n";
            CloseHandle(hFile);
            return false;
        }
        cur_dst += bytes_read;
        bytes_left -= bytes_read;
        m_bytes_loaded += bytes_read;
    }

    CloseHandle(hFile);
    update_peak_memory();
    return true;
}

bool Flux2Tiler::load_and_stage(
    const std::string& dit_path,
    const std::string& text_enc_path,
    const std::string& vae_path,
    Flux2ThreadPool* threadpool)
{
    if (!m_unified_pool) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "  -> Staging Text Encoder: " << text_enc_path << "\\n";
    if (!stage_file_direct_io(text_enc_path, m_text_buf)) return false;

    std::cout << "  -> Staging FLUX.2 Klein DiT: " << dit_path << "\\n";
    if (!stage_file_direct_io(dit_path, m_dit_buf)) return false;

    std::cout << "  -> Staging FLUX.2 VAE: " << vae_path << "\\n";
    if (!stage_file_direct_io(vae_path, m_vae_buf)) return false;

    auto t1 = std::chrono::high_resolution_clock::now();
    m_load_time_sec = std::chrono::duration<double>(t1 - t0).count();

    // Pre-fault all pages across 8 Zen 3 cores to avoid soft page faults
    if (threadpool) {
        std::cout << "  -> Pre-faulting 4.82 GB static memory across 8 Zen 3 cores...\\n";
        threadpool->stream_contiguous(m_unified_pool, m_pool_allocated);
    }

    update_peak_memory();
    return true;
}

double Flux2Tiler::get_staging_bandwidth_gb_s() const {
    if (m_load_time_sec <= 0.0) return 0.0;
    return ((double)m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) / m_load_time_sec;
}

} // namespace flux2
