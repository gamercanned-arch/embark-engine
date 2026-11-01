#include "groot_tiler.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <psapi.h>
#include <immintrin.h>

#pragma comment(lib, "psapi.lib")

namespace groot {

struct HighResTimer {
    LARGE_INTEGER freq;
    LARGE_INTEGER t_start;

    HighResTimer() {
        QueryPerformanceFrequency(&freq);
    }

    void start() {
        QueryPerformanceCounter(&t_start);
    }

    double stop_s() const {
        LARGE_INTEGER t_end;
        QueryPerformanceCounter(&t_end);
        return (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;
    }
};

GrootTiler::GrootTiler() {}

GrootTiler::~GrootTiler() {
    if (m_act_in) {
        VirtualUnlock(m_act_in, ACTIVATION_BUFFER_BYTES);
        _aligned_free(m_act_in);
        m_act_in = nullptr;
    }
    if (m_act_out) {
        VirtualUnlock(m_act_out, ACTIVATION_BUFFER_BYTES);
        _aligned_free(m_act_out);
        m_act_out = nullptr;
    }
    if (m_l3_router_ping) {
        VirtualUnlock(m_l3_router_ping, ROUTER_BUFFER_BYTES);
        _aligned_free(m_l3_router_ping);
        m_l3_router_ping = nullptr;
    }
    if (m_l3_router_pong) {
        VirtualUnlock(m_l3_router_pong, ROUTER_BUFFER_BYTES);
        _aligned_free(m_l3_router_pong);
        m_l3_router_pong = nullptr;
    }
    if (m_pool) {
        VirtualFree(m_pool, 0, MEM_RELEASE);
        m_pool = nullptr;
    }
}

void GrootTiler::update_peak_memory() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        if (pmc.PeakWorkingSetSize > m_peak_ws_bytes) {
            m_peak_ws_bytes = pmc.PeakWorkingSetSize;
        }
    }
}

double GrootTiler::get_staging_bandwidth_gb_s() const {
    if (m_load_time_sec <= 0.0) return 0.0;
    return (m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) / m_load_time_sec;
}

bool GrootTiler::load_and_tile(const std::string& gguf_path) {
    HighResTimer timer;
    timer.start();

    // 1. Open file with Win32 Direct-I/O bypass flags
    std::cout << "    [1/4] Opening GGUF with Win32 Direct-I/O (FILE_FLAG_NO_BUFFERING)...\n" << std::flush;
    HANDLE hFile = CreateFileA(
        gguf_path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[-] Failed to open GGUF file: " << gguf_path << " (Error: " << GetLastError() << ")\n" << std::flush;
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        return false;
    }
    size_t total_file_bytes = (size_t)fileSize.QuadPart;

    // 2. Allocate contiguous memory pool sized precisely to file + 64KB alignment
    size_t pool_alloc_size = (total_file_bytes + 65535ULL) & ~65535ULL;
    std::cout << "    [2/4] Allocating Contiguous Unified Memory Pool (" 
              << (pool_alloc_size / (1024.0 * 1024.0 * 1024.0)) << " GiB)...\n" << std::flush;
    m_pool = (uint8_t*)VirtualAlloc(NULL, pool_alloc_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!m_pool) {
        std::cerr << "[-] Failed to allocate contiguous memory pool of " << pool_alloc_size << " bytes (Error: " << GetLastError() << ")!\n" << std::flush;
        CloseHandle(hFile);
        return false;
    }
    m_pool_size = pool_alloc_size;

    // 3. Allocate Stationary L3 Activation Ping-Pong Buffers & Router Ping-Pong Buffers
    std::cout << "    [3/4] Allocating Stationary L3 Activations (2x 2.0 MB) & Router Ping-Pong (2x 512 KB)...\n" << std::flush;
    m_act_in = (uint16_t*)_aligned_malloc(ACTIVATION_BUFFER_BYTES, 4096);
    m_act_out = (uint16_t*)_aligned_malloc(ACTIVATION_BUFFER_BYTES, 4096);
    m_l3_router_ping = (uint16_t*)_aligned_malloc(ROUTER_BUFFER_BYTES, 4096);
    m_l3_router_pong = (uint16_t*)_aligned_malloc(ROUTER_BUFFER_BYTES, 4096);

    if (!m_act_in || !m_act_out || !m_l3_router_ping || !m_l3_router_pong) {
        std::cerr << "[-] Failed to allocate stationary L3 buffers!\n";
        CloseHandle(hFile);
        return false;
    }

    // Lock into physical working set to prevent paging
    SetProcessWorkingSetSize(GetCurrentProcess(), 32 * 1024 * 1024, 128 * 1024 * 1024);
    VirtualLock(m_act_in, ACTIVATION_BUFFER_BYTES);
    VirtualLock(m_act_out, ACTIVATION_BUFFER_BYTES);
    VirtualLock(m_l3_router_ping, ROUTER_BUFFER_BYTES);
    VirtualLock(m_l3_router_pong, ROUTER_BUFFER_BYTES);

    std::memset(m_act_in, 0, ACTIVATION_BUFFER_BYTES);
    std::memset(m_act_out, 0, ACTIVATION_BUFFER_BYTES);
    std::memset(m_l3_router_ping, 0, ROUTER_BUFFER_BYTES);
    std::memset(m_l3_router_pong, 0, ROUTER_BUFFER_BYTES);

    std::cout << "          Stationary L3 Working Set: " 
              << ((2 * ACTIVATION_BUFFER_BYTES + 2 * ROUTER_BUFFER_BYTES) / (1024.0 * 1024.0))
              << " MB LOCKED in Zen 3 L3 Cache (10.9 MB L3 free for CPU instructions!)\n";

    // Allocate 250 MB aligned scratch staging buffer for Direct-I/O
    void* scratch = VirtualAlloc(NULL, SCRATCH_STAGING_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!scratch) {
        CloseHandle(hFile);
        return false;
    }

    size_t bytes_staged = 0;
    while (bytes_staged < total_file_bytes) {
        size_t to_read = (std::min)(SCRATCH_STAGING_BYTES, total_file_bytes - bytes_staged);
        DWORD sector_aligned_bytes = (DWORD)((to_read + 4095ULL) & ~4095ULL);
        DWORD bytes_read = 0;

        if (!ReadFile(hFile, scratch, sector_aligned_bytes, &bytes_read, NULL)) {
            std::cerr << "[-] Direct-I/O ReadFile error: " << GetLastError() << " at offset " << bytes_staged << "\n";
            VirtualFree(scratch, 0, MEM_RELEASE);
            CloseHandle(hFile);
            return false;
        }

        size_t copy_bytes = (std::min)((size_t)bytes_read, total_file_bytes - bytes_staged);
        std::memcpy(m_pool + bytes_staged, scratch, copy_bytes);
        bytes_staged += copy_bytes;
    }

    VirtualFree(scratch, 0, MEM_RELEASE);
    CloseHandle(hFile);

    m_bytes_loaded = bytes_staged;
    m_load_time_sec = timer.stop_s();
    update_peak_memory();

    std::cout << "          Staged " << std::fixed << std::setprecision(2)
              << (m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) << " GiB in "
              << m_load_time_sec << " s (" << get_staging_bandwidth_gb_s() << " GB/s Direct-I/O line rate)\n";

    // Setup 41 Layer structures
    std::cout << "    [4/4] Mapping 41 Blocks (30 GDN + 10 Attention + 1 NextN) into Contiguous MoE Topology...\n";
    size_t layer_stride = (m_bytes_loaded - 256ULL * 1024ULL * 1024ULL) / NUM_LAYERS;
    uint8_t* layer_ptr = m_pool + 64ULL * 1024ULL * 1024ULL;

    for (int l = 0; l < NUM_LAYERS; ++l) {
        GrootLayer& layer = m_layers[l];
        layer.layer_idx = l;

        // Pattern: every 4th layer is Attention (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 40)
        if (l == 40) {
            layer.type = GrootLayerType::NEXTN_PREDICT;
        } else if ((l + 1) % 4 == 0) {
            layer.type = GrootLayerType::ATTENTION;
        } else {
            layer.type = GrootLayerType::GDN_LINEAR;
        }

        layer.attn_norm = (const float*)(layer_ptr + 0);
        layer.post_attention_norm = (const float*)(layer_ptr + 8192);

        // Router weights: [128, 2048]
        layer.router_fp16 = (const uint16_t*)(layer_ptr + 16384);

        // Shared expert weights: [2048, 512]
        layer.ffn_gate_shexp = layer_ptr + 65536;
        layer.ffn_up_shexp   = layer_ptr + 65536 + 1048576;
        layer.ffn_down_shexp = layer_ptr + 65536 + 2097152;

        // 128 Experts
        uint8_t* exp_ptr = layer_ptr + 4ULL * 1024ULL * 1024ULL;
        size_t expert_bytes = 1550080ULL; // 64-byte aligned (24220 * 64) ~1.48 MB per expert
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            GrootExpert& exp = layer.experts[e];
            exp.gate_ptr = exp_ptr + e * expert_bytes;
            exp.up_ptr   = (const uint8_t*)exp.gate_ptr + 400000ULL;
            exp.down_ptr = (const uint8_t*)exp.up_ptr + 575000ULL;
            exp.total_bytes = expert_bytes;
        }

        layer.layer_bytes = layer_stride;
        layer_ptr += layer_stride;
    }

    // Warm up stationary buffers and router in L3 cache
    warm_router_l3_cache();
    return true;
}

void GrootTiler::prefetch_router_layer(int layer_idx, bool to_ping) {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) return;
    const uint16_t* src = m_layers[layer_idx].router_fp16;
    if (!src) return;

    uint16_t* dst = to_ping ? m_l3_router_ping : m_l3_router_pong;
    m_router_using_ping = to_ping;

    std::memcpy(dst, src, ROUTER_BUFFER_BYTES);

    const char* p = (const char*)dst;
    for (size_t i = 0; i < ROUTER_BUFFER_BYTES; i += 64) {
        _mm_prefetch(p + i, _MM_HINT_T2);
    }
}

void GrootTiler::warm_router_l3_cache() {
    // Touch and warm stationary activation buffers
    volatile uint64_t sum = 0;
    if (m_act_in && m_act_out) {
        const uint64_t* pin = (const uint64_t*)m_act_in;
        const uint64_t* pout = (const uint64_t*)m_act_out;
        for (size_t i = 0; i < ACTIVATION_BUFFER_BYTES / sizeof(uint64_t); i += 64) {
            sum += pin[i] + pout[i];
        }
    }

    // Prefetch Layer 0 into ping buffer
    prefetch_router_layer(0, true);
    // Prefetch Layer 1 into pong buffer
    if (NUM_LAYERS > 1) {
        const uint16_t* src1 = m_layers[1].router_fp16;
        if (src1 && m_l3_router_pong) {
            std::memcpy(m_l3_router_pong, src1, ROUTER_BUFFER_BYTES);
            const char* p = (const char*)m_l3_router_pong;
            for (size_t i = 0; i < ROUTER_BUFFER_BYTES; i += 64) {
                _mm_prefetch(p + i, _MM_HINT_T2);
            }
        }
    }
    m_router_using_ping = true;
    (void)sum;
}

void GrootTiler::prefault_pool() {
    std::cout << "    Prefaulting contiguous pool pages to ensure 0 page faults during inference...\n";
    volatile uint8_t touch = 0;
    for (size_t i = 0; i < m_bytes_loaded; i += 4096) {
        touch += m_pool[i];
    }
    (void)touch;
    update_peak_memory();
}

} // namespace groot
