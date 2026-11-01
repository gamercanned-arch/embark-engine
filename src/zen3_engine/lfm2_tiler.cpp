#include "lfm2_tiler.h"
#include "moe_layer_offsets.h"
#include "moe_layer_offsets_q3.h"

#include <psapi.h>
#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>

#pragma comment(lib, "psapi.lib")

namespace zen3 {

static bool seek_file(HANDLE hFile, uint64_t offset) {
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)offset;
    return SetFilePointerEx(hFile, li, NULL, FILE_BEGIN) != FALSE;
}

static bool read_exact(HANDLE hFile, void* buffer, size_t bytes) {
    uint8_t* p = (uint8_t*)buffer;
    size_t total_read = 0;
    while (total_read < bytes) {
        DWORD to_read = (DWORD)std::min<size_t>(bytes - total_read, 64ULL * 1024ULL * 1024ULL);
        DWORD bytes_read = 0;
        if (!ReadFile(hFile, p + total_read, to_read, &bytes_read, NULL) || bytes_read == 0) {
            return false;
        }
        total_read += bytes_read;
    }
    return true;
}

LFM2Tiler::LFM2Tiler() {
    m_expert_pool_size = TOTAL_EXPERT_POOL_BYTES;
    m_router_pool_size = TOTAL_ROUTER_POOL_BYTES;
    m_scratch_buf_size = SCRATCH_BUFFER_BYTES;
}

LFM2Tiler::~LFM2Tiler() {
    if (m_scratch_buf) {
        VirtualFree(m_scratch_buf, 0, MEM_RELEASE);
        m_scratch_buf = nullptr;
    }
    if (m_expert_pool) {
        VirtualFree(m_expert_pool, 0, MEM_RELEASE);
        m_expert_pool = nullptr;
    }
    if (m_router_fp16_pool) {
        _aligned_free(m_router_fp16_pool);
        m_router_fp16_pool = nullptr;
    }
}

void LFM2Tiler::update_peak_working_set() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        if (pmc.PeakWorkingSetSize > m_peak_ws_during_load) {
            m_peak_ws_during_load = pmc.PeakWorkingSetSize;
        }
    }
}

void LFM2Tiler::warm_router_l3_cache() {
    if (!m_router_fp16_pool) return;
    const char* p = (const char*)m_router_fp16_pool;
    for (size_t i = 0; i < m_router_pool_size; i += 64) {
        _mm_prefetch(p + i, _MM_HINT_T0);
    }
}

void LFM2Tiler::prefault_pool() {
    if (!m_expert_pool) return;
    volatile uint64_t s = 0;
    const uint64_t* p = (const uint64_t*)m_expert_pool;
    size_t count = m_expert_pool_size / sizeof(uint64_t);
    for (size_t i = 0; i < count; i += 512) { // 4 KB page stride
        s += p[i];
    }
}

bool LFM2Tiler::load_and_tile(const std::string& gguf_path) {
    std::cout << "================================================================================\n";
    std::cout << "  Zen 3 Contiguous MoE Tiler & Staging Loader\n";
    std::cout << "  File: " << gguf_path << "\n";
    std::cout << "================================================================================\n\n";

    bool is_q3 = (gguf_path.find("Q3") != std::string::npos);
    if (is_q3) {
        m_expert_pool_size = Q3_TOTAL_EXPERT_POOL_BYTES;
        m_scratch_buf_size = SCRATCH_BUFFER_BYTES;
    } else {
        m_expert_pool_size = TOTAL_EXPERT_POOL_BYTES;
        m_scratch_buf_size = 250ULL * 1024ULL * 1024ULL;
    }

    // 1. Open GGUF file with Win32 FILE_FLAG_SEQUENTIAL_SCAN
    HANDLE hFile = CreateFileA(
        gguf_path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[-] Error opening GGUF file: " << GetLastError() << "\n";
        return false;
    }

    // 2. Allocate the contiguous target pool once via VirtualAlloc aligned to 64 KB
    double pool_gb = (double)m_expert_pool_size / (1024.0 * 1024.0 * 1024.0);
    std::cout << "[1/4] Allocating Contiguous Target Pool (" << std::fixed << std::setprecision(2) << pool_gb 
              << " GB / " << m_expert_pool_size << " bytes) via VirtualAlloc...\n";
    m_expert_pool = (uint8_t*)VirtualAlloc(NULL, m_expert_pool_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_expert_pool) {
        std::cerr << "[-] Failed to allocate expert pool: " << GetLastError() << "\n";
        CloseHandle(hFile);
        return false;
    }
    std::cout << "      Contiguous Expert Pool base: 0x" << (void*)m_expert_pool 
              << " (Aligned to 64 KB)\n";

    // 3. Allocate reusable scratch buffer
    std::cout << "[2/4] Allocating Reusable Scratch Buffer (" << (m_scratch_buf_size / (1024 * 1024)) << " MB / " 
              << m_scratch_buf_size << " bytes)...\n";
    m_scratch_buf = (uint8_t*)VirtualAlloc(NULL, m_scratch_buf_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_scratch_buf) {
        std::cerr << "[-] Failed to allocate scratch buffer: " << GetLastError() << "\n";
        CloseHandle(hFile);
        return false;
    }

    // 4. Allocate 9.73 MB FP16 router buffer marked for L3 cache pinning
    std::cout << "[3/4] Allocating FP16 Router Buffer (9.73 MB / " 
              << m_router_pool_size << " bytes, 64-byte aligned for L3 cache)...\n";
    m_router_fp16_pool = (uint16_t*)_aligned_malloc(m_router_pool_size, 64);
    if (!m_router_fp16_pool) {
        std::cerr << "[-] Failed to allocate router FP16 buffer\n";
        CloseHandle(hFile);
        return false;
    }

    m_layers.resize(N_MOE_LAYERS);
    size_t current_pool_offset = 0;

    std::cout << "[4/4] Streaming and Slicing 38 MoE Layers into Contiguous Format...\n";
    if (is_q3) {
        std::cout << "      [Gate (1.15 MB) | Up (1.15 MB) | Down (1.59 MB)] -> 3.89 MB per expert\n\n";
    } else {
        std::cout << "      [Gate (984 KB) | Up (984 KB) | Down (1320 KB)] -> 3.21 MB per expert\n\n";
    }

    alignas(64) float router_f32_buf[ROUTER_LAYER_FLOATS];

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    for (int l = 0; l < N_MOE_LAYERS; ++l) {
        const MoELayerMeta& meta = is_q3 ? MOE_LAYERS_META_Q3[l] : MOE_LAYERS_META[l];
        PackedMoELayer& layer = m_layers[l];
        layer.model_layer_idx = meta.layer_idx;
        layer.moe_idx = l;
        layer.layer_pool_offset = current_pool_offset;

        if (is_q3) {
            bool is_hp = (meta.layer_idx == 7 || meta.layer_idx == 8);
            size_t exp_gate_bytes = is_hp ? Q3_HP_GATE_BYTES : Q3_GATE_BYTES;
            size_t exp_up_bytes   = is_hp ? Q3_HP_UP_BYTES   : Q3_UP_BYTES;
            size_t exp_down_bytes = is_hp ? Q3_HP_DOWN_BYTES : Q3_DOWN_BYTES;
            size_t exp_total      = exp_gate_bytes + exp_up_bytes + exp_down_bytes;
            size_t layer_bytes    = N_EXPERTS * exp_total;

            layer.expert_bytes = exp_total;
            layer.layer_total_bytes = layer_bytes;
            layer.router_fp16 = m_router_fp16_pool + l * ROUTER_LAYER_FLOATS;

            // --- Step A: Read and Convert Router Matrix ---
            if (!seek_file(hFile, meta.gate_inp.file_offset) || !read_exact(hFile, router_f32_buf, meta.gate_inp.size_bytes)) {
                std::cerr << "[-] Read error for layer " << meta.layer_idx << " router\n";
                CloseHandle(hFile);
                return false;
            }

            uint16_t* dst_router = m_router_fp16_pool + l * ROUTER_LAYER_FLOATS;
            for (size_t i = 0; i < ROUTER_LAYER_FLOATS; i += 8) {
                __m256 v = _mm256_loadu_ps(router_f32_buf + i);
                __m128i h = _mm256_cvtps_ph(v, 0);
                _mm_storeu_si128((__m128i*)(dst_router + i), h);
            }

            // --- Step B: Read Expert Weights into Scratch Buffer ---
            uint8_t* down_scratch = m_scratch_buf;
            uint8_t* gate_scratch = down_scratch + meta.down_exps.size_bytes;
            uint8_t* up_scratch   = gate_scratch + meta.gate_exps.size_bytes;

            if (!seek_file(hFile, meta.down_exps.file_offset) || !read_exact(hFile, down_scratch, meta.down_exps.size_bytes) ||
                !seek_file(hFile, meta.gate_exps.file_offset) || !read_exact(hFile, gate_scratch, meta.gate_exps.size_bytes) ||
                !seek_file(hFile, meta.up_exps.file_offset) || !read_exact(hFile, up_scratch, meta.up_exps.size_bytes)) {
                std::cerr << "[-] Read error for Q3 layer " << meta.layer_idx << " experts\n";
                CloseHandle(hFile);
                return false;
            }

            // Slices and packs each expert consecutively
            for (int e = 0; e < N_EXPERTS; ++e) {
                uint8_t* target_exp = m_expert_pool + current_pool_offset + e * exp_total;

                std::memcpy(target_exp, gate_scratch + e * exp_gate_bytes, exp_gate_bytes);
                std::memcpy(target_exp + exp_gate_bytes, up_scratch + e * exp_up_bytes, exp_up_bytes);
                std::memcpy(target_exp + exp_gate_bytes + exp_up_bytes, down_scratch + e * exp_down_bytes, exp_down_bytes);

                PackedExpert& pe = layer.experts[e];
                pe.expert_base = target_exp;
                pe.gate_ptr    = target_exp;
                pe.up_ptr      = target_exp + exp_gate_bytes;
                pe.down_ptr    = target_exp + exp_gate_bytes + exp_up_bytes;
                pe.gate_bytes  = exp_gate_bytes;
                pe.up_bytes    = exp_up_bytes;
                pe.down_bytes  = exp_down_bytes;
                pe.total_bytes = exp_total;
                pe.gate_type   = is_hp ? QuantType::IQ4_XS : QuantType::IQ3_XXS;
                pe.up_type     = is_hp ? QuantType::IQ4_XS : QuantType::IQ3_XXS;
                pe.down_type   = is_hp ? QuantType::Q6_K : QuantType::IQ4_XS;
            }

            current_pool_offset += layer_bytes;
        } else {
            bool is_high_precision = (meta.layer_idx == 7 || meta.layer_idx == 8);
            size_t exp_gate_bytes = is_high_precision ? HP_GATE_BYTES : STD_GATE_BYTES;
            size_t exp_up_bytes   = is_high_precision ? HP_UP_BYTES   : STD_UP_BYTES;
            size_t exp_down_bytes = is_high_precision ? HP_DOWN_BYTES : STD_DOWN_BYTES;
            size_t exp_total      = exp_gate_bytes + exp_up_bytes + exp_down_bytes;
            size_t layer_bytes    = N_EXPERTS * exp_total;

            layer.expert_bytes = exp_total;
            layer.layer_total_bytes = layer_bytes;
            layer.router_fp16 = m_router_fp16_pool + l * ROUTER_LAYER_FLOATS;

            // --- Step A: Read and Convert Router Matrix ---
            if (!seek_file(hFile, meta.gate_inp.file_offset)) {
                std::cerr << "[-] Seek error for layer " << meta.layer_idx << " router\n";
                CloseHandle(hFile);
                return false;
            }
            if (!read_exact(hFile, router_f32_buf, meta.gate_inp.size_bytes)) {
                std::cerr << "[-] Read error for layer " << meta.layer_idx << " router\n";
                CloseHandle(hFile);
                return false;
            }

            uint16_t* dst_router = m_router_fp16_pool + l * ROUTER_LAYER_FLOATS;
            for (size_t i = 0; i < ROUTER_LAYER_FLOATS; i += 8) {
                __m256 v = _mm256_loadu_ps(router_f32_buf + i);
                __m128i h = _mm256_cvtps_ph(v, 0);
                _mm_storeu_si128((__m128i*)(dst_router + i), h);
            }

            // --- Step B: Read Expert Weights into 250 MB Scratch Buffer ---
            if (!is_high_precision) {
                uint8_t* down_scratch = m_scratch_buf;
                uint8_t* gate_scratch = down_scratch + meta.down_exps.size_bytes;
                uint8_t* up_scratch   = gate_scratch + meta.gate_exps.size_bytes;

                if (!seek_file(hFile, meta.down_exps.file_offset) ||
                    !read_exact(hFile, down_scratch, meta.down_exps.size_bytes)) {
                    std::cerr << "[-] Read error for layer " << meta.layer_idx << " down_exps\n";
                    CloseHandle(hFile);
                    return false;
                }
                if (!seek_file(hFile, meta.gate_exps.file_offset) ||
                    !read_exact(hFile, gate_scratch, meta.gate_exps.size_bytes)) {
                    std::cerr << "[-] Read error for layer " << meta.layer_idx << " gate_exps\n";
                    CloseHandle(hFile);
                    return false;
                }
                if (!seek_file(hFile, meta.up_exps.file_offset) ||
                    !read_exact(hFile, up_scratch, meta.up_exps.size_bytes)) {
                    std::cerr << "[-] Read error for layer " << meta.layer_idx << " up_exps\n";
                    CloseHandle(hFile);
                    return false;
                }

                for (int e = 0; e < N_EXPERTS; ++e) {
                    uint8_t* target_exp = m_expert_pool + current_pool_offset + e * exp_total;

                    std::memcpy(target_exp, gate_scratch + e * exp_gate_bytes, exp_gate_bytes);
                    std::memcpy(target_exp + exp_gate_bytes, up_scratch + e * exp_up_bytes, exp_up_bytes);
                    std::memcpy(target_exp + exp_gate_bytes + exp_up_bytes, down_scratch + e * exp_down_bytes, exp_down_bytes);

                    PackedExpert& pe = layer.experts[e];
                    pe.expert_base = target_exp;
                    pe.gate_ptr    = target_exp;
                    pe.up_ptr      = target_exp + exp_gate_bytes;
                    pe.down_ptr    = target_exp + exp_gate_bytes + exp_up_bytes;
                    pe.gate_bytes  = exp_gate_bytes;
                    pe.up_bytes    = exp_up_bytes;
                    pe.down_bytes  = exp_down_bytes;
                    pe.total_bytes = exp_total;
                    pe.gate_type   = QuantType::IQ2_S;
                    pe.up_type     = QuantType::IQ2_S;
                    pe.down_type   = QuantType::IQ3_S;
                }
            } else {
                uint8_t* scratch_down = m_scratch_buf;
                if (!seek_file(hFile, meta.down_exps.file_offset) ||
                    !read_exact(hFile, scratch_down, meta.down_exps.size_bytes)) {
                    std::cerr << "[-] Read error for HP layer " << meta.layer_idx << " down_exps\n";
                    CloseHandle(hFile);
                    return false;
                }
                for (int e = 0; e < N_EXPERTS; ++e) {
                    uint8_t* target_exp = m_expert_pool + current_pool_offset + e * exp_total;
                    std::memcpy(target_exp + exp_gate_bytes + exp_up_bytes, scratch_down + e * exp_down_bytes, exp_down_bytes);
                }

                uint8_t* scratch_gate = m_scratch_buf;
                uint8_t* scratch_up   = scratch_gate + meta.gate_exps.size_bytes;
                if (!seek_file(hFile, meta.gate_exps.file_offset) ||
                    !read_exact(hFile, scratch_gate, meta.gate_exps.size_bytes)) {
                    std::cerr << "[-] Read error for HP layer " << meta.layer_idx << " gate_exps\n";
                    CloseHandle(hFile);
                    return false;
                }
                if (!seek_file(hFile, meta.up_exps.file_offset) ||
                    !read_exact(hFile, scratch_up, meta.up_exps.size_bytes)) {
                    std::cerr << "[-] Read error for HP layer " << meta.layer_idx << " up_exps\n";
                    CloseHandle(hFile);
                    return false;
                }

                for (int e = 0; e < N_EXPERTS; ++e) {
                    uint8_t* target_exp = m_expert_pool + current_pool_offset + e * exp_total;
                    std::memcpy(target_exp, scratch_gate + e * exp_gate_bytes, exp_gate_bytes);
                    std::memcpy(target_exp + exp_gate_bytes, scratch_up + e * exp_up_bytes, exp_up_bytes);

                    PackedExpert& pe = layer.experts[e];
                    pe.expert_base = target_exp;
                    pe.gate_ptr    = target_exp;
                    pe.up_ptr      = target_exp + exp_gate_bytes;
                    pe.down_ptr    = target_exp + exp_gate_bytes + exp_up_bytes;
                    pe.gate_bytes  = exp_gate_bytes;
                    pe.up_bytes    = exp_up_bytes;
                    pe.down_bytes  = exp_down_bytes;
                    pe.total_bytes = exp_total;
                    pe.gate_type   = QuantType::IQ3_S;
                    pe.up_type     = QuantType::IQ3_S;
                    pe.down_type   = QuantType::IQ4_XS;
                }
            }
            current_pool_offset += layer_bytes;
        }

        update_peak_working_set();

        if ((l + 1) % 5 == 0 || (l + 1) == N_MOE_LAYERS) {
            std::cout << "  -> Loaded Layer " << std::setw(2) << (l + 1) << "/" << N_MOE_LAYERS
                      << " (blk." << meta.layer_idx << ") | Pool Committed: "
                      << std::fixed << std::setprecision(2) << ((double)current_pool_offset / (1024.0 * 1024.0 * 1024.0)) << " GB"
                      << " | Peak Working Set: " << get_peak_working_set_gb() << " GB\n";
        }
    }

    QueryPerformanceCounter(&t_end);
    double load_sec = (double)(t_end.QuadPart - t_start.QuadPart) / freq.QuadPart;
    double load_gb = (double)current_pool_offset / (1024.0 * 1024.0 * 1024.0);

    CloseHandle(hFile);

    // Warm router weights into L3 cache
    warm_router_l3_cache();

    std::cout << "\n[+] Chunked Staging & Tiling Complete in " << std::fixed << std::setprecision(2) << load_sec << " s ("
              << (load_gb / load_sec) << " GB/s effective I/O rate)\n";
    std::cout << "    Total Expert Pool Committed: " << load_gb << " GB\n";
    std::cout << "    Peak Loading Working Set:   " << get_peak_working_set_gb() << " GB\n\n";

    return true;
}

} // namespace zen3
