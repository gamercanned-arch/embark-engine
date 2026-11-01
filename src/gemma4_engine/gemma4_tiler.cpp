#include "gemma4_tiler.h"
#include <psapi.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <string>

#pragma comment(lib, "psapi.lib")

namespace gemma4 {

// Helper: seek file using 64-bit integer
static bool seek_file(HANDLE hFile, uint64_t offset) {
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)offset;
    return SetFilePointerEx(hFile, li, NULL, FILE_BEGIN) != FALSE;
}

// Helper: robust chunked read
static bool read_exact(HANDLE hFile, void* buffer, size_t bytes) {
    uint8_t* p = (uint8_t*)buffer;
    size_t total_read = 0;
    while (total_read < bytes) {
        DWORD to_read = (DWORD)(std::min<size_t>)(bytes - total_read, 64ULL * 1024ULL * 1024ULL);
        DWORD bytes_read = 0;
        if (!ReadFile(hFile, p + total_read, to_read, &bytes_read, NULL) || bytes_read == 0) {
            return false;
        }
        total_read += bytes_read;
    }
    return true;
}

// GGUF parser helpers
static std::string read_gguf_string(HANDLE hFile) {
    uint64_t len = 0;
    DWORD bytes_read = 0;
    if (!ReadFile(hFile, &len, sizeof(len), &bytes_read, NULL) || bytes_read != sizeof(len)) {
        return "";
    }
    if (len == 0) return "";
    std::string s(len, '\0');
    if (!ReadFile(hFile, &s[0], (DWORD)len, &bytes_read, NULL) || bytes_read != len) {
        return "";
    }
    return s;
}

static bool skip_gguf_value(HANDLE hFile, uint32_t val_type) {
    DWORD bytes_read = 0;
    switch (val_type) {
        case 0: // UINT8
        case 1: // INT8
        case 7: { // BOOL
            uint8_t dummy;
            return ReadFile(hFile, &dummy, 1, &bytes_read, NULL) && bytes_read == 1;
        }
        case 2: // UINT16
        case 3: { // INT16
            uint16_t dummy;
            return ReadFile(hFile, &dummy, 2, &bytes_read, NULL) && bytes_read == 2;
        }
        case 4: // UINT32
        case 5: // INT32
        case 6: { // FLOAT32
            uint32_t dummy;
            return ReadFile(hFile, &dummy, 4, &bytes_read, NULL) && bytes_read == 4;
        }
        case 8: { // STRING
            read_gguf_string(hFile);
            return true;
        }
        case 10: // UINT64
        case 11: // INT64
        case 12: { // FLOAT64
            uint64_t dummy;
            return ReadFile(hFile, &dummy, 8, &bytes_read, NULL) && bytes_read == 8;
        }
        case 9: { // ARRAY
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!ReadFile(hFile, &elem_type, 4, &bytes_read, NULL) || bytes_read != 4) return false;
            if (!ReadFile(hFile, &count, 8, &bytes_read, NULL) || bytes_read != 8) return false;
            for (uint64_t i = 0; i < count; ++i) {
                if (!skip_gguf_value(hFile, elem_type)) return false;
            }
            return true;
        }
        default:
            return false;
    }
}

struct ParsedTensor {
    std::string name;
    uint32_t n_dims = 0;
    std::vector<uint64_t> dims;
    uint32_t type = 0; // 0=F32, 12=Q4_K, 14=Q6_K
    uint64_t offset = 0;
    uint64_t raw_bytes = 0;
    uint64_t n_elements = 0;
};

Gemma4Tiler::Gemma4Tiler() {
    m_pool_size = EXPANDED_POOL_SIZE;
    m_scratch_size = SCRATCH_BUFFER_BYTES;
    m_layers.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; ++i) {
        m_layers[i].layer_idx = i;
        m_layers[i].is_global = (i % 6 == 5);
        if (m_layers[i].is_global) {
            m_layers[i].q_dim = 8192;
            m_layers[i].k_dim = 512;
            m_layers[i].v_dim = 0; // TIED TO K (V = K)
            m_layers[i].o_dim = 8192;
        } else {
            m_layers[i].q_dim = 4096;
            m_layers[i].k_dim = 2048;
            m_layers[i].v_dim = 2048;
            m_layers[i].o_dim = 4096;
        }
    }
}

Gemma4Tiler::~Gemma4Tiler() {
    if (m_scratch_buf) {
        VirtualFree(m_scratch_buf, 0, MEM_RELEASE);
        m_scratch_buf = nullptr;
    }
    if (m_pool) {
        VirtualFree(m_pool, 0, MEM_RELEASE);
        m_pool = nullptr;
    }
}

void Gemma4Tiler::update_peak_memory() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        if (pmc.PeakWorkingSetSize > m_peak_ws_bytes) {
            m_peak_ws_bytes = pmc.PeakWorkingSetSize;
        }
    }
}

void Gemma4Tiler::prefault_pool() {
    if (!m_pool) return;
    volatile uint64_t sum = 0;
    const uint64_t* p = (const uint64_t*)m_pool;
    size_t count = m_pool_offset / sizeof(uint64_t);
    for (size_t i = 0; i < count; i += 512) { // 4 KB page stride
        sum += p[i];
    }
}

double Gemma4Tiler::get_staging_bandwidth_gb_s() const {
    if (m_load_time_sec <= 0.0) return 0.0;
    return (m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) / m_load_time_sec;
}

bool Gemma4Tiler::load_and_stage(const std::string& gguf_path) {
    std::cout << "================================================================================\n";
    std::cout << "  Gemma 4 12B Direct-I/O Chunked Staging Loader & Scale Expander\n";
    std::cout << "  Target APU: AMD Ryzen 7 7730U (Barcelo-R Zen 3 + Vega 8 UMA)\n";
    std::cout << "  Model: " << gguf_path << "\n";
    std::cout << "================================================================================\n\n";

    // 1. Open GGUF with Win32 FILE_FLAG_SEQUENTIAL_SCAN for line-rate NVMe streaming
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

    // 2. Allocate contiguous memory pool (7,163.34 MB / 6.99 GiB) via VirtualAlloc aligned to 64 KB
    std::cout << "[1/4] Allocating Contiguous Memory Pool (6.99 GiB / " 
              << (m_pool_size / (1024 * 1024)) << " MB) via VirtualAlloc...\n";
    m_pool = (uint8_t*)VirtualAlloc(NULL, m_pool_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_pool) {
        std::cerr << "[-] Failed to allocate contiguous memory pool: " << GetLastError() << "\n";
        CloseHandle(hFile);
        return false;
    }
    std::cout << "      Contiguous Pool Base: 0x" << (void*)m_pool << " (64 KB Aligned)\n";

    // 3. Allocate 250 MB reusable scratch buffer
    std::cout << "[2/4] Allocating Reusable Staging Scratch Buffer (250 MB / " 
              << (m_scratch_size / (1024 * 1024)) << " MB)...\n";
    m_scratch_buf = (uint8_t*)VirtualAlloc(NULL, m_scratch_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_scratch_buf) {
        std::cerr << "[-] Failed to allocate 250 MB scratch buffer: " << GetLastError() << "\n";
        CloseHandle(hFile);
        return false;
    }

    // 4. Parse GGUF Header
    std::cout << "[3/4] Parsing GGUF Header & Tensor Dictionary (667 Tensors)...\n";
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    DWORD bytes_read = 0;

    ReadFile(hFile, &magic, 4, &bytes_read, NULL);
    ReadFile(hFile, &version, 4, &bytes_read, NULL);
    ReadFile(hFile, &n_tensors, 8, &bytes_read, NULL);
    ReadFile(hFile, &n_kv, 8, &bytes_read, NULL);

    if (magic != 0x46554747) { // "GGUF"
        std::cerr << "[-] Invalid GGUF magic: 0x" << std::hex << magic << "\n";
        CloseHandle(hFile);
        return false;
    }

    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = read_gguf_string(hFile);
        uint32_t val_type = 0;
        ReadFile(hFile, &val_type, 4, &bytes_read, NULL);
        if (key == "general.alignment") {
            ReadFile(hFile, &alignment, 4, &bytes_read, NULL);
        } else {
            skip_gguf_value(hFile, val_type);
        }
    }

    std::vector<ParsedTensor> tensor_list;
    tensor_list.reserve(n_tensors);

    for (uint64_t i = 0; i < n_tensors; ++i) {
        ParsedTensor pt;
        pt.name = read_gguf_string(hFile);
        ReadFile(hFile, &pt.n_dims, 4, &bytes_read, NULL);
        pt.dims.resize(pt.n_dims);
        ReadFile(hFile, pt.dims.data(), (DWORD)(pt.n_dims * sizeof(uint64_t)), &bytes_read, NULL);
        ReadFile(hFile, &pt.type, 4, &bytes_read, NULL);
        ReadFile(hFile, &pt.offset, 8, &bytes_read, NULL);

        pt.n_elements = 1;
        for (auto d : pt.dims) pt.n_elements *= d;

        if (pt.type == 0) { // F32
            pt.raw_bytes = pt.n_elements * 4;
        } else if (pt.type == 12) { // Q4_K
            pt.raw_bytes = (pt.n_elements / QK_K) * 144;
        } else if (pt.type == 14) { // Q6_K
            pt.raw_bytes = (pt.n_elements / QK_K) * 210;
        } else {
            pt.raw_bytes = pt.n_elements * 4;
        }
        tensor_list.push_back(pt);
    }

    LARGE_INTEGER current_pos;
    current_pos.QuadPart = 0;
    SetFilePointerEx(hFile, current_pos, &current_pos, FILE_CURRENT);
    uint64_t data_start = (current_pos.QuadPart + alignment - 1) & ~((uint64_t)alignment - 1);

    std::cout << "      GGUF Header Parsed: " << n_tensors << " tensors, alignment=" 
              << alignment << ", data_offset=" << data_start << "\n";

    // 5. Stage Chunks and Pre-Expand Q4_K Scales (+139 MB)
    std::cout << "[4/4] Staging Tensors via 250 MB Scratch Buffer & Expanding Q4_K Scales...\n";

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    m_pool_offset = 0;
    m_bytes_loaded = 0;

    for (const auto& pt : tensor_list) {
        uint64_t file_offset = data_start + pt.offset;
        size_t raw_size = pt.raw_bytes;

        // Align pool pointer to 64 bytes
        m_pool_offset = (m_pool_offset + 63ULL) & ~63ULL;
        uint8_t* dst_ptr = m_pool + m_pool_offset;

        if (pt.type == 12) { // Q4_K: Pre-expand 12-byte packed scales into 16 aligned uint8_t (+4 bytes/block)
            size_t n_blocks = pt.n_elements / QK_K;
            size_t blocks_per_chunk = m_scratch_size / sizeof(block_q4_K);
            size_t blocks_processed = 0;

            if (!seek_file(hFile, file_offset)) {
                std::cerr << "[-] Failed to seek tensor: " << pt.name << "\n";
                CloseHandle(hFile);
                return false;
            }

            while (blocks_processed < n_blocks) {
                size_t cur_blocks = (std::min)(n_blocks - blocks_processed, blocks_per_chunk);
                size_t cur_bytes = cur_blocks * sizeof(block_q4_K);

                if (!read_exact(hFile, m_scratch_buf, cur_bytes)) {
                    std::cerr << "[-] Failed to read chunk of tensor: " << pt.name << "\n";
                    CloseHandle(hFile);
                    return false;
                }
                m_bytes_loaded += cur_bytes;

                const block_q4_K* src_blocks = (const block_q4_K*)m_scratch_buf;
                block_q4_K_expanded* dst_blocks = (block_q4_K_expanded*)(dst_ptr + blocks_processed * sizeof(block_q4_K_expanded));

                for (size_t b = 0; b < cur_blocks; ++b) {
                    expand_q4_k_block(&src_blocks[b], &dst_blocks[b]);
                }
                blocks_processed += cur_blocks;
            }
            m_pool_offset += n_blocks * sizeof(block_q4_K_expanded);
        } else {
            // Direct chunked staging (e.g. Q6_K, F32)
            size_t bytes_processed = 0;

            if (!seek_file(hFile, file_offset)) {
                std::cerr << "[-] Failed to seek tensor: " << pt.name << "\n";
                CloseHandle(hFile);
                return false;
            }

            while (bytes_processed < raw_size) {
                size_t cur_bytes = (std::min)(raw_size - bytes_processed, m_scratch_size);

                if (!read_exact(hFile, m_scratch_buf, cur_bytes)) {
                    std::cerr << "[-] Failed to read chunk of tensor: " << pt.name << "\n";
                    CloseHandle(hFile);
                    return false;
                }
                m_bytes_loaded += cur_bytes;

                memcpy(dst_ptr + bytes_processed, m_scratch_buf, cur_bytes);
                bytes_processed += cur_bytes;
            }
            m_pool_offset += raw_size;
        }

        // Map pointers to layers
        if (pt.name == "token_embd.weight") {
            m_token_embd = dst_ptr;
        } else if (pt.name == "output_norm.weight") {
            m_output_norm = (const float*)dst_ptr;
        } else if (pt.name.rfind("blk.", 0) == 0) {
            // Format: blk.L.tensor_name
            size_t dot1 = pt.name.find('.');
            size_t dot2 = pt.name.find('.', dot1 + 1);
            if (dot1 != std::string::npos && dot2 != std::string::npos) {
                int lyr = std::stoi(pt.name.substr(dot1 + 1, dot2 - dot1 - 1));
                std::string tname = pt.name.substr(dot2 + 1);

                if (lyr >= 0 && lyr < NUM_LAYERS) {
                    Gemma4Layer& l = m_layers[lyr];
                    l.layer_bytes += raw_size;

                    if (tname == "attn_q.weight") l.attn_q = dst_ptr;
                    else if (tname == "attn_k.weight") l.attn_k = dst_ptr;
                    else if (tname == "attn_v.weight") {
                        l.attn_v = dst_ptr;
                        l.v_is_q6_k = (pt.type == 14);
                    }
                    else if (tname == "attn_output.weight") l.attn_o = dst_ptr;
                    else if (tname == "ffn_gate.weight") l.ffn_gate = dst_ptr;
                    else if (tname == "ffn_up.weight") l.ffn_up = dst_ptr;
                    else if (tname == "ffn_down.weight") {
                        l.ffn_down = dst_ptr;
                        l.down_is_q6_k = (pt.type == 14);
                    }
                    else if (tname == "attn_norm.weight") l.attn_norm = (const float*)dst_ptr;
                    else if (tname == "attn_q_norm.weight") l.attn_q_norm = (const float*)dst_ptr;
                    else if (tname == "attn_k_norm.weight") l.attn_k_norm = (const float*)dst_ptr;
                    else if (tname == "post_attention_norm.weight") l.attn_o_norm = (const float*)dst_ptr;
                    else if (tname == "ffn_norm.weight") l.ffn_norm = (const float*)dst_ptr;
                    else if (tname == "post_ffw_norm.weight") l.post_ffw_norm = (const float*)dst_ptr;
                    else if (tname == "layer_output_scale.weight") l.layer_output_scale = (const float*)dst_ptr;
                }
            }
        }
        update_peak_memory();
    }

    QueryPerformanceCounter(&t_end);
    m_load_time_sec = (double)(t_end.QuadPart - t_start.QuadPart) / freq.QuadPart;

    // Prefault pages to ensure zero page faults during GEMM/GEMV benchmark
    prefault_pool();
    update_peak_memory();

    CloseHandle(hFile);

    // Free scratch buffer now that staging is finished
    if (m_scratch_buf) {
        VirtualFree(m_scratch_buf, 0, MEM_RELEASE);
        m_scratch_buf = nullptr;
    }

    std::cout << "\n[+] Successfully Staged and Scaled Gemma 4 12B Contiguous Pool!\n";
    std::cout << "    - Disk Staged:        " << (m_bytes_loaded / (1024.0 * 1024.0)) << " MB (" 
              << (m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << "    - Expanded RAM Pool:  " << (m_pool_offset / (1024.0 * 1024.0)) << " MB ("
              << (m_pool_offset / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << "    - Scale Expansion:    +139.0 MB (+1.98% RAM, eliminates scalar bit shifts)\n";
    std::cout << "    - Staging Time:       " << std::fixed << std::setprecision(2) << m_load_time_sec << " s\n";
    std::cout << "    - Staging Line Rate:  " << get_staging_bandwidth_gb_s() << " GB/s\n";
    std::cout << "    - Peak Working Set:   " << get_peak_ws_gb() << " GB\n\n";

    return true;
}

} // namespace gemma4
