#include "lfm2_tiler.h"
#include <psapi.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <string>

#pragma comment(lib, "psapi.lib")

namespace lfm2 {

static bool seek_file(HANDLE hFile, uint64_t offset) {
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)offset;
    return SetFilePointerEx(hFile, li, NULL, FILE_BEGIN) != FALSE;
}

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
    uint32_t type = 0; // 0=F32, 8=Q8_0
    uint64_t offset = 0;
    uint64_t raw_bytes = 0;
    uint64_t n_elements = 0;
};

LFM2Tiler::LFM2Tiler() {
    m_pool_size = LFM2_POOL_SIZE;
    m_scratch_size = SCRATCH_BUFFER_BYTES;
    m_layers.resize(NUM_LAYERS);

    // 16 layers:
    // ShortConv (10): 0, 1, 3, 4, 6, 7, 9, 11, 13, 15
    // Attention (6):  2, 5, 8, 10, 12, 14
    int shortconv_cnt = 0;
    int attn_cnt = 0;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        m_layers[i].layer_idx = i;
        bool is_conv = (i == 0 || i == 1 || i == 3 || i == 4 || i == 6 || i == 7 || i == 9 || i == 11 || i == 13 || i == 15);
        if (is_conv) {
            m_layers[i].type = LayerType::SHORTCONV;
            m_layers[i].sub_idx = shortconv_cnt++;
        } else {
            m_layers[i].type = LayerType::ATTENTION;
            m_layers[i].sub_idx = attn_cnt++;
        }
    }
}

LFM2Tiler::~LFM2Tiler() {
    if (m_scratch_buf) {
        VirtualFree(m_scratch_buf, 0, MEM_RELEASE);
        m_scratch_buf = nullptr;
    }
    if (m_pool) {
        VirtualFree(m_pool, 0, MEM_RELEASE);
        m_pool = nullptr;
    }
}

void LFM2Tiler::update_peak_memory() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        if (pmc.PeakWorkingSetSize > m_peak_ws_bytes) {
            m_peak_ws_bytes = pmc.PeakWorkingSetSize;
        }
    }
}

void LFM2Tiler::prefault_pool() {
    if (!m_pool) return;
    volatile uint64_t sum = 0;
    const uint64_t* p = (const uint64_t*)m_pool;
    size_t count = m_pool_offset / sizeof(uint64_t);
    for (size_t i = 0; i < count; i += 512) { // 4 KB page stride
        sum += p[i];
    }
}

double LFM2Tiler::get_staging_bandwidth_gb_s() const {
    if (m_load_time_sec <= 0.0) return 0.0;
    return (m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) / m_load_time_sec;
}

bool LFM2Tiler::load_and_stage(const std::string& gguf_path) {
    std::cout << "================================================================================\n";
    std::cout << "  LFM 2.5 350M Direct-I/O Staging Loader into Contiguous Memory Pool\n";
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

    // 2. Allocate contiguous memory pool via VirtualAlloc aligned to 64 KB
    std::cout << "[1/4] Allocating Contiguous Memory Pool (" << (m_pool_size / (1024 * 1024)) << " MB) via VirtualAlloc...\n";
    m_pool = (uint8_t*)VirtualAlloc(NULL, m_pool_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_pool) {
        std::cerr << "[-] Failed to allocate contiguous memory pool: " << GetLastError() << "\n";
        CloseHandle(hFile);
        return false;
    }
    std::cout << "      Contiguous Pool Base: 0x" << (void*)m_pool << " (64 KB Aligned)\n";

    // 3. Allocate 250 MB reusable scratch buffer
    std::cout << "[2/4] Allocating Reusable Staging Scratch Buffer (" << (m_scratch_size / (1024 * 1024)) << " MB)...\n";
    m_scratch_buf = (uint8_t*)VirtualAlloc(NULL, m_scratch_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_scratch_buf) {
        std::cerr << "[-] Failed to allocate scratch buffer: " << GetLastError() << "\n";
        CloseHandle(hFile);
        return false;
    }

    // 4. Parse GGUF Header
    std::cout << "[3/4] Parsing GGUF Header & Tensor Dictionary (148 Tensors)...\n";
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
        } else if (pt.type == 8) { // Q8_0
            pt.raw_bytes = (pt.n_elements / 32) * 34;
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

    // 5. Stage Chunks directly into 64-byte aligned Contiguous Memory Pool
    std::cout << "[4/4] Staging Tensors into 450 MB Contiguous Memory Pool via Direct-I/O...\n";

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
        if (m_pool_offset + raw_size > m_pool_size) {
            std::cerr << "[-] Error: Contiguous pool overflow at tensor " << pt.name << "\n";
            CloseHandle(hFile);
            return false;
        }

        uint8_t* dst_ptr = m_pool + m_pool_offset;

        if (!seek_file(hFile, file_offset)) {
            std::cerr << "[-] Failed to seek tensor: " << pt.name << "\n";
            CloseHandle(hFile);
            return false;
        }

        // Direct staging read
        if (!read_exact(hFile, dst_ptr, raw_size)) {
            std::cerr << "[-] Failed to read tensor: " << pt.name << "\n";
            CloseHandle(hFile);
            return false;
        }

        // Map pointers to layers
        if (pt.name == "token_embd.weight") {
            m_token_embd = (const block_q8_0*)dst_ptr;
        } else if (pt.name == "token_embd_norm.weight") {
            m_output_norm = (const float*)dst_ptr;
        } else if (pt.name.rfind("blk.", 0) == 0) {
            size_t dot1 = pt.name.find('.');
            size_t dot2 = pt.name.find('.', dot1 + 1);
            int layer_idx = std::stoi(pt.name.substr(dot1 + 1, dot2 - dot1 - 1));
            std::string sub_name = pt.name.substr(dot2 + 1);

            auto& layer = m_layers[layer_idx];
            layer.layer_bytes += raw_size;

            if (sub_name == "attn_norm.weight") {
                layer.attn_norm = (const float*)dst_ptr;
            } else if (sub_name == "ffn_norm.weight") {
                layer.ffn_norm = (const float*)dst_ptr;
            } else if (sub_name == "ffn_gate.weight") {
                layer.ffn_gate = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "ffn_up.weight") {
                layer.ffn_up = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "ffn_down.weight") {
                layer.ffn_down = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "shortconv.in_proj.weight") {
                layer.shortconv_in_proj = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "shortconv.conv.weight") {
                layer.shortconv_conv = (const float*)dst_ptr;
            } else if (sub_name == "shortconv.out_proj.weight") {
                layer.shortconv_out_proj = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "attn_q.weight") {
                layer.attn_q = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "attn_q_norm.weight") {
                layer.attn_q_norm = (const float*)dst_ptr;
            } else if (sub_name == "attn_k.weight") {
                layer.attn_k = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "attn_k_norm.weight") {
                layer.attn_k_norm = (const float*)dst_ptr;
            } else if (sub_name == "attn_v.weight") {
                layer.attn_v = (const block_q8_0*)dst_ptr;
            } else if (sub_name == "attn_output.weight") {
                layer.attn_output = (const block_q8_0*)dst_ptr;
            }
        }

        m_pool_offset += raw_size;
        m_bytes_loaded += raw_size;
    }

    QueryPerformanceCounter(&t_end);
    m_load_time_sec = (double)(t_end.QuadPart - t_start.QuadPart) / freq.QuadPart;

    CloseHandle(hFile);

    prefault_pool();
    update_peak_memory();

    std::cout << "      Staging Complete!\n";
    std::cout << "      Total Weight Bytes:  " << (m_bytes_loaded / (1024 * 1024)) << " MB (" 
              << m_bytes_loaded << " bytes)\n";
    std::cout << "      Staging Load Time:   " << std::fixed << std::setprecision(3) << m_load_time_sec << " seconds\n";
    std::cout << "      Staging Bandwidth:   " << std::setprecision(2) << get_staging_bandwidth_gb_s() << " GB/s\n";
    std::cout << "      Contiguous Footprint:" << (m_pool_offset / (1024 * 1024)) << " MB / " 
              << (m_pool_size / (1024 * 1024)) << " MB pool\n";
    std::cout << "      Peak Working Set:    " << get_peak_ws_gb() << " GB\n\n";

    return true;
}

} // namespace lfm2
