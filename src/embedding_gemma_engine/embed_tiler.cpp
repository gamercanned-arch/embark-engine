#include "embed_tiler.h"
#include <psapi.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <string>

#pragma comment(lib, "psapi.lib")

namespace embed_gemma {

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
    uint32_t type = 0; // 0=F32, 1=F16, 8=Q8_0, 14=Q6_K, 30=BF16
    uint64_t offset = 0;
    uint64_t raw_bytes = 0;
    uint64_t n_elements = 0;
};

EmbedTiler::EmbedTiler() {
    m_pool_size = EXPANDED_POOL_SIZE;
    m_scratch_size = SCRATCH_BUFFER_BYTES;
    m_layers.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; ++i) {
        m_layers[i].layer_idx = i;
        bool is_swa = (i % 6 != 5); // 0..4, 6..10, 12..16, 18..22 are SWA
        m_layers[i].is_sliding_window = is_swa;
        m_layers[i].q_heads = 4;
        m_layers[i].kv_heads = is_swa ? 2 : 1;
        m_layers[i].head_dim = is_swa ? 256 : 512;
        m_layers[i].q_dim = is_swa ? 1024 : 2048;
        m_layers[i].kv_dim = 512;
    }
    m_vision_layers.resize(VISION_NUM_LAYERS);
    for (int i = 0; i < VISION_NUM_LAYERS; ++i) {
        m_vision_layers[i].layer_idx = i;
    }
}

EmbedTiler::~EmbedTiler() {
    if (m_scratch_buf) {
        VirtualFree(m_scratch_buf, 0, MEM_RELEASE);
        m_scratch_buf = nullptr;
    }
    if (m_pool) {
        VirtualFree(m_pool, 0, MEM_RELEASE);
        m_pool = nullptr;
    }
}

void EmbedTiler::update_peak_memory() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        if (pmc.PeakWorkingSetSize > m_peak_ws_bytes) {
            m_peak_ws_bytes = pmc.PeakWorkingSetSize;
        }
    }
}

void EmbedTiler::prefault_pool() {
    if (!m_pool) return;
    volatile uint64_t sum = 0;
    const uint64_t* p = (const uint64_t*)m_pool;
    size_t count = m_pool_offset / sizeof(uint64_t);
    for (size_t i = 0; i < count; i += 512) { // 4 KB page stride
        sum += p[i];
    }
}

double EmbedTiler::get_staging_bandwidth_gb_s() const {
    if (m_load_time_sec <= 0.0) return 0.0;
    return (m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) / m_load_time_sec;
}

bool EmbedTiler::load_and_stage(const std::string& text_gguf_path, const std::string& mmproj_gguf_path) {
    std::cout << "================================================================================\n";
    std::cout << "  EmbeddingGemma-2 Direct-I/O Chunked Staging Loader\n";
    std::cout << "  Target APU: AMD Ryzen 7 7730U (Barcelo-R Zen 3 8C/16T + Vega 8 UMA)\n";
    std::cout << "  Text Backbone: " << text_gguf_path << "\n";
    if (!mmproj_gguf_path.empty()) {
        std::cout << "  Vision mmproj: " << mmproj_gguf_path << "\n";
    }
    std::cout << "================================================================================\n\n";

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    // 1. Allocate contiguous memory pool via VirtualAlloc aligned to 64 KB
    std::cout << "[1/4] Allocating Contiguous Memory Pool (~1.35 GiB / " 
              << (m_pool_size / (1024 * 1024)) << " MB) via VirtualAlloc...\n";
    m_pool = (uint8_t*)VirtualAlloc(NULL, m_pool_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_pool) {
        std::cerr << "[-] Failed to allocate contiguous memory pool: " << GetLastError() << "\n";
        return false;
    }
    std::cout << "      Contiguous Pool Base: 0x" << (void*)m_pool << " (64 KB Aligned)\n";

    // 2. Allocate 250 MB reusable scratch buffer
    std::cout << "[2/4] Allocating Reusable Staging Scratch Buffer (250 MB)...\n";
    m_scratch_buf = (uint8_t*)VirtualAlloc(NULL, m_scratch_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!m_scratch_buf) {
        std::cerr << "[-] Failed to allocate 250 MB scratch buffer: " << GetLastError() << "\n";
        return false;
    }

    // 3. Stage Text Backbone GGUF
    std::cout << "[3/4] Direct-I/O Staging Text Backbone (EmbeddingGemma-2 271M Q6_K)...\n";
    if (!stage_text_gguf(text_gguf_path)) {
        return false;
    }

    // 4. Stage Vision Projector GGUF (if provided)
    if (!mmproj_gguf_path.empty()) {
        std::cout << "[4/4] Direct-I/O Staging Vision Projector (mmproj 935 MB F16)...\n";
        if (!stage_vision_gguf(mmproj_gguf_path)) {
            return false;
        }
    } else {
        std::cout << "[4/4] Vision Projector omitted (Text-only mode).\n";
    }

    prefault_pool();
    update_peak_memory();

    QueryPerformanceCounter(&t_end);
    m_load_time_sec = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;

    std::cout << "\n[+] Direct-I/O Staging Complete!\n";
    std::cout << "    Total Staged Bytes: " << std::fixed << std::setprecision(2)
              << (m_bytes_loaded / (1024.0 * 1024.0)) << " MB (" 
              << (m_bytes_loaded / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << "    Elapsed Time:       " << std::fixed << std::setprecision(3) << m_load_time_sec << " s\n";
    std::cout << "    Staging Bandwidth:  " << std::fixed << std::setprecision(2) << get_staging_bandwidth_gb_s() << " GB/s\n";
    std::cout << "    Peak Working Set:   " << std::fixed << std::setprecision(2) << get_peak_ws_gb() << " GiB\n\n";

    return true;
}

bool EmbedTiler::stage_text_gguf(const std::string& path) {
    HANDLE hFile = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[-] Error opening text GGUF: " << GetLastError() << "\n";
        return false;
    }

    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    DWORD bytes_read = 0;

    ReadFile(hFile, &magic, 4, &bytes_read, NULL);
    ReadFile(hFile, &version, 4, &bytes_read, NULL);
    ReadFile(hFile, &n_tensors, 8, &bytes_read, NULL);
    ReadFile(hFile, &n_kv, 8, &bytes_read, NULL);

    if (magic != 0x46554747) {
        std::cerr << "[-] Invalid GGUF magic in text model: 0x" << std::hex << magic << "\n";
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
        } else if (pt.type == 1) { // F16
            pt.raw_bytes = pt.n_elements * 2;
        } else if (pt.type == 8) { // Q8_0
            pt.raw_bytes = (pt.n_elements / 32) * sizeof(block_q8_0);
        } else if (pt.type == 14) { // Q6_K
            pt.raw_bytes = (pt.n_elements / 256) * sizeof(block_q6_K);
        } else if (pt.type == 30) { // BF16
            pt.raw_bytes = pt.n_elements * 2;
        } else {
            pt.raw_bytes = pt.n_elements; // fallback
        }

        tensor_list.push_back(pt);
    }

    LARGE_INTEGER cur_pos;
    cur_pos.QuadPart = 0;
    SetFilePointerEx(hFile, cur_pos, &cur_pos, FILE_CURRENT);
    uint64_t data_base_offset = (cur_pos.QuadPart + (alignment - 1)) & ~(uint64_t)(alignment - 1);

    std::unordered_map<std::string, void*> mapped_ptrs;

    for (const auto& t : tensor_list) {
        uint64_t file_offset = data_base_offset + t.offset;
        seek_file(hFile, file_offset);

        // Align pool pointer to 64 bytes
        m_pool_offset = (m_pool_offset + 63ULL) & ~63ULL;
        void* dst = m_pool + m_pool_offset;

        if (!read_exact(hFile, dst, t.raw_bytes)) {
            std::cerr << "[-] Failed reading tensor " << t.name << "\n";
            CloseHandle(hFile);
            return false;
        }

        m_pool_offset += t.raw_bytes;
        m_bytes_loaded += t.raw_bytes;
        mapped_ptrs[t.name] = dst;
    }

    CloseHandle(hFile);

    // Map global text model tensors
    m_token_embd = mapped_ptrs["token_embd.weight"];
    m_per_layer_model_proj = mapped_ptrs["per_layer_model_proj.weight"];
    m_per_layer_proj_norm = (const float*)mapped_ptrs["per_layer_proj_norm.weight"];
    m_output_norm = (const float*)mapped_ptrs["output_norm.weight"];
    m_output_proj = mapped_ptrs["output.weight"];

    // Map 24 layer tensors
    for (int i = 0; i < NUM_LAYERS; ++i) {
        std::string pfx = "blk." + std::to_string(i) + ".";
        auto& lyr = m_layers[i];

        lyr.attn_norm = (const float*)mapped_ptrs[pfx + "attn_norm.weight"];
        lyr.attn_q = mapped_ptrs[pfx + "attn_q.weight"];
        lyr.attn_k = mapped_ptrs[pfx + "attn_k.weight"];
        lyr.attn_v = mapped_ptrs[pfx + "attn_v.weight"];
        lyr.attn_output = mapped_ptrs[pfx + "attn_output.weight"];
        lyr.attn_q_norm = (const float*)mapped_ptrs[pfx + "attn_q_norm.weight"];
        lyr.attn_k_norm = (const float*)mapped_ptrs[pfx + "attn_k_norm.weight"];
        lyr.post_attention_norm = (const float*)mapped_ptrs[pfx + "post_attention_norm.weight"];

        lyr.ffn_norm = (const float*)mapped_ptrs[pfx + "ffn_norm.weight"];
        lyr.ffn_gate = mapped_ptrs[pfx + "ffn_gate.weight"];
        lyr.ffn_up = mapped_ptrs[pfx + "ffn_up.weight"];
        lyr.ffn_down = mapped_ptrs[pfx + "ffn_down.weight"];
        lyr.post_ffw_norm = (const float*)mapped_ptrs[pfx + "post_ffw_norm.weight"];

        lyr.inp_gate = mapped_ptrs[pfx + "inp_gate.weight"];
        lyr.proj = mapped_ptrs[pfx + "proj.weight"];
        lyr.post_norm = (const float*)mapped_ptrs[pfx + "post_norm.weight"];

        const float* scale_ptr = (const float*)mapped_ptrs[pfx + "layer_output_scale.weight"];
        if (scale_ptr) lyr.layer_output_scale = *scale_ptr;
    }

    std::cout << "      Staged 24 Text Backbone Layers (" << (m_pool_offset / (1024 * 1024)) << " MB in Pool)\n";
    return true;
}

bool EmbedTiler::stage_vision_gguf(const std::string& path) {
    HANDLE hFile = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[-] Error opening vision GGUF: " << GetLastError() << "\n";
        return false;
    }

    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    DWORD bytes_read = 0;

    ReadFile(hFile, &magic, 4, &bytes_read, NULL);
    ReadFile(hFile, &version, 4, &bytes_read, NULL);
    ReadFile(hFile, &n_tensors, 8, &bytes_read, NULL);
    ReadFile(hFile, &n_kv, 8, &bytes_read, NULL);

    if (magic != 0x46554747) {
        std::cerr << "[-] Invalid GGUF magic in vision model: 0x" << std::hex << magic << "\n";
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
        } else if (pt.type == 1) { // F16
            pt.raw_bytes = pt.n_elements * 2;
        } else {
            pt.raw_bytes = pt.n_elements * 2;
        }

        tensor_list.push_back(pt);
    }

    LARGE_INTEGER cur_pos;
    cur_pos.QuadPart = 0;
    SetFilePointerEx(hFile, cur_pos, &cur_pos, FILE_CURRENT);
    uint64_t data_base_offset = (cur_pos.QuadPart + (alignment - 1)) & ~(uint64_t)(alignment - 1);

    std::unordered_map<std::string, void*> mapped_ptrs;

    for (const auto& t : tensor_list) {
        uint64_t file_offset = data_base_offset + t.offset;
        seek_file(hFile, file_offset);

        m_pool_offset = (m_pool_offset + 63ULL) & ~63ULL;
        void* dst = m_pool + m_pool_offset;

        if (!read_exact(hFile, dst, t.raw_bytes)) {
            std::cerr << "[-] Failed reading vision tensor " << t.name << "\n";
            CloseHandle(hFile);
            return false;
        }

        m_pool_offset += t.raw_bytes;
        m_bytes_loaded += t.raw_bytes;
        mapped_ptrs[t.name] = dst;
    }

    CloseHandle(hFile);

    m_has_vision = true;
    m_vision_patch_embd = mapped_ptrs["v.patch_embd.weight"];
    m_vision_position_embd = (const float*)mapped_ptrs["v.position_embd.weight"];
    m_mm_input_projection = mapped_ptrs["mm.input_projection.weight"];

    for (int i = 0; i < VISION_NUM_LAYERS; ++i) {
        std::string pfx = "v.blk." + std::to_string(i) + ".";
        auto& lyr = m_vision_layers[i];
        lyr.ln1_weight = (const float*)mapped_ptrs[pfx + "ln1.weight"];
        lyr.attn_q = mapped_ptrs[pfx + "attn_q.weight"];
        lyr.attn_k = mapped_ptrs[pfx + "attn_k.weight"];
        lyr.attn_v = mapped_ptrs[pfx + "attn_v.weight"];
        lyr.attn_out = mapped_ptrs[pfx + "attn_out.weight"];
        lyr.ln2_weight = (const float*)mapped_ptrs[pfx + "ln2.weight"];
        lyr.ffn_gate = mapped_ptrs[pfx + "ffn_gate.weight"];
        lyr.ffn_up = mapped_ptrs[pfx + "ffn_up.weight"];
        lyr.ffn_down = mapped_ptrs[pfx + "ffn_down.weight"];
    }

    std::cout << "      Staged 16 Vision Transformer Blocks (" << (m_pool_offset / (1024 * 1024)) << " MB in Pool)\n";
    return true;
}

} // namespace embed_gemma
