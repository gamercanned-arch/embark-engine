#include "embark_session_manager.h"
#include <iostream>
#include <chrono>
#include <thread>

namespace embark {

SessionManager::SessionManager(UniversalVocab& vocab) 
    : m_vocab(vocab) {}

SessionManager::~SessionManager() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Free LFM2 contexts & model
    for (auto* ctx : m_lfm2_contexts) {
        if (ctx) llama_free(ctx);
    }
    m_lfm2_contexts.clear();
    if (m_lfm2_model) {
        llama_model_free(m_lfm2_model);
        m_lfm2_model = nullptr;
    }

    // Free LFM3 contexts & model
    for (auto* ctx : m_lfm3_contexts) {
        if (ctx) llama_free(ctx);
    }
    m_lfm3_contexts.clear();
    if (m_lfm3_model) {
        llama_model_free(m_lfm3_model);
        m_lfm3_model = nullptr;
    }

    // Free Groot context & model
    if (m_groot_ctx) {
        llama_free(m_groot_ctx);
        m_groot_ctx = nullptr;
    }
    if (m_groot_model) {
        llama_model_free(m_groot_model);
        m_groot_model = nullptr;
    }

    m_states.clear();
}

void SessionManager::register_model(const ModelConfig& config) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_models[config.id] = config;
    m_states[config.id] = ModelState::UNLOADED;

    // Pre-register vocabulary for rapid tokenization
    m_vocab.register_model_vocab(config.id, config.gguf_path);
}

bool SessionManager::preload_resident_models() {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::cout << "[SessionManager] Pre-loading Resident Models (LFM3 VL & LFM2 350M)...\n";
    
    // 1. Staging LFM2 350M
    auto it_lfm2 = m_models.find("lfm2-350m");
    if (it_lfm2 != m_models.end()) {
        std::cout << "  -> Staging Resident Model: lfm2-350m (" << it_lfm2->second.gguf_path << ")...\n";
        llama_model_params mparams = llama_model_default_params();
        m_lfm2_model = llama_model_load_from_file(it_lfm2->second.gguf_path.c_str(), mparams);
        if (m_lfm2_model) {
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = 2048;
            cparams.n_threads = 8;
            cparams.n_threads_batch = 8;
            for (int i = 0; i < 4; ++i) {
                m_lfm2_contexts.push_back(llama_init_from_model(m_lfm2_model, cparams));
                m_lfm2_busy.push_back(false);
            }
            m_states["lfm2-350m"] = ModelState::RESIDENT;
            std::cout << "     [+] lfm2-350m active with 4 execution contexts.\n";
        }
    }

    // 2. Staging LFM3-VL-2B (Exactly 3 slots for 3 concurrent sessions)
    auto it_lfm3 = m_models.find("lfm3-vl-2b");
    if (it_lfm3 != m_models.end()) {
        std::cout << "  -> Staging Resident Model: lfm3-vl-2b (" << it_lfm3->second.gguf_path << ")...\n";
        llama_model_params mparams = llama_model_default_params();
        m_lfm3_model = llama_model_load_from_file(it_lfm3->second.gguf_path.c_str(), mparams);
        if (m_lfm3_model) {
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = 2048;
            cparams.n_threads = 8;
            cparams.n_threads_batch = 8;
            for (int i = 0; i < 3; ++i) {
                m_lfm3_contexts.push_back(llama_init_from_model(m_lfm3_model, cparams));
                m_lfm3_busy.push_back(false);
            }
            m_states["lfm3-vl-2b"] = ModelState::RESIDENT;
            std::cout << "     [+] lfm3-vl-2b active with 3 concurrent execution contexts.\n";
        }
    }

    std::cout << "[SessionManager] Resident models active in memory (~1.70 GB baseline working set).\n";
    return true;
}

bool SessionManager::acquire_slot(const std::string& model_id) {
    if (model_id == "groot-19b" || model_id == "ornith-9b") {
        // Strictly 1 concurrent session for heavy models
        m_groot_mutex.lock();
        ensure_model_staged(model_id);
        return true;
    } else if (model_id == "lfm3-vl-2b") {
        // Up to 3 concurrent sessions for LFM3 VL
        std::unique_lock<std::mutex> lock(m_lfm3_mutex);
        m_lfm3_cv.wait(lock, [this]() { return m_lfm3_active_slots < MAX_LFM3_CONCURRENCY; });
        m_lfm3_active_slots++;
        ensure_model_staged(model_id);
        return true;
    } else {
        // LFM2 350M and other fast utilities: direct access
        ensure_model_staged(model_id);
        return true;
    }
}

void SessionManager::release_slot(const std::string& model_id) {
    if (model_id == "groot-19b" || model_id == "ornith-9b") {
        m_groot_mutex.unlock();
    } else if (model_id == "lfm3-vl-2b") {
        std::lock_guard<std::mutex> lock(m_lfm3_mutex);
        m_lfm3_active_slots--;
        m_lfm3_cv.notify_one();
    }
}

bool SessionManager::ensure_model_staged(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(m_staging_mutex);
    auto it = m_models.find(model_id);
    if (it == m_models.end()) return false;

    if (it->second.always_resident) {
        m_states[model_id] = ModelState::RESIDENT;
        return true;
    }

    if (m_currently_staged_heavy_model == model_id && m_groot_model) {
        return true; // Already staged
    }

    unstage_heavy_model_if_needed(model_id);
    std::cout << "[SessionManager] Swapping IN heavy model: " << model_id << " (" << it->second.gguf_path << ")...\n";

    llama_model_params mparams = llama_model_default_params();
    m_groot_model = llama_model_load_from_file(it->second.gguf_path.c_str(), mparams);
    if (!m_groot_model) {
        std::cerr << "[-] Failed to load model weights for: " << model_id << "\n";
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; // Bound KV buffer
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    m_groot_ctx = llama_init_from_model(m_groot_model, cparams);
    if (!m_groot_ctx) {
        std::cerr << "[-] Failed to create context for: " << model_id << "\n";
        llama_model_free(m_groot_model);
        m_groot_model = nullptr;
        return false;
    }

    m_currently_staged_heavy_model = model_id;
    m_states[model_id] = ModelState::SWAPPED_IN;
    std::cout << "[SessionManager] Heavy model " << model_id << " staged and ready for inference.\n";
    return true;
}

void SessionManager::unstage_heavy_model_if_needed(const std::string& next_heavy_model) {
    if (!m_currently_staged_heavy_model.empty() && m_currently_staged_heavy_model != next_heavy_model) {
        std::cout << "[SessionManager] Swapping OUT previous model: " << m_currently_staged_heavy_model << "...\n";
        if (m_groot_ctx) {
            llama_free(m_groot_ctx);
            m_groot_ctx = nullptr;
        }
        if (m_groot_model) {
            llama_model_free(m_groot_model);
            m_groot_model = nullptr;
        }
        m_states[m_currently_staged_heavy_model] = ModelState::UNLOADED;
        m_currently_staged_heavy_model.clear();
    }
}

std::vector<std::string> SessionManager::get_registered_models() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<std::string> res;
    for (const auto& pair : m_models) {
        res.push_back(pair.first);
    }
    return res;
}

ModelState SessionManager::get_model_state(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_states.find(model_id);
    return (it != m_states.end()) ? it->second : ModelState::UNLOADED;
}

std::string SessionManager::format_prompt(const std::string& model_id, const std::vector<ChatMessage>& messages) const {
    return m_vocab.format_chat(model_id, messages, /*add_assistant_prompt=*/true);
}

bool SessionManager::generate_stream(
    const std::string& model_id,
    const std::string& prompt,
    const SamplingParams& params,
    std::function<bool(const std::string& token_text)> token_callback
) {
    std::vector<int32_t> prompt_tokens = m_vocab.tokenize(model_id, prompt, false, true);
    if (prompt_tokens.empty()) {
        std::cerr << "[-] [generate_stream] Failed to tokenize prompt for model " << model_id << "\n";
        return false;
    }

    llama_context* ctx = nullptr;
    int ctx_idx = -1;

    if (model_id == "lfm2-350m") {
        std::lock_guard<std::mutex> lk(m_lfm2_ctx_mutex);
        for (size_t i = 0; i < m_lfm2_contexts.size(); ++i) {
            if (!m_lfm2_busy[i]) {
                m_lfm2_busy[i] = true;
                ctx = m_lfm2_contexts[i];
                ctx_idx = (int)i;
                break;
            }
        }
    } else if (model_id == "lfm3-vl-2b") {
        std::lock_guard<std::mutex> lk(m_lfm3_ctx_mutex);
        for (size_t i = 0; i < m_lfm3_contexts.size(); ++i) {
            if (!m_lfm3_busy[i]) {
                m_lfm3_busy[i] = true;
                ctx = m_lfm3_contexts[i];
                ctx_idx = (int)i;
                break;
            }
        }
    } else if (model_id == "groot-19b" || model_id == "ornith-9b") {
        ctx = m_groot_ctx;
    }

    if (!ctx) {
        std::cerr << "[-] [generate_stream] No available context for model " << model_id << "\n";
        return false;
    }

    struct CtxReleaser {
        std::mutex* mtx;
        std::vector<bool>* busy;
        int idx;
        ~CtxReleaser() {
            if (mtx && busy && idx >= 0) {
                std::lock_guard<std::mutex> lk(*mtx);
                (*busy)[idx] = false;
            }
        }
    } releaser{
        (model_id == "lfm2-350m") ? &m_lfm2_ctx_mutex : (model_id == "lfm3-vl-2b") ? &m_lfm3_ctx_mutex : nullptr,
        (model_id == "lfm2-350m") ? &m_lfm2_busy : (model_id == "lfm3-vl-2b") ? &m_lfm3_busy : nullptr,
        ctx_idx
    };

    // 1. Clear KV memory for clean generation session
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // 2. Prefill prompt tokens
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "[-] [generate_stream] Prefill decode failed!\n";
        return false;
    }

    size_t vocab_size = m_vocab.get_vocab_size(model_id);
    if (vocab_size == 0) vocab_size = 248320;

    std::vector<int32_t> generated_tokens;
    int max_toks = (params.max_tokens > 0) ? params.max_tokens : 512;

    // 3. Autoregressive token generation loop
    for (int step = 0; step < max_toks; ++step) {
        float* logits = llama_get_logits_ith(ctx, -1);
        int32_t next_tok = m_sampler.sample(logits, vocab_size, generated_tokens, params);

        if (m_vocab.is_eog(model_id, next_tok) || next_tok == m_vocab.get_eos(model_id)) {
            break;
        }

        generated_tokens.push_back(next_tok);
        std::string piece = m_vocab.detokenize(model_id, next_tok);

        if (!token_callback(piece)) {
            break; // Client aborted connection
        }

        llama_batch single = llama_batch_get_one(&next_tok, 1);
        if (llama_decode(ctx, single) != 0) {
            break;
        }
    }

    return true;
}

std::string SessionManager::generate_completion(
    const std::string& model_id,
    const std::string& prompt,
    const SamplingParams& params
) {
    std::string full_response;
    generate_stream(model_id, prompt, params, [&](const std::string& piece) {
        full_response += piece;
        return true;
    });
    return full_response;
}

} // namespace embark
