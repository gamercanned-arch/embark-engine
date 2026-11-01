#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <functional>
#include "embark_vocab.h"
#include "embark_sampler.h"

namespace embark {

enum class ModelState {
    UNLOADED,
    RESIDENT,
    SWAPPED_IN
};

struct ModelConfig {
    std::string id;
    std::string gguf_path;
    std::string mmproj_path;
    bool always_resident = false;
    int max_concurrency = 1;
};

class SessionManager {
public:
    SessionManager(UniversalVocab& vocab);
    ~SessionManager();

    // Register supported models
    void register_model(const ModelConfig& config);

    // Acquire execution slot according to concurrency rules:
    // (Up to 3 for LFM3 VL, 1 for Groot)
    bool acquire_slot(const std::string& model_id);
    void release_slot(const std::string& model_id);

    // RAII Slot Guard for strict leak-free concurrency
    struct SlotGuard {
        SessionManager& mgr;
        std::string model_id;
        bool acquired = false;
        SlotGuard(SessionManager& m, const std::string& id) : mgr(m), model_id(id) {
            acquired = mgr.acquire_slot(model_id);
        }
        ~SlotGuard() {
            if (acquired) mgr.release_slot(model_id);
        }
    };

    // Generates tokens with streaming callback
    bool generate_stream(
        const std::string& model_id,
        const std::string& prompt,
        const SamplingParams& params,
        std::function<bool(const std::string& token_text)> token_callback
    );

    // Non-streaming completion
    std::string generate_completion(
        const std::string& model_id,
        const std::string& prompt,
        const SamplingParams& params
    );

    // Get list of registered models
    std::vector<std::string> get_registered_models() const;
    ModelState get_model_state(const std::string& model_id) const;

    // Direct pre-load for resident models (LFM3 VL and LFM2 350M)
    bool preload_resident_models();

private:
    UniversalVocab& m_vocab;
    Sampler m_sampler;

    mutable std::mutex m_mutex;
    std::unordered_map<std::string, ModelConfig> m_models;
    std::unordered_map<std::string, ModelState> m_states;

    // Concurrency controllers
    std::mutex m_groot_mutex; // Strict 1 concurrent session for Groot
    int m_lfm3_active_slots = 0;
    const int MAX_LFM3_CONCURRENCY = 3; // Up to 3 concurrent LFM3 sessions
    std::condition_variable m_lfm3_cv;
    std::mutex m_lfm3_mutex;

    // Model weight staging locks
    std::mutex m_staging_mutex;
    std::string m_currently_staged_heavy_model;

    bool ensure_model_staged(const std::string& model_id);
    void unstage_heavy_model_if_needed(const std::string& next_heavy_model);
};

} // namespace embark
