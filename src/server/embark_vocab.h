#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "llama.h"

namespace embark {

struct ChatMessage {
    std::string role;
    std::string content;
};

class UniversalVocab {
public:
    UniversalVocab();
    ~UniversalVocab();

    UniversalVocab(const UniversalVocab&) = delete;
    UniversalVocab& operator=(const UniversalVocab&) = delete;

    // Registers and loads vocabulary-only for a model from GGUF path
    bool register_model_vocab(const std::string& model_id, const std::string& gguf_path);

    // Checks if a model's vocabulary is registered
    bool has_vocab(const std::string& model_id) const;

    // Tokenizes input text using the model's vocabulary and pre-tokenizer
    std::vector<int32_t> tokenize(const std::string& model_id, const std::string& text, bool add_special = false, bool parse_special = true) const;

    // Converts a single token ID back to text
    std::string detokenize(const std::string& model_id, int32_t token_id) const;

    // Converts a sequence of token IDs to text
    std::string detokenize_all(const std::string& model_id, const std::vector<int32_t>& tokens) const;

    // Formats a multi-turn chat using the model's native chat template
    std::string format_chat(const std::string& model_id, const std::vector<ChatMessage>& messages, bool add_assistant_prompt = true) const;

    // Checks if token is end-of-generation (EOS / EOG)
    bool is_eog(const std::string& model_id, int32_t token_id) const;

    // Special token IDs
    int32_t get_bos(const std::string& model_id) const;
    int32_t get_eos(const std::string& model_id) const;
    int32_t get_vocab_size(const std::string& model_id) const;

private:
    struct ModelVocabEntry {
        llama_model* model = nullptr;
        const llama_vocab* vocab = nullptr;
        std::string chat_template;
        int32_t n_vocab = 0;
        int32_t bos_token = -1;
        int32_t eos_token = -1;
    };

    mutable std::mutex m_mutex;
    std::unordered_map<std::string, std::unique_ptr<ModelVocabEntry>> m_vocabs;
};

} // namespace embark
