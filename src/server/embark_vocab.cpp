#include "embark_vocab.h"
#include <iostream>
#include <cstring>

namespace embark {

UniversalVocab::UniversalVocab() {
    llama_backend_init();
}

UniversalVocab::~UniversalVocab() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& pair : m_vocabs) {
        if (pair.second && pair.second->model) {
            llama_model_free(pair.second->model);
            pair.second->model = nullptr;
        }
    }
    m_vocabs.clear();
    llama_backend_free();
}

bool UniversalVocab::register_model_vocab(const std::string& model_id, const std::string& gguf_path) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_vocabs.find(model_id) != m_vocabs.end()) {
        return true; // Already registered
    }

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;

    llama_model* model = llama_model_load_from_file(gguf_path.c_str(), params);
    if (!model) {
        std::cerr << "[-] [UniversalVocab] Failed to load vocab from: " << gguf_path << "\n";
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        std::cerr << "[-] [UniversalVocab] Failed to extract vocab from model: " << model_id << "\n";
        llama_model_free(model);
        return false;
    }

    auto entry = std::make_unique<ModelVocabEntry>();
    entry->model = model;
    entry->vocab = vocab;
    entry->n_vocab = llama_vocab_n_tokens(vocab);
    entry->bos_token = llama_vocab_bos(vocab);
    entry->eos_token = llama_vocab_eos(vocab);

    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl) {
        entry->chat_template = tmpl;
    }

    std::cout << "[+] [UniversalVocab] Registered '" << model_id 
              << "' (Vocab: " << entry->n_vocab << " tokens, BOS: " << entry->bos_token 
              << ", EOS: " << entry->eos_token << ", Template: " << (!entry->chat_template.empty() ? "YES" : "NO") << ")\n";

    m_vocabs[model_id] = std::move(entry);
    return true;
}

bool UniversalVocab::has_vocab(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_vocabs.find(model_id) != m_vocabs.end();
}

std::vector<int32_t> UniversalVocab::tokenize(const std::string& model_id, const std::string& text, bool add_special, bool parse_special) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_vocabs.find(model_id);
    if (it == m_vocabs.end()) return {};

    const auto& entry = it->second;
    std::vector<int32_t> tokens(text.length() + 32);
    int32_t n = llama_tokenize(entry->vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int32_t)tokens.size(), add_special, parse_special);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(entry->vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int32_t)tokens.size(), add_special, parse_special);
    }
    if (n > 0) {
        tokens.resize(n);
    } else {
        tokens.clear();
    }
    return tokens;
}

std::string UniversalVocab::detokenize(const std::string& model_id, int32_t token_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_vocabs.find(model_id);
    if (it == m_vocabs.end()) return "";

    char buf[256];
    int len = llama_token_to_piece(it->second->vocab, token_id, buf, sizeof(buf), 0, true);
    if (len > 0) {
        return std::string(buf, len);
    }
    return "";
}

std::string UniversalVocab::detokenize_all(const std::string& model_id, const std::vector<int32_t>& tokens) const {
    std::string out;
    for (int32_t t : tokens) {
        out += detokenize(model_id, t);
    }
    return out;
}

std::string UniversalVocab::format_chat(const std::string& model_id, const std::vector<ChatMessage>& messages, bool add_assistant_prompt) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_vocabs.find(model_id);
    if (it == m_vocabs.end()) return "";

    const auto& entry = it->second;
    std::vector<llama_chat_message> chat_msgs;
    chat_msgs.reserve(messages.size());
    for (const auto& m : messages) {
        chat_msgs.push_back({m.role.c_str(), m.content.c_str()});
    }

    const char* tmpl_ptr = entry->chat_template.empty() ? nullptr : entry->chat_template.c_str();

    // First call to determine size
    int res = llama_chat_apply_template(tmpl_ptr, chat_msgs.data(), chat_msgs.size(), add_assistant_prompt, nullptr, 0);
    if (res <= 0) {
        // Fallback simple ChatML formatting if template engine returns <= 0
        std::string fallback;
        for (const auto& m : messages) {
            fallback += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        }
        if (add_assistant_prompt) {
            fallback += "<|im_start|>assistant\n";
        }
        return fallback;
    }

    std::vector<char> buf(res + 16);
    int actual = llama_chat_apply_template(tmpl_ptr, chat_msgs.data(), chat_msgs.size(), add_assistant_prompt, buf.data(), (int32_t)buf.size());
    if (actual > 0) {
        return std::string(buf.data(), actual);
    }

    return "";
}

bool UniversalVocab::is_eog(const std::string& model_id, int32_t token_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_vocabs.find(model_id);
    if (it == m_vocabs.end()) return true;

    return llama_vocab_is_eog(it->second->vocab, token_id);
}

int32_t UniversalVocab::get_bos(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_vocabs.find(model_id);
    return (it != m_vocabs.end()) ? it->second->bos_token : -1;
}

int32_t UniversalVocab::get_eos(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_vocabs.find(model_id);
    return (it != m_vocabs.end()) ? it->second->eos_token : -1;
}

int32_t UniversalVocab::get_vocab_size(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_vocabs.find(model_id);
    return (it != m_vocabs.end()) ? it->second->n_vocab : 0;
}

} // namespace embark
