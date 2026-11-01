#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include "llama.h"

int main() {
    llama_backend_init();

    std::string model_path = "C:/Users/abhik/Downloads/LFM2.5-350M-Q4_K_M.gguf";
    std::cout << "[*] Loading model: " << model_path << "...\n";

    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "[-] Failed to load model\n";
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "[-] Failed to init context\n";
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    std::string prompt = "<|startoftext|><|user|>\nWhat is 2+2? Answer in one word.<|endoftext|>\n<|assistant|>\n";
    std::vector<llama_token> tokens(256);
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.length(), tokens.data(), (int32_t)tokens.size(), true, true);
    tokens.resize(n_prompt);

    std::cout << "[*] Prompt tokens: " << n_prompt << "\n";
    std::cout << "[*] Prefilling...\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // Prefill prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "[-] Prefill failed\n";
        return 1;
    }

    auto t_first = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
    std::cout << "[+] TTFT (Prefill Latency): " << ttft_ms << " ms\n";
    std::cout << "[+] Output: ";

    int gen_tokens = 0;
    llama_token curr_tok = -1;

    for (int i = 0; i < 32; ++i) {
        float* logits = llama_get_logits_ith(ctx, -1);
        
        // Argmax
        int best_id = 0;
        float best_val = logits[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_id = v;
            }
        }

        curr_tok = best_id;
        if (llama_vocab_is_eog(vocab, curr_tok)) break;

        char buf[64];
        int l = llama_token_to_piece(vocab, curr_tok, buf, sizeof(buf), 0, true);
        std::cout << std::string(buf, l) << std::flush;
        gen_tokens++;

        llama_batch single = llama_batch_get_one(&curr_tok, 1);
        if (llama_decode(ctx, single) != 0) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double decode_time_sec = std::chrono::duration<double>(t_end - t_first).count();
    double tps = (gen_tokens > 0 && decode_time_sec > 0.0) ? (gen_tokens / decode_time_sec) : 0.0;

    std::cout << "\n\n[+] Tokens Generated: " << gen_tokens << "\n";
    std::cout << "[+] Decode Time: " << (decode_time_sec * 1000.0) << " ms\n";
    std::cout << "[+] Decode TPS: " << tps << " tok/s\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
