#include <iostream>
#include <string>
#include <vector>
#include "llama.h"

int main() {
    llama_backend_init();

    std::string model_path = "C:/Users/abhik/Downloads/GrootTeam_Groot-19B-A3B-261023-IQ3_XXS.gguf";
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;

    llama_model* model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model) return 1;

    const char* tmpl = llama_model_chat_template(model, nullptr);
    std::cout << "[+] Default chat template found: " << (tmpl ? "YES" : "NO") << "\n";

    std::vector<llama_chat_message> messages = {
        {"system", "You are Groot, a helpful AI assistant."},
        {"user", "Hello! Who are you?"}
    };

    char buf[2048];
    int res = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, buf, sizeof(buf));
    if (res > 0) {
        std::cout << "[+] Formatted Prompt:\n---\n" << std::string(buf, res) << "\n---\n";
    } else {
        std::cout << "[-] Failed to apply chat template (res=" << res << ")\n";
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
