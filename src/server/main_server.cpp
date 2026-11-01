#include <iostream>
#include <string>
#include <csignal>
#include "embark_vocab.h"
#include "embark_session_manager.h"
#include "embark_server.h"

static embark::HttpServer* g_server = nullptr;

void signal_handler(int signum) {
    std::cout << "\n[!] Caught shutdown signal (" << signum << "). Stopping EMBARK Server...\n";
    if (g_server) {
        g_server->stop();
    }
}

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: embark_server.exe [--host 127.0.0.1] [--port 8080]\n";
            return 0;
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "================================================================================\n";
    std::cout << "  EMBARK ENGINE: PURE C++ NATIVE OPENAI API SERVER (v1.0.0)\n";
    std::cout << "  Silicon: AMD Ryzen 7 7730U (Zen 3 8C/16T + Vega 8 gfx90c iGPU)\n";
    std::cout << "  Residency: LFM3-VL (2B) + LFM2 (350M) Resident 24/7 in ~1.70 GB RAM\n";
    std::cout << "  On-Demand: Groot 19B (8.24 GB) & Ornith 9B (5.2 GB) Swapped In/Out\n";
    std::cout << "  Concurrency: Up to 3 LFM3-VL sessions | 1 Groot session\n";
    std::cout << "================================================================================\n\n";

    // 1. Initialize Universal Vocab (supports all models)
    embark::UniversalVocab vocab;

    // 2. Initialize Session & Concurrency Manager
    embark::SessionManager session_mgr(vocab);

    // Register models
    session_mgr.register_model({
        "lfm3-vl-2b",
        "C:/Users/abhik/Downloads/LFM3-VL-2B-Q4_K_M.gguf",
        "C:/Users/abhik/Downloads/mmproj-LFM3-VL-2B-F16.gguf",
        true, // Always resident
        3     // Up to 3 concurrent sessions
    });

    session_mgr.register_model({
        "lfm2-350m",
        "C:/Users/abhik/Downloads/LFM2.5-350M-Q4_K_M.gguf",
        "",
        true, // Always resident
        8     // High concurrency
    });

    session_mgr.register_model({
        "groot-19b",
        "C:/Users/abhik/Downloads/GrootTeam_Groot-19B-A3B-261023-IQ3_XXS.gguf",
        "C:/Users/abhik/Downloads/mmproj-GrootTeam_Groot-19B-A3B-261023-f16.gguf",
        false, // Swapped in/out on-demand
        1      // Strict 1 concurrent session
    });

    session_mgr.register_model({
        "ornith-9b",
        "C:/Users/abhik/Downloads/Ornith-1.5-9B-Q4_K_M.gguf",
        "C:/Users/abhik/Downloads/mmproj-Ornith-1.5-9B-BF16.gguf",
        false, // Swapped in/out on-demand
        1      // Strict 1 concurrent session
    });

    // 3. Pre-load resident models
    session_mgr.preload_resident_models();

    // 4. Start HTTP Server
    embark::HttpServer server(session_mgr, host, port);
    g_server = &server;

    server.start_blocking();

    std::cout << "[*] EMBARK Server exited cleanly.\n";
    return 0;
}
