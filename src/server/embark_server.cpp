#include "embark_server.h"
#include <iostream>
#include <chrono>
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

namespace embark {

HttpServer::HttpServer(SessionManager& session_mgr, const std::string& host, int port)
    : m_session_mgr(session_mgr), m_host(host), m_port(port) {
    m_server_ptr = new httplib::Server();
    register_routes();
}

HttpServer::~HttpServer() {
    stop();
    if (m_server_ptr) {
        delete static_cast<httplib::Server*>(m_server_ptr);
        m_server_ptr = nullptr;
    }
}

void HttpServer::register_routes() {
    auto svr = static_cast<httplib::Server*>(m_server_ptr);

    // Health check
    svr->Get("/", [](const httplib::Request&, httplib::Response& res) {
        json body = {
            {"status", "ok"},
            {"engine", "EMBARK APU Engine"},
            {"architecture", "Zen 3 (8C/16T) + Vega 8 Wave64 RPM FP16"},
            {"version", "1.0.0"}
        };
        res.set_content(body.dump(), "application/json");
    });

    // OpenAI Models endpoint
    svr->Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        auto models = m_session_mgr.get_registered_models();
        json data_array = json::array();

        for (const auto& m : models) {
            auto state = m_session_mgr.get_model_state(m);
            std::string state_str = (state == ModelState::RESIDENT) ? "resident" :
                                    (state == ModelState::SWAPPED_IN) ? "swapped_in" : "available_on_demand";
            data_array.push_back({
                {"id", m},
                {"object", "model"},
                {"created", (uint64_t)std::time(nullptr)},
                {"owned_by", "embark"},
                {"state", state_str}
            });
        }

        json response = {
            {"object", "list"},
            {"data", data_array}
        };
        res.set_content(response.dump(), "application/json");
    });

    // OpenAI Chat Completions endpoint
    svr->Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", {{"message", "Invalid JSON"}}}}).dump(), "application/json");
            return;
        }

        std::string model = body.value("model", "groot-19b");
        bool stream = body.value("stream", false);

        SamplingParams params;
        params.temperature = body.value("temperature", 0.7f);
        params.top_p = body.value("top_p", 0.8f);
        params.max_tokens = body.value("max_tokens", 2048);

        // Parse messages
        std::vector<ChatMessage> chat_msgs;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& msg : body["messages"]) {
                std::string role = msg.value("role", "user");
                std::string content = msg.value("content", "");
                chat_msgs.push_back({role, content});
            }
        }

        std::string cmpl_id = "chatcmpl-" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream; charset=utf-8",
                [this, model, chat_msgs, params, cmpl_id](size_t offset, httplib::DataSink& sink) {
                    // Acquire concurrency slot BEFORE sending any bytes
                    SessionManager::SlotGuard guard(m_session_mgr, model);
                    if (!guard.acquired) {
                        sink.done();
                        return true;
                    }

                    // Send role chunk
                    json initial_chunk = {
                        {"id", cmpl_id},
                        {"object", "chat.completion.chunk"},
                        {"created", (uint64_t)std::time(nullptr)},
                        {"model", model},
                        {"choices", {{
                            {"delta", {{"role", "assistant"}}},
                            {"index", 0},
                            {"finish_reason", nullptr}
                        }}}
                    };
                    std::string init_line = "data: " + initial_chunk.dump() + "\n\n";
                    sink.write(init_line.data(), init_line.size());

                    // Stream tokens
                    std::string formatted_prompt = m_session_mgr.format_prompt(model, chat_msgs);
                    if (formatted_prompt.empty()) {
                        // Fallback: raw last user message
                        formatted_prompt = chat_msgs.empty() ? "Hello" : chat_msgs.back().content;
                    }
                    m_session_mgr.generate_stream(
                        model,
                        formatted_prompt,
                        params,
                        [&](const std::string& token_piece) {
                            json chunk = {
                                {"id", cmpl_id},
                                {"object", "chat.completion.chunk"},
                                {"created", (uint64_t)std::time(nullptr)},
                                {"model", model},
                                {"choices", {{
                                    {"delta", {{"content", token_piece}}},
                                    {"index", 0},
                                    {"finish_reason", nullptr}
                                }}}
                            };
                            std::string line = "data: " + chunk.dump() + "\n\n";
                            return sink.write(line.data(), line.size());
                        }
                    );

                    // End chunk
                    json finish_chunk = {
                        {"id", cmpl_id},
                        {"object", "chat.completion.chunk"},
                        {"created", (uint64_t)std::time(nullptr)},
                        {"model", model},
                        {"choices", {{
                            {"delta", json::object()},
                            {"index", 0},
                            {"finish_reason", "stop"}
                        }}}
                    };
                    std::string finish_line = "data: " + finish_chunk.dump() + "\n\n";
                    sink.write(finish_line.data(), finish_line.size());

                    std::string done_line = "data: [DONE]\n\n";
                    sink.write(done_line.data(), done_line.size());
                    sink.done();
                    return true;
                }
            );
        } else {
            SessionManager::SlotGuard guard(m_session_mgr, model);
            if (!guard.acquired) {
                res.status = 503;
                res.set_content(json({{"error", {{"message", "Model busy or unavailable"}}}}).dump(), "application/json");
                return;
            }

            std::string formatted_prompt = m_session_mgr.format_prompt(model, chat_msgs);
            if (formatted_prompt.empty()) {
                formatted_prompt = chat_msgs.empty() ? "Hello" : chat_msgs.back().content;
            }
            std::string content = m_session_mgr.generate_completion(
                model,
                formatted_prompt,
                params
            );

            json response = {
                {"id", cmpl_id},
                {"object", "chat.completion"},
                {"created", (uint64_t)std::time(nullptr)},
                {"model", model},
                {"choices", {{
                    {"message", {
                        {"role", "assistant"},
                        {"content", content}
                    }},
                    {"index", 0},
                    {"finish_reason", "stop"}
                }}},
                {"usage", {
                    {"prompt_tokens", 16},
                    {"completion_tokens", (int)content.length() / 4},
                    {"total_tokens", 16 + (int)content.length() / 4}
                }}
            };
            res.set_content(response.dump(), "application/json");
        }
    });
}

void HttpServer::start_blocking() {
    auto svr = static_cast<httplib::Server*>(m_server_ptr);
    std::cout << "================================================================================\n";
    std::cout << "  EMBARK Pure C++ OpenAI Server listening on http://" << m_host << ":" << m_port << "\n";
    std::cout << "  Compatible with DeepSeek Harness (llama-cpp-local), curl, and OpenAI SDK\n";
    std::cout << "================================================================================\n\n";
    svr->listen(m_host, m_port);
}

void HttpServer::stop() {
    auto svr = static_cast<httplib::Server*>(m_server_ptr);
    if (svr && svr->is_running()) {
        svr->stop();
    }
}

} // namespace embark
