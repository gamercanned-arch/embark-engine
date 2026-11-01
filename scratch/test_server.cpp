#include <iostream>
#include <thread>
#include <chrono>
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

int main() {
    httplib::Server svr;

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"object", "list"},
            {"data", {
                {{"id", "groot-19b"}, {"object", "model"}, {"owned_by", "embark"}},
                {{"id", "lfm3-vl-2b"}, {"object", "model"}, {"owned_by", "embark"}},
                {{"id", "lfm2-350m"}, {"object", "model"}, {"owned_by", "embark"}},
                {{"id", "ornith-9b"}, {"object", "model"}, {"owned_by", "embark"}}
            }}
        };
        res.set_content(response.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body);
        bool stream = body.value("stream", false);
        std::string model = body.value("model", "groot-19b");

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream; charset=utf-8",
                [](size_t offset, httplib::DataSink& sink) {
                    std::vector<std::string> tokens = {"I", " am", " Groot", "!", " Ready", " for", " tactical", " radio", " comms."};
                    for (const auto& tok : tokens) {
                        json chunk = {
                            {"id", "chatcmpl-test"},
                            {"object", "chat.completion.chunk"},
                            {"choices", {
                                {{"delta", {{"content", tok}}}, {"index", 0}, {"finish_reason", nullptr}}
                            }}
                        };
                        std::string line = "data: " + chunk.dump() + "\n\n";
                        sink.write(line.data(), line.size());
                        std::this_thread::sleep_for(std::chrono::milliseconds(32)); // ~31 tok/s
                    }
                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                }
            );
        } else {
            json resp = {
                {"id", "chatcmpl-test"},
                {"object", "chat.completion"},
                {"choices", {
                    {{"message", {{"role", "assistant"}, {"content", "I am Groot! Ready for tactical radio comms."}}}, {"finish_reason", "stop"}}
                }}
            };
            res.set_content(resp.dump(), "application/json");
        }
    });

    std::cout << "[+] HTTP Server listening on http://127.0.0.1:8089...\n";
    svr.listen("127.0.0.1", 8089);
    return 0;
}
