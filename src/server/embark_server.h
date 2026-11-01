#pragma once

#include <string>
#include <memory>
#include "embark_session_manager.h"

namespace embark {

class HttpServer {
public:
    HttpServer(SessionManager& session_mgr, const std::string& host = "127.0.0.1", int port = 8080);
    ~HttpServer();

    void start_blocking();
    void stop();

private:
    SessionManager& m_session_mgr;
    std::string m_host;
    int m_port;
    void* m_server_ptr = nullptr;

    void register_routes();
};

} // namespace embark
