/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 25/10/2026 by @author Tsukini

File Name:
##  @file LabServer.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _CustomType
#define _Exception
#define _Network
#define _System
#include <utils/utils.hpp>
#include "brain/network/RawBrainNetwork.hpp"
#include "LabServer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

lab::LabServer::LabServer(const std::string& page)
: _page{page}
{
    // the gpu is found and its kernels are compiled now (a few seconds), not by the first request of the page
    brain::RawBrainNetwork probe(6, {1.0, 1.0}, utils::system::MemoryMode::Auto);

    probe.size(2, 1);
    probe.weightInit();
    (void)probe.apply({0.5, 0.5});
    this->_info["engine"] = "server";
    this->_info["library"] = "banana-brain";
    this->_info["gpu"] = probe.onGpu() ? probe.gpuName() : "";
}

std::string lab::LabServer::reply_(const int status, const std::string& type, const std::string& body)
{
    std::ostringstream out;

    out << "HTTP/1.1 " << status << (status == 200 ? " OK" : (status == 404 ? " Not Found" : " Bad Request")) << "\r\n"
        << "Content-Type: " << type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Headers: Content-Type\r\n"
        << "Connection: keep-alive\r\n\r\n" << body;
    return out.str();
}

nlohmann::json lab::LabServer::route_(const std::string& path, const nlohmann::json& body)
{
    nlohmann::json out;

    // a network of the page: built from its options and its task
    if (path == "/api/create") {
        const int id = this->_next++;

        this->_sessions[id] = std::make_unique<lab::BrainSession>(body["options"], lab::BrainSession::task(body["task"]));
        this->_sessions[id]->points(body);
        if (body.contains("previous") && body["previous"].is_number()) this->_sessions.erase(body["previous"].get<int>()); // the network it replaces
        nlohmann::json asked = body;

        asked["positions"] = true; // the page keeps them: the other answers do not send them again
        out = this->_sessions[id]->snapshot(asked);
        out["id"] = id;
        return out;
    }

    // rounds of training then the snapshot, or only the snapshot (another sample, another step)
    if (path == "/api/train" || path == "/api/snapshot") {
        const auto found = this->_sessions.find(body.value("id", 0));

        if (found == this->_sessions.end()) throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "unknown network (created again by the page)");
        found->second->points(body);
        return path == "/api/train" ? found->second->train(body) : found->second->snapshot(body);
    }

    // many networks of the same options (batch mode of the page)
    if (path == "/api/batch") return lab::BrainSession::batch(body);
    throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "unknown route " + path);
}

std::string lab::LabServer::answer_(const lab::Request& request, const std::string& body)
{
    // the page itself, then the information of the engine, then the api
    if (request.method == "OPTIONS") return reply_(200, "text/plain", "");
    if (request.method == "GET" && (request.path == "/" || request.path.rfind("/?", 0) == 0)) {
        std::ifstream file(this->_page);
        std::stringstream content;

        if (!file) return reply_(404, "text/plain", "lab page not found: " + this->_page);
        content << file.rdbuf();
        return reply_(200, "text/html; charset=utf-8", content.str());
    }
    if (request.method == "GET" && request.path == "/api/info") return reply_(200, "application/json", this->_info.dump());
    if (request.method != "POST") return reply_(404, "text/plain", "not found");
    try {
        return reply_(200, "application/json", this->route_(request.path, nlohmann::json::parse(body)).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
    } catch (const utils::exception::IException& e) {
        return reply_(400, "application/json", nlohmann::json{{"error", e.info()}}.dump());
    } catch (const std::exception& e) {
        return reply_(400, "application/json", nlohmann::json{{"error", e.what()}}.dump());
    }
}

void lab::LabServer::run(const std::uint16_t port)
{
    std::shared_ptr<utils::network::ISocket> socket = std::make_shared<utils::network::TCPSocket>();
    utils::network::Address address;

    // an HTTP head ends by an empty line; the page ends its body by an empty line too (JSON never holds one)
    socket->setPayloadSeparator(std::string("\r\n\r\n"));
    socket->setOverflow(0);
    address.port = port;
    utils::network::Server server(socket, address);
    server.start();
    std::cout << "banana-brain lab: http://localhost:" << port << "/" << std::endl;
    while (server.getStatus() != utils::network::Status::Down) {
        server.join();
        for (const std::pair<const int, utils::network::Payloads>& client: server.listen()) {
            for (const utils::network::Payload& payload: client.second) {
                const auto pending = this->_pending.find(client.first);

                // the body of a request already read
                if (pending != this->_pending.end()) {
                    const lab::Request request = pending->second;

                    this->_pending.erase(pending);
                    server.send(client.first, this->answer_(request, payload));
                    continue;
                }

                // a new request: its first line, then the length of its body
                lab::Request request;
                std::istringstream head(payload);
                std::string line;

                head >> request.method >> request.path;
                while (std::getline(head, line)) {
                    if (line.rfind("Content-Length:", 0) == 0 || line.rfind("content-length:", 0) == 0) request.length = std::stoul(line.substr(15));
                }
                if (request.length > 0) this->_pending[client.first] = request;
                else server.send(client.first, this->answer_(request, ""));
            }
        }
    }
}
