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
##  @file LabServer.hpp

File Description:
##  The engine of the lab page on the host: the page talks to it in
##  HTTP (JSON bodies ended by an empty line), it runs the library
\**************************************************************/

#ifndef LABSERVER_H
    #define LABSERVER_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _CustomType
    #define _Network
    #define _System
    #include <utils/utils.hpp>      // utils::network::Server, utils::network::Payload (every macro of the program before its first include)
    #include "BrainSession.hpp"     // lab::BrainSession
    #include <nlohmann/json.hpp>    // nlohmann::json
    #include <unordered_map>        // std::unordered_map
    #include <cstdint>              // std::uint16_t
    #include <memory>               // std::unique_ptr
    #include <string>               // std::string

namespace lab { // namespace start
//----------------------------------------------------------------//
/* STRUCT */

// A request of a client: its line and its body (the next payload when it has one)
struct Request {
    std::string method;
    std::string path;
    std::size_t length = 0; // Content-Length: a body follows
};

//----------------------------------------------------------------//
/* CLASS */

// HTTP on the server of libutils (payloads separated by an empty line), the routes of the page
class LabServer {
    private:
        std::string _page; // the lab page served on /
        std::unordered_map<int, lab::Request> _pending; // <fd, request waiting for its body>
        std::unordered_map<int, std::unique_ptr<lab::BrainSession>> _sessions; // <id, network of the page>
        int _next = 1; // id of the next session
        nlohmann::json _info; // what /api/info answers: the gpu found at the start

        // ---------- Pre-Function -------- //
        std::string answer_(const lab::Request& request, const std::string& body); // the response of a request (whole HTTP message)
        nlohmann::json route_(const std::string& path, const nlohmann::json& body); // the JSON of an api route
        static std::string reply_(const int status, const std::string& type, const std::string& body); // HTTP response with the CORS header (a page from file:// can call it)

    public:
        // ---------- Pre-Function -------- //
        void run(const std::uint16_t port); // serve until killed

        // ---------- Constructor --------- //
        LabServer(const std::string& page); // path of the lab page
};

} // namespace end
#endif /* LABSERVER_H */
