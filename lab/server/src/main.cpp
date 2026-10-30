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
##  @file main.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Arguments
#define _CustomType
#define _Exception
#define _Network
#define _System
#include <utils/utils.hpp>
#include "LabServer.hpp"
#include <optional>
#include <iostream>
#include <string>

// A port: digits only, at most 65535
static std::optional<std::string> PortHook(const std::string& option)
{
    if (!option.empty() && option.size() <= 5 && option.find_first_not_of("0123456789") == std::string::npos && std::stoul(option) <= 65535) return std::nullopt;
    return "Invalid port, should be a number up to 65535, but got: " + option;
}

// Any path (the server says it when the page is missing)
static std::optional<std::string> PathHook(const std::string&)
{
    return std::nullopt;
}

// lab_server [-p port] [--page path]: serve the lab page and run its networks with the library (the gpu when there is one)
int main(int argc, char** argv)
{
    utils::arguments::ArgParser parser = utils::arguments::ArgParser("lab_server", "Engine of the banana-brain lab page: serves the page and runs its networks with the library");
    std::uint16_t port = 8765;
    std::string page = "network-lab.html";

    try {
        // Setup the usages and the flags
        parser.setDefaultUsage();
        parser.setFlag("port",
            {"p", "", "port", "LAB_PORT"},
            {
                {"number", true, PortHook}
            },
            "Port of the server (default: 8765)"
        );
        parser.setFlag("page",
            {"", "", "page", "LAB_PAGE"},
            {
                {"path", true, PathHook}
            },
            "Path of the lab page served on / (default: network-lab.html)"
        );

        // Extract the settings
        const utils::arguments::ParsedUsages usages = parser.parse(argc, argv);
        for (const auto& [id, _, options]: usages.front().arguments) {
            if (id == "port") port = static_cast<std::uint16_t>(std::stoul(options.front()));
            else if (id == "page") page = options.front();
        }
        lab::LabServer server(page);
        server.run(port);
    } catch (const utils::exception::IException& e) {
        if (e.isNone() && e.getCode() == utils::exception::InternalCode::Exit) return OK;
        std::cerr << e.formated() << std::endl;
        return KO;
    }
    return OK;
}
