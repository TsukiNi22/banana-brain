/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 12/11/2026 by @author Tsukini

File Name:
##  @file Metrics.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/Metrics.hpp"
#include <filesystem>
#include <ostream>

_cold brain::FType brain::Metrics::accuracy(void) const
{
    /* Nothing */
    return {};
}

_cold brain::FType brain::Metrics::accuracy(const brain::UIType expected) const
{
    /* Nothing */
    return {};
}

_cold brain::FType brain::Metrics::precision(const brain::UIType c) const
{
    /* Nothing */
    return {};
}

_cold brain::FType brain::Metrics::recall(const brain::UIType c) const
{
    /* Nothing */
    return {};
}

_cold brain::FType brain::Metrics::f1(const brain::UIType c) const
{
    /* Nothing */
    return {};
}

_cold void brain::Metrics::csv(const std::filesystem::path& path) const
{
    /* Nothing */
}

_cold std::ostream& brain::operator<<(std::ostream& os, const brain::Metrics& metrics)
{
    /* Nothing */
    return os;
}
