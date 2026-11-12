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
##  @file ANetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/ANetwork.hpp"
#include <cstdint>
#include <memory>

_cold void brain::ANetwork::serializeHeader_(brain::Bytes& out, const std::uint8_t kind) const
{
    /* Nothing */
}

_cold brain::UIType brain::ANetwork::deserializeHeader_(const brain::Bytes& data, const std::uint8_t kind)
{
    /* Nothing */
    return {};
}

_hot brain::Value brain::ANetwork::activate_(const brain::Value value) const
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::ANetwork::activate_(const brain::Values& values) const
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::ANetwork::derivative_(const brain::Value output) const
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::ANetwork::derivative_(const brain::Values& outputs) const
{
    /* Nothing */
    return {};
}

_cold brain::FType brain::ANetwork::random_(const brain::FType fallback, const brain::UIType fan_in, const brain::UIType fan_out) const
{
    /* Nothing */
    return {};
}

_hot void brain::ANetwork::update_(brain::FType& weight, const brain::UIType index, const brain::FType gradient)
{
    /* Nothing */
}

_cold void brain::ANetwork::init_(void)
{
    /* Nothing */
}

_hot void brain::ANetwork::inputs(const brain::Values& inputs)
{
    /* Nothing */
}

_hot brain::Values brain::ANetwork::apply(const brain::Values& inputs) const
{
    /* Nothing */
    return {};
}

_hot brain::Dataset brain::ANetwork::applyAll(const brain::Dataset& inputs) const
{
    /* Nothing */
    return {};
}

_cold void brain::ANetwork::size(const brain::UIType input, const brain::UIType output)
{
    /* Nothing */
}

_cold void brain::ANetwork::input(const brain::UIType input)
{
    /* Nothing */
}

_cold void brain::ANetwork::output(const brain::UIType output)
{
    /* Nothing */
}

_cold void brain::ANetwork::rand(const brain::FType min, const brain::FType max)
{
    /* Nothing */
}

_cold void brain::ANetwork::rand(const brain::Rand& rand)
{
    /* Nothing */
}

_cold void brain::ANetwork::activation(const brain::Activation& fn)
{
    /* Nothing */
}

_cold void brain::ANetwork::backend(std::shared_ptr<brain::IBackend> backend)
{
    /* Nothing */
}
