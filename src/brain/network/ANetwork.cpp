/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 10/11/2026 by @author Tsukini

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

_cold void brain::ANetwork::serializeHeader_(brain::Bytes& out, const std::uint8_t kind) const
{
    /* Nothing */
}

_cold brain::UIType brain::ANetwork::deserializeHeader_(const brain::Bytes& data, const std::uint8_t kind)
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::ANetwork::activate_(const brain::FType value) const
{
    /* Nothing */
    return {};
}

_cold brain::FType brain::ANetwork::random_(const brain::FType fallback) const
{
    /* Nothing */
    return {};
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

_cold void brain::ANetwork::rand(const brain::RandFn& fn)
{
    /* Nothing */
}

_cold void brain::ANetwork::activationFunction(const brain::ActivationFn& fn)
{
    /* Nothing */
}

_cold void brain::ANetwork::internalActivationFunction(const brain::ActivationFunction fn)
{
    /* Nothing */
}
