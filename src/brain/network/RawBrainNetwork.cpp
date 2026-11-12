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
##  @file RawBrainNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/RawBrainNetwork.hpp"
#include <cstdint>
#include <memory>
#include <vector>

brain::RawBrainNetwork::RawBrainNetwork(const brain::UIType neuron, const brain::BrainOptions& options)
: brain::BrainNetwork(neuron, options)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::init_(void)
{
    /* Nothing */
}

_cold brain::Bytes brain::RawBrainNetwork::serialize(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::RawBrainNetwork::deserialize(const brain::Bytes& data)
{
    /* Nothing */
}

_hot brain::Values brain::RawBrainNetwork::apply(void)
{
    /* Nothing */
    return {};
}

_hot void brain::RawBrainNetwork::backpropagation(const brain::FType coef)
{
    /* Nothing */
}

_hot void brain::RawBrainNetwork::backpropagation(const brain::Values& coefs)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::weightInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::biasInit(const brain::FType fallback)
{
    /* Nothing */
}

_hot void brain::RawBrainNetwork::applyGradients(void)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::dimension(const std::uint16_t dimension)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::space(const std::vector<brain::FType>& space)
{
    /* Nothing */
}

_cold std::unique_ptr<brain::BrainNetwork> brain::RawBrainNetwork::reduce(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::RawBrainNetwork::relink(void)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::revive(void)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::grow(const brain::UIType count)
{
    /* Nothing */
}

_cold void brain::RawBrainNetwork::prune(const brain::FType influence, const brain::PruneTarget target)
{
    /* Nothing */
}

_cold std::unique_ptr<brain::INetwork> brain::RawBrainNetwork::clone(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::RawBrainNetwork::reset(void)
{
    /* Nothing */
}
