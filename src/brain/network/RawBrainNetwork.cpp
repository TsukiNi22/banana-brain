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
##  @file RawBrainNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/RawBrainNetwork.hpp"
#include <utility>
#include <memory>

brain::RawBrainNetwork::RawBrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size)
: brain::BrainNetwork(neuron), _space{size}
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

_cold std::unique_ptr<brain::BrainNetwork> brain::RawBrainNetwork::reduce(void) const
{
    /* Nothing */
    return {};
}
