/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 19/10/2026 by @author Tsukini

File Name:
##  @file BrainNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/BrainNetwork.hpp"
#include <utility>

brain::BrainNetwork::BrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size)
: _size{size}, _neuron{neuron}
{
    /* Nothing */
}

_cold void brain::BrainNetwork::init_(void)
{
    /* Nothing */
}

_hot brain::Values brain::BrainNetwork::apply(void) const
{
    /* Nothing */
    return {};
}

_hot void brain::BrainNetwork::backpropagation(const brain::FType coef)
{
    /* Nothing */
}

_hot void brain::BrainNetwork::backpropagation(const brain::Values& coefs)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::weightInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::biasInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold std::pair<brain::UIType, brain::UIType> brain::BrainNetwork::neurons(void) const
{
    /* Nothing */
    return {};
}
