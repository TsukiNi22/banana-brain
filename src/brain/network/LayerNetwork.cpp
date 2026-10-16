/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 17/10/2026 by @author Tsukini

File Name:
##  @file LayerNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/LayerNetwork.hpp"
#include <utility>

brain::LayerNetwork::LayerNetwork(const brain::UIType layer, const brain::UIType neuron_per_layer)
: _neuronPerLayer{neuron_per_layer}, _layer{layer}
{
    /* Nothing */
}

_cold void brain::LayerNetwork::init_(void)
{
    /* Nothing */
}

/* execution */
_hot brain::Values brain::LayerNetwork::apply(void)
{
    /* Nothing */
    return {};
}

_hot void brain::LayerNetwork::backpropagation(const brain::FType coef)
{
    /* Nothing */
}

_hot void brain::LayerNetwork::backpropagation(const brain::Values& coefs)
{
    /* Nothing */
}

/* setup */
_cold void brain::LayerNetwork::weightInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::biasInit(const brain::FType fallback)
{
    /* Nothing */
}

/* getter */
_cold std::pair<brain::UIType, brain::UIType> brain::LayerNetwork::neurons(void) const
{
    /* Nothing */
    return {};
}
