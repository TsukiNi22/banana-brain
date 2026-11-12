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
##  @file LayerNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/LayerNetwork.hpp"
#include <utility>
#include <memory>

brain::LayerNetwork::LayerNetwork(const brain::Layers& layers, const brain::LayerOptions& options)
: _layers{layers}, _activations{options.activations}, _dropout{options.dropout}
{
    /* Nothing */
}

_cold void brain::LayerNetwork::init_(void)
{
    /* Nothing */
}

_cold brain::Bytes brain::LayerNetwork::serialize(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::LayerNetwork::deserialize(const brain::Bytes& data)
{
    /* Nothing */
}

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

_cold void brain::LayerNetwork::weightInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::biasInit(const brain::FType fallback)
{
    /* Nothing */
}

_hot void brain::LayerNetwork::applyGradients(void)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::activation(const brain::UIType layer, const brain::Activation& fn)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::dropout(const brain::UIType layer, const brain::FType probability)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::dropout(const brain::FType probability)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::addLayer(const brain::UIType index, const brain::UIType neurons)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::removeLayer(const brain::UIType index)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::resize(const brain::UIType layer, const brain::UIType neurons)
{
    /* Nothing */
}

_cold void brain::LayerNetwork::prune(const brain::FType influence, const brain::PruneTarget target)
{
    /* Nothing */
}

_cold std::unique_ptr<brain::INetwork> brain::LayerNetwork::clone(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::LayerNetwork::reset(void)
{
    /* Nothing */
}

_cold std::pair<brain::UIType, brain::UIType> brain::LayerNetwork::neurons(void) const
{
    /* Nothing */
    return {};
}
