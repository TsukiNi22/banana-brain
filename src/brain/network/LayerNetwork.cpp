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
##  @file LayerNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/network/LayerNetwork.hpp"
#include "brain/NetworkDefine.hpp"
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

_cold brain::Bytes brain::LayerNetwork::serialize(void) const
{
    /* Nothing */
    return {static_cast<std::uint8_t>(BRAIN_MAGIC), static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER)};
}

_cold void brain::LayerNetwork::deserialize(const brain::Bytes& data)
{
    // check the magic bytes
    if (data.size() < 2 || data[0] != static_cast<std::uint8_t>(BRAIN_MAGIC) || data[1] != static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "bad magic bytes for a layer network");
    }
    /* Nothing */
}

_hot brain::Values brain::LayerNetwork::apply(void) const
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
