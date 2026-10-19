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
#include <cmath>

brain::LayerNetwork::LayerNetwork(const brain::UIType layer, const brain::UIType neuron_per_layer)
: _neuronPerLayer{neuron_per_layer}, _layer{layer}
{
    // check inputs
    if (layer == 0 || neuron_per_layer == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the number of layers and of neurons per layer should be above 0");
    }
    /* Nothing */
}

_cold void brain::LayerNetwork::init_(void)
{
    /* Nothing */
}

_cold brain::Bytes brain::LayerNetwork::serialize(void) const
{
    /* Nothing */
    return {static_cast<std::uint8_t>(BRAIN_MAGIC), static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER), this->serializeActivation_()};
}

_cold void brain::LayerNetwork::deserialize(const brain::Bytes& data)
{
    // check the magic bytes
    if (data.size() < 3 || data[0] != static_cast<std::uint8_t>(BRAIN_MAGIC) || data[1] != static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "bad magic bytes for a layer network");
    }
    this->deserializeActivation_(data[2]);
    /* Nothing */
}

_hot brain::Values brain::LayerNetwork::apply(void) const
{
    // check inputs
    if (this->_inputs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no inputs loaded (call inputs() first)");
    }
    /* Nothing */
    return {};
}

_hot void brain::LayerNetwork::backpropagation(const brain::FType coef)
{
    // check inputs
    if (!std::isfinite(coef)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficient should be a finite number");
    }
    /* Nothing */
}

_hot void brain::LayerNetwork::backpropagation(const brain::Values& coefs)
{
    // check inputs
    if (coefs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "no coefficients given");
    } else if (this->_ioSize.second != 0 && coefs.size() != this->_ioSize.second) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "given " + std::to_string(coefs.size()) + ", expected " + std::to_string(this->_ioSize.second));
    }
    /* Nothing */
}

_cold void brain::LayerNetwork::weightInit(const brain::FType fallback)
{
    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }
    /* Nothing */
}

_cold void brain::LayerNetwork::biasInit(const brain::FType fallback)
{
    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }
    /* Nothing */
}
