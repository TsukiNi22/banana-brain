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
#define _Exception
#include <utils/utils.hpp>
#include "brain/network/BrainNetwork.hpp"
#include "brain/NetworkDefine.hpp"
#include <utility>
#include <cmath>

brain::BrainNetwork::BrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size)
: _size{size}, _neuron{neuron}
{
    // check inputs
    if (neuron == 0 || !(size.first > 0) || !(size.second > 0)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the number of neurons and the size of the space should be above 0");
    }
    /* Nothing */
}

_cold void brain::BrainNetwork::init_(void)
{
    /* Nothing */
}

_cold brain::Bytes brain::BrainNetwork::serialize(void) const
{
    /* Nothing */
    return {static_cast<std::uint8_t>(BRAIN_MAGIC), static_cast<std::uint8_t>(BRAIN_MAGIC_BRAIN)};
}

_cold void brain::BrainNetwork::deserialize(const brain::Bytes& data)
{
    // check the magic bytes
    if (data.size() < 2 || data[0] != static_cast<std::uint8_t>(BRAIN_MAGIC) || data[1] != static_cast<std::uint8_t>(BRAIN_MAGIC_BRAIN)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "bad magic bytes for a brain network");
    }
    /* Nothing */
}

_hot brain::Values brain::BrainNetwork::apply(void) const
{
    // check inputs
    if (this->_inputs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no inputs loaded (call inputs() first)");
    }
    /* Nothing */
    return {};
}

_hot void brain::BrainNetwork::backpropagation(const brain::FType coef)
{
    // check inputs
    if (!std::isfinite(coef)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficient should be a finite number");
    }
    /* Nothing */
}

_hot void brain::BrainNetwork::backpropagation(const brain::Values& coefs)
{
    // check inputs
    if (coefs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "no coefficients given");
    }
    /* Nothing */
}

_cold void brain::BrainNetwork::weightInit(const brain::FType fallback)
{
    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }
    /* Nothing */
}

_cold void brain::BrainNetwork::biasInit(const brain::FType fallback)
{
    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }
    /* Nothing */
}
