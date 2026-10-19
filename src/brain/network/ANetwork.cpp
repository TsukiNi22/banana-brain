/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 20/10/2026 by @author Tsukini

File Name:
##  @file ANetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/network/ANetwork.hpp"
#include "brain/NetworkDefine.hpp"
#include <iostream>
#include <cmath>

_cold void brain::ANetwork::init_(void)
{
    /* Nothing */
}

_cold std::uint8_t brain::ANetwork::serializeActivation_(void) const
{
    if (this->_activationFunction.second) return static_cast<std::uint8_t>(BRAIN_ACTIVATION_CUSTOM);
    return static_cast<std::uint8_t>(this->_activationFunction.first);
}

_cold void brain::ANetwork::deserializeActivation_(const std::uint8_t mode)
{
    switch (mode) {
        case BRAIN_ACTIVATION_NONE: this->internalActivationFunction(brain::ActivationFunction::None); break;
        case BRAIN_ACTIVATION_CUSTOM:
            this->internalActivationFunction(brain::ActivationFunction::None);
            std::cerr << utils::exception::WarningException(utils::exception::ExternalCode::CustomActivationRequired).formated() << std::endl;
            break;
        case static_cast<std::uint8_t>(brain::ActivationFunction::Sigmoid): this->internalActivationFunction(brain::ActivationFunction::Sigmoid); break;
        case static_cast<std::uint8_t>(brain::ActivationFunction::Linear):  this->internalActivationFunction(brain::ActivationFunction::Linear); break;
        case static_cast<std::uint8_t>(brain::ActivationFunction::ReLU):    this->internalActivationFunction(brain::ActivationFunction::ReLU); break;
        default: throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "unknown activation mode " + std::to_string(mode));
    }
}

_hot void brain::ANetwork::inputs(const brain::Values& inputs) const
{
    // check inputs
    if (inputs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no inputs given");
    }
    this->_inputs = inputs;
}

_cold void brain::ANetwork::rand(const brain::FType min, const brain::FType max)
{
    // check inputs
    if (!std::isfinite(min) || !std::isfinite(max) || min > max) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the random range should be finite and min should be under or equal to max");
    }
    this->_rand.first = {min, max};
}

_cold void brain::ANetwork::rand(const brain::RandFn& fn)
{
    // check inputs
    if (!fn) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the random function is empty");
    }
    this->_rand.second = fn;
}

_cold void brain::ANetwork::activationFunction(const brain::ActivationFn& fn)
{
    // check inputs
    if (!fn) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the activation function is empty");
    }
    this->_activationFunction = {brain::ActivationFunction::None, fn};
}

_cold void brain::ANetwork::internalActivationFunction(const brain::ActivationFunction fn)
{
    this->_activationFunction = {fn, nullptr};
}
