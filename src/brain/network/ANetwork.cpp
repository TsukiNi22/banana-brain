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
##  @file ANetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/network/ANetwork.hpp"
#include <cmath>

_cold void brain::ANetwork::init_(void)
{
    /* Nothing */
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
