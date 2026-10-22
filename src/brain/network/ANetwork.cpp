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
#define _System
#include <utils/utils.hpp>
#include "brain/network/ANetwork.hpp"
#include "brain/tools/activation.hpp"
#include "brain/NetworkDefine.hpp"
#include "brain/tools/random.hpp"
#include "brain/tools/bytes.hpp"
#include <iostream>
#include <mutex>
#include <cmath>
#include <map>

// One gpu handler per mode for the whole program: finding a gpu (loading the drivers) takes a few hundred ms, the networks share the result (nullptr: no gpu, ram)
_cold static std::shared_ptr<utils::system::IGpu> sharedGpu(const utils::system::MemoryMode mode)
{
    static std::mutex lock;
    static std::map<utils::system::MemoryMode, std::shared_ptr<utils::system::IGpu>> handlers;
    std::lock_guard guard(lock);
    auto it = handlers.find(mode); // xstyle: ignore CPP-AUTO

    if (it == handlers.end()) it = handlers.emplace(mode, utils::system::make_shared_gpu(mode)).first; // an error (GpuUnavailable...) is not kept
    return it->second;
}

brain::ANetwork::ANetwork(const utils::system::MemoryMode mode)
: _gpu{sharedGpu(mode)}, _mode{mode}
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

_hot void brain::ANetwork::inputs(const brain::Values& inputs)
{
    // check inputs
    if (inputs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no inputs given");
    } else if (this->_ioSize.first != 0 && inputs.size() != this->_ioSize.first) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "given " + std::to_string(inputs.size()) + ", expected " + std::to_string(this->_ioSize.first));
    }
    this->_inputs = inputs;
}

_hot brain::Values brain::ANetwork::apply(const brain::Values& inputs) const
{
    // check inputs
    if (inputs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no inputs given");
    } else if (this->_ioSize.first != 0 && inputs.size() != this->_ioSize.first) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "given " + std::to_string(inputs.size()) + ", expected " + std::to_string(this->_ioSize.first));
    }
    return this->compute_(inputs);
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

_cold void brain::ANetwork::serializeHeader_(brain::Bytes& out, const std::uint8_t kind) const
{
    brain::tools::put<std::uint8_t>(out, static_cast<std::uint8_t>(BRAIN_MAGIC));
    brain::tools::put<std::uint8_t>(out, kind);
    brain::tools::put<std::uint8_t>(out, this->serializeActivation_());
    brain::tools::put<std::uint64_t>(out, this->_ioSize.first);
    brain::tools::put<std::uint64_t>(out, this->_ioSize.second);
    brain::tools::put<brain::FType>(out, this->_rand.first.first);
    brain::tools::put<brain::FType>(out, this->_rand.first.second);
    brain::tools::put<std::uint8_t>(out, this->_rand.second ? static_cast<std::uint8_t>(BRAIN_ACTIVATION_CUSTOM) : static_cast<std::uint8_t>(BRAIN_ACTIVATION_NONE));
}

_cold brain::UIType brain::ANetwork::deserializeHeader_(const brain::Bytes& data, const std::uint8_t kind)
{
    brain::UIType offset = 0;

    // check the magic bytes
    if (data.size() < 3 || data[0] != static_cast<std::uint8_t>(BRAIN_MAGIC) || data[1] != kind) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "bad magic bytes for this kind of network");
    }
    offset = 2;

    // the common part is read before being applied: nothing is edited on an invalid file
    const std::uint8_t activation = brain::tools::get<std::uint8_t>(data, offset);
    const brain::UIType input = brain::tools::get<std::uint64_t>(data, offset);
    const brain::UIType output = brain::tools::get<std::uint64_t>(data, offset);
    const brain::FType min = brain::tools::get<brain::FType>(data, offset);
    const brain::FType max = brain::tools::get<brain::FType>(data, offset);
    const std::uint8_t random = brain::tools::get<std::uint8_t>(data, offset);
    if (!std::isfinite(min) || !std::isfinite(max) || min > max) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid random range");
    } else if (random != BRAIN_ACTIVATION_NONE && random != BRAIN_ACTIVATION_CUSTOM) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "unknown random mode " + std::to_string(random));
    }
    this->deserializeActivation_(activation);
    this->_ioSize = {input, output};
    this->_rand = {{min, max}, nullptr};
    if (random == BRAIN_ACTIVATION_CUSTOM) {
        std::cerr << utils::exception::WarningException(utils::exception::ExternalCode::CustomRandomRequired).formated() << std::endl;
    }
    return offset;
}

_hot brain::FType brain::ANetwork::random_(const brain::FType fallback) const
{
    brain::FType value;

    // the random function wins over the range (the seed given to it comes from the generator of the library)
    if (this->_rand.second) value = this->_rand.second(static_cast<brain::IType>(brain::tools::rand_unsigned_integer()));
    else value = this->_rand.first.first + brain::tools::rand_floating() * (this->_rand.first.second - this->_rand.first.first);
    if (!std::isfinite(value)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the random value should be a finite number");
    }
    return value == 0 ? fallback : value;
}

_hot brain::FType brain::ANetwork::activate_(const brain::FType value) const
{
    if (this->_activationFunction.second) return this->_activationFunction.second(value);
    switch (this->_activationFunction.first) {
        case brain::ActivationFunction::Sigmoid: return brain::tools::sigmoid(value);
        case brain::ActivationFunction::Linear:  return brain::tools::linear(value);
        case brain::ActivationFunction::ReLU:    return brain::tools::relu(value);
        default: return value;
    }
}

_cold void brain::ANetwork::size(const brain::UIType input, const brain::UIType output)
{
    const std::pair<brain::UIType, brain::UIType> previous = this->_ioSize;

    this->_ioSize = {input, output};
    try {
        this->resized_();
    } catch (...) {
        this->_ioSize = previous;
        throw;
    }
}
