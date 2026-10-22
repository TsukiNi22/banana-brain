/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 22/10/2026 by @author Tsukini

File Name:
##  @file LayerNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#define _System
#include <utils/utils.hpp>
#include "brain/network/LayerNetwork.hpp"
#include "brain/NetworkDefine.hpp"
#include "brain/tools/bytes.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cmath>

brain::LayerNetwork::LayerNetwork(const brain::Layers& layers, const utils::system::MemoryMode mode)
: ANetwork{mode}, _layers{layers}, _neurons{this->makeMemory_<brain::LayerNeuron>()}, _weights{this->makeMemory_<brain::FType>()}
{
    // check inputs
    if (layers.empty() || std::any_of(layers.begin(), layers.end(), [](const brain::UIType neurons) {return neurons == 0;})) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the network needs at least one layer and every layer needs at least one neuron");
    }
    this->init_();
}

_cold void brain::LayerNetwork::layout_(void)
{
    std::vector<brain::LayerNeuron> neurons;
    brain::UIType previous = this->_ioSize.first; // a neuron has one weight per output of the previous layer (the inputs for the first)
    brain::UIType weights = 0;
    brain::UIType product;

    // where the weights of each neuron start, layer after layer (a count that overflows the size is refused)
    for (const brain::UIType layer: this->_layers) {
        if (__builtin_mul_overflow(layer, previous, &product) || __builtin_add_overflow(weights, product, &product)) _unlikely {
            throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfMemory, "the weights of the layers overflow the size");
        }
        for (brain::UIType neuron = 0; neuron < layer; ++neuron)
            neurons.push_back({weights + neuron * previous, previous, 0.0});
        weights = product;
        previous = layer;
    }
    this->_neurons.allocate(neurons.size());
    this->_neurons.store(neurons);
    this->_weights.allocate(weights);
    this->_weights.store(std::vector<brain::FType>(weights, 1.0)); // 1.0 by default: weightInit gives the random ones
    this->_layoutInput = this->_ioSize.first;
}

_cold brain::Bytes brain::LayerNetwork::serialize(void) const
{
    const std::vector<brain::LayerNeuron> neurons = this->_neurons.toVector();
    const std::vector<brain::FType> weights = this->_weights.toVector();
    brain::Bytes out;

    // common part, then the layers, the neurons and the weights
    this->serializeHeader_(out, static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER));
    brain::tools::put<std::uint64_t>(out, this->_layers.size());
    for (const brain::UIType layer: this->_layers)
        brain::tools::put<std::uint64_t>(out, layer);
    brain::tools::put<std::uint64_t>(out, neurons.size());
    for (const brain::LayerNeuron& neuron: neurons) {
        brain::tools::put<std::uint64_t>(out, neuron.incoming);
        brain::tools::put<std::uint64_t>(out, neuron.count);
        brain::tools::put<brain::FType>(out, neuron.bias);
    }
    brain::tools::put<std::uint64_t>(out, weights.size());
    for (const brain::FType weight: weights)
        brain::tools::put<brain::FType>(out, weight);
    return out;
}

_cold void brain::LayerNetwork::deserialize(const brain::Bytes& data)
{
    brain::UIType offset = this->deserializeHeader_(data, static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER));
    brain::UIType count = brain::tools::get<std::uint64_t>(data, offset);

    // the layers
    if (!brain::tools::fits(data, offset, count, sizeof(std::uint64_t))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "truncated layers");
    }
    brain::Layers layers(count);
    for (brain::UIType& layer: layers)
        layer = brain::tools::get<std::uint64_t>(data, offset);
    if (layers.empty() || std::any_of(layers.begin(), layers.end(), [](const brain::UIType neurons) {return neurons == 0;})) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the network needs at least one layer and every layer needs at least one neuron");
    }

    // the neurons
    count = brain::tools::get<std::uint64_t>(data, offset);
    if (!brain::tools::fits(data, offset, count, 2 * sizeof(std::uint64_t) + sizeof(brain::FType))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "truncated neurons");
    }
    std::vector<brain::LayerNeuron> neurons(count);
    for (brain::LayerNeuron& neuron: neurons) {
        neuron.incoming = brain::tools::get<std::uint64_t>(data, offset);
        neuron.count = brain::tools::get<std::uint64_t>(data, offset);
        neuron.bias = brain::tools::get<brain::FType>(data, offset);
    }

    // the weights
    count = brain::tools::get<std::uint64_t>(data, offset);
    if (!brain::tools::fits(data, offset, count, sizeof(brain::FType))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "truncated weights");
    }
    std::vector<brain::FType> weights(count);
    for (brain::FType& weight: weights)
        weight = brain::tools::get<brain::FType>(data, offset);
    if (offset != data.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "unexpected data after the weights");
    }

    // everything is valid: the network is replaced
    this->_layers = layers;
    this->_neurons.allocate(neurons.size());
    this->_neurons.store(neurons);
    this->_weights.allocate(weights.size());
    this->_weights.store(weights);
    this->_layoutInput = this->_ioSize.first;
}

_hot brain::Values brain::LayerNetwork::compute_(const brain::Values& inputs) const
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::LayerNetwork::apply(void)
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
    std::vector<brain::FType> weights;

    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }

    // a random value for every weight
    weights = this->_weights.toVector();
    for (brain::FType& weight: weights)
        weight = this->random_(fallback);
    this->_weights.store(weights);
}

_cold void brain::LayerNetwork::biasInit(const brain::FType fallback)
{
    std::vector<brain::LayerNeuron> neurons;

    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }

    // a random value for every bias
    neurons = this->_neurons.toVector();
    for (brain::LayerNeuron& neuron: neurons)
        neuron.bias = this->random_(fallback);
    this->_neurons.store(neurons);
}
