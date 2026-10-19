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
##  @file NeuralNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/network/Network.hpp"
#include "brain/NeuralNetwork.hpp"
#include "brain/tools/loss.hpp"
#include "brain/NetworkDefine.hpp"
#include <filesystem>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <cctype>
#include <string>
#include <utility>
#include <memory>
#include <cmath>

_hot void brain::NeuralNetwork::decay_(void)
{
    if (this->_decay == 0) return;
    if (++this->_steps % this->_decay == 0) this->_learningRateDecayed *= this->_decayCoef.second;
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Layered>(const brain::UIType layer, const brain::UIType neuron_per_layer)
{
    this->clear();
    this->_network = std::make_unique<brain::LayerNetwork>(layer, neuron_per_layer);
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Brain>(const brain::UIType neurons, const std::pair<brain::FType, brain::FType>& size)
{
    this->clear();
    this->_network = std::make_unique<brain::BrainNetwork>(neurons, size);
}

_cold void brain::NeuralNetwork::learningRate(const brain::FType learning_rate, const brain::UIType learning_rate_decay, const std::pair<bool, brain::FType>& coef)
{
    // setup
    this->_learningRate = learning_rate;
    this->_decay = learning_rate_decay;
    this->_decayCoef = coef;
    this->_learningRateDecayed = learning_rate;
    this->_steps = 0;

    // pre-compute
    if (this->_decay != 0 && !this->_decayCoef.first) {
        this->_decayCoef.second = std::pow(this->_decayCoef.second, 1.0 / static_cast<double>(this->_decay));
        this->_decay = 1;
    }
}

_cold void brain::NeuralNetwork::resetLearningRate(void)
{
    this->_learningRateDecayed = this->_learningRate;
    this->_steps = 0;
}

/* train (explicit instantiations, the templates are defined in this file, force the compiler to generate each version) */
template void brain::NeuralNetwork::train<false>(const brain::UIType, const brain::Dataset&, const brain::Dataset&);
template void brain::NeuralNetwork::train<true>(const brain::UIType, const brain::Dataset&, const brain::Dataset&);
template void brain::NeuralNetwork::train<false>(const brain::FType, const brain::Dataset&, const brain::Dataset&);
template void brain::NeuralNetwork::train<true>(const brain::FType, const brain::Dataset&, const brain::Dataset&);

template<bool mean>
_hot void brain::NeuralNetwork::train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs)
{
    std::size_t size = inputs.size();

    // check inputs
    if (inputs.size() != expected_outputs.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "both inputs and expected dataset must be of the same size");
    }

    // execution
    for (std::size_t n = 0; n < round; ++n) {
        for (std::size_t i = 0; i < size; ++i) {
            brain::Values values = this->exec(inputs[i]);
            this->backpropagation(this->loss<mean>(values, expected_outputs[i], this->_learningRateDecayed));
            this->decay_();
        }
    }
}

template<bool mean>
_hot void brain::NeuralNetwork::train(const brain::FType loss, const brain::Dataset& inputs, const brain::Dataset& expected_outputs)
{
    std::size_t size = inputs.size();
    std::pair<brain::Values, brain::FType> actual_loss = {};

    // check inputs
    if (inputs.size() != expected_outputs.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "both inputs and expected dataset must be of the same size");
    } else if (loss < 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the loss trigger should be above or equal to 0 (abs is used)");
    }

    // execution (the stop check uses the loss without the learning rate)
    for (std::size_t i = 0; std::abs(actual_loss.second) > loss || i == 0; ++i) {
        brain::Values values = this->exec(inputs[i % size]);
        actual_loss = this->loss(values, expected_outputs[i % size]);
        if constexpr (mean) {
            this->backpropagation(actual_loss.second * this->_learningRateDecayed);
        } else {
            if (this->_learningRateDecayed != 1.0)
                for (brain::FType& value: actual_loss.first) value *= this->_learningRateDecayed;
            this->backpropagation(actual_loss.first);
        }
        this->decay_();
    }
}

_hot brain::FType brain::NeuralNetwork::loss(const brain::FType output, const brain::FType expected) const
{
    if (this->_lossFunction.second) return this->_lossFunction.second(output, expected);
    switch (this->_lossFunction.first) {
        case brain::LossFunction::MSE:          return brain::tools::mse(output, expected);
        case brain::LossFunction::CrossEntropy: return brain::tools::cross_entropy(output, expected);
        default: return {};
    }
}

template<>
_hot brain::FType brain::NeuralNetwork::loss<true>(const brain::Values& outputs, const brain::Values& expected, const brain::FType coef) const
{
    std::size_t size = outputs.size();
    brain::FType total = 0.0;

    // check inputs
    if (outputs.size() != expected.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "both output and expected values must be of the same size");
    }
    if (size == 0) _unlikely return {};

    // compute mean
    if (this->_lossFunction.second) {
        for (std::size_t i = 0; i < size; ++i) total += this->_lossFunction.second(outputs[i], expected[i]) * coef;
    } else {
        switch (this->_lossFunction.first) {
            case brain::LossFunction::MSE:
                for (std::size_t i = 0; i < size; ++i) total += brain::tools::mse(outputs[i], expected[i]) * coef;
                break;
            case brain::LossFunction::CrossEntropy:
                for (std::size_t i = 0; i < size; ++i) total += brain::tools::cross_entropy(outputs[i], expected[i]) * coef;
                break;
            default: return {};
        }
    }
    return total / static_cast<brain::FType>(size);
}

template<>
_hot brain::Values brain::NeuralNetwork::loss<false>(const brain::Values& outputs, const brain::Values& expected, const brain::FType coef) const
{
    std::size_t size = outputs.size();
    brain::Values values(size);

    // check inputs
    if (outputs.size() != expected.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "both output and expected values must be of the same size");
    }

    // compute each loss
    if (this->_lossFunction.second) {
        for (std::size_t i = 0; i < size; ++i) values[i] = this->_lossFunction.second(outputs[i], expected[i]);
    } else {
        switch (this->_lossFunction.first) {
            case brain::LossFunction::MSE:
                for (std::size_t i = 0; i < size; ++i) values[i] = brain::tools::mse(outputs[i], expected[i]);
                break;
            case brain::LossFunction::CrossEntropy:
                for (std::size_t i = 0; i < size; ++i) values[i] = brain::tools::cross_entropy(outputs[i], expected[i]);
                break;
            default: break;
        }
    }

    // apply the coef
    if (coef != 1.0) for (brain::FType& value: values) value *= coef;
    return values;
}

_hot std::pair<brain::Values, brain::FType> brain::NeuralNetwork::loss(const brain::Values& outputs, const brain::Values& expected, const brain::FType coef) const
{
    std::size_t size = outputs.size();
    brain::Values values(size);
    brain::FType loss, total = 0.0;

    // check inputs
    if (outputs.size() != expected.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "both output and expected values must be of the same size");
    }
    if (size == 0) _unlikely return {};

    // one pass over the values: each call of the generic lambda is its own instantiation (fn inlined), same code as a hand written loop
    auto compute = [&](const auto& fn) {
        for (std::size_t i = 0; i < size; ++i) {
            loss = fn(outputs[i], expected[i]) * coef;
            values[i] = loss;
            total += loss;
        }
    };

    // compute each loss and the mean
    if (this->_lossFunction.second) {
        compute(this->_lossFunction.second);
    } else {
        switch (this->_lossFunction.first) {
            case brain::LossFunction::MSE:          compute([](const brain::FType o, const brain::FType e) {return brain::tools::mse(o, e);}); break;
            case brain::LossFunction::CrossEntropy: compute([](const brain::FType o, const brain::FType e) {return brain::tools::cross_entropy(o, e);}); break;
            default: return {std::move(values), {}};
        }
    }
    return {std::move(values), total / static_cast<brain::FType>(size)};
}

_hot brain::Values brain::NeuralNetwork::exec(const brain::Values& inputs) const
{
    this->inputs(inputs);
    return this->apply();
}

// Warn (and continue) when the extension of the file is not a known one
_cold static void checkExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char c) {return static_cast<char>(std::tolower(c));});
    if (extension == BRAIN_EXTENSION_BRAIN || extension == BRAIN_EXTENSION_LAYER || extension == BRAIN_EXTENSION_NETWORK) return;
    std::cerr << utils::exception::WarningException(utils::exception::ExternalCode::InvalidNetworkFile, "unknown extension '" + path.extension().string() + "' for " + path.string() + " (known: " BRAIN_EXTENSION_BRAIN ", " BRAIN_EXTENSION_LAYER ", " BRAIN_EXTENSION_NETWORK ")").formated() << std::endl;
}

_cold void brain::NeuralNetwork::load(const std::filesystem::path& path)
{
    std::unique_ptr<brain::INetwork> network;
    std::ifstream file(path, std::ios::binary);
    brain::Bytes data;

    // read the file
    checkExtension(path);
    if (!file) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "can't open " + path.string());
    }
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (data.size() < 2 || data[0] != static_cast<std::uint8_t>(BRAIN_MAGIC)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "bad magic bytes in " + path.string());
    }

    // the second magic byte gives the kind of network (the network checks both again)
    switch (data[1]) {
        case BRAIN_MAGIC_BRAIN: network = std::make_unique<brain::BrainNetwork>(0); break;
        case BRAIN_MAGIC_LAYER: network = std::make_unique<brain::LayerNetwork>(0, 0); break;
        default: throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "unknown network kind in " + path.string());
    }

    // the actual network is replaced only when the data are valid
    network->deserialize(data);
    this->clear();
    this->_network = std::move(network);
}

_cold void brain::NeuralNetwork::save(const std::filesystem::path& path) const
{
    // check inputs
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    checkExtension(path);

    // write the file
    const brain::Bytes data = this->_network->serialize();
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "can't open " + path.string());
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!file) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "can't write " + path.string());
    }
}

_cold void brain::NeuralNetwork::clear(void)
{
    // network
    this->_network.reset();

    // training
    this->_learningRateDecayed = this->_learningRate;
    this->_steps = 0;
}
