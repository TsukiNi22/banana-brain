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
#include "brain/NetworkDefine.hpp"
#include "brain/tools/random.hpp"
#include "brain/tools/loss.hpp"
#include <filesystem>
#include <algorithm>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <numeric>
#include <fstream>
#include <utility>
#include <random>
#include <chrono>
#include <cctype>
#include <string>
#include <memory>
#include <cmath>

_hot void brain::NeuralNetwork::decay_(void)
{
    if (this->_decay == 0) return;
    if (++this->_steps % this->_decay == 0) this->_learningRateDecayed *= this->_decayCoef.second;
}

_hot void brain::NeuralNetwork::checkDataset_(const brain::Dataset& inputs, const brain::Dataset& expected_outputs) const
{
    std::size_t size = inputs.size();

    // check the datasets
    if (inputs.size() != expected_outputs.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "both inputs and expected dataset must be of the same size");
    } else if (size == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the datasets can't be empty");
    }

    // check each sample (the callers already checked the network)
    const std::pair<brain::UIType, brain::UIType>& sizes = this->_network->size();
    for (std::size_t i = 0; i < size; ++i) {
        if (sizes.first != 0 && inputs[i].size() != sizes.first) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "given " + std::to_string(inputs[i].size()) + ", expected " + std::to_string(sizes.first));
        } else if (sizes.second != 0 && expected_outputs[i].size() != sizes.second) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "given " + std::to_string(expected_outputs[i].size()) + ", expected " + std::to_string(sizes.second));
        }
    }
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Layered>(const brain::Layers& layers)
{
    this->requireUnfrozen();
    std::unique_ptr<brain::INetwork> network = std::make_unique<brain::LayerNetwork>(layers); // the network checks its parameters, the actual one is kept on an error
    this->clear();
    this->_network = std::move(network);
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Layered>(const brain::UIType layer, const brain::UIType neuron_per_layer)
{
    this->init<brain::NetworkType::Layered>(brain::Layers(layer, neuron_per_layer));
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Brain>(const brain::UIType neurons, const std::pair<brain::FType, brain::FType>& size)
{
    this->requireUnfrozen();
    std::unique_ptr<brain::INetwork> network = std::make_unique<brain::RawBrainNetwork>(neurons, size); // the network checks its parameters, the actual one is kept on an error
    this->clear();
    this->_network = std::move(network);
}

_cold void brain::NeuralNetwork::internalLossFunction(const brain::LossFunction fn)
{
    this->requireUnfrozen();
    this->_lossFunction = {fn, nullptr};
}

_cold void brain::NeuralNetwork::size(const brain::UIType input, const brain::UIType output)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->size(input, output);
}

_cold void brain::NeuralNetwork::input(const brain::UIType input)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->input(input);
}

_cold void brain::NeuralNetwork::output(const brain::UIType output)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->output(output);
}

_cold brain::UIType brain::NeuralNetwork::input(void) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->input();
}

_cold brain::UIType brain::NeuralNetwork::output(void) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->output();
}

_cold const std::pair<brain::UIType, brain::UIType>& brain::NeuralNetwork::size(void) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->size();
}

_cold void brain::NeuralNetwork::rand(const brain::FType min, const brain::FType max)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->rand(min, max);
}

_cold void brain::NeuralNetwork::rand(const brain::RandFn& fn)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->rand(fn);
}

_cold void brain::NeuralNetwork::activationFunction(const brain::ActivationFn& fn)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->activationFunction(fn);
}

_cold void brain::NeuralNetwork::internalActivationFunction(const brain::ActivationFunction fn)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->internalActivationFunction(fn);
}

_cold void brain::NeuralNetwork::scoreFunction(const brain::ScoreFn& fn)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->scoreFunction(fn);
}

_cold void brain::NeuralNetwork::internalScoreFunction(const brain::ScoreFunction fn)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->internalScoreFunction(fn);
}

_cold void brain::NeuralNetwork::scoreRate(const brain::FType rate)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->scoreRate(rate);
}

_cold const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& brain::NeuralNetwork::rand(void) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->rand();
}

_cold const std::pair<brain::ActivationFunction, brain::ActivationFn>& brain::NeuralNetwork::activationFunction(void) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->activationFunction();
}

_cold void brain::NeuralNetwork::learningRate(const brain::FType learning_rate, const brain::UIType learning_rate_decay, const std::pair<bool, brain::FType>& coef)
{
    // check inputs
    this->requireUnfrozen();
    if (!std::isfinite(learning_rate) || learning_rate <= 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidLearningRate, "the learning rate should be a finite number above 0");
    } else if (!std::isfinite(coef.second) || coef.second <= 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidLearningRate, "the decay coef should be a finite number above 0");
    }

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
    this->requireUnfrozen();
    this->_learningRateDecayed = this->_learningRate;
    this->_steps = 0;
}

// The signed error of each output times the learning rate: (output - expected) to descend for a layered network, (expected - output) as a score for a brain one (mean: one coefficient, their mean)
template<bool mean>
_hot void brain::NeuralNetwork::learn_(const brain::Values& outputs, const brain::Values& expected)
{
    brain::Values coefs = this->_network->scores(outputs, expected); // the score function of the network and its score rate
    brain::FType total = 0.0;

    for (brain::FType& coef: coefs) {
        coef *= this->_learningRateDecayed;
        total += coef;
    }
    if constexpr (mean) {
        this->backpropagation(outputs.empty() ? 0.0 : total / static_cast<brain::FType>(outputs.size()));
    } else {
        this->backpropagation(coefs);
    }
    this->decay_();
}

_hot brain::TrainResult brain::NeuralNetwork::train(const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::TrainOptions& options)
{
    const std::size_t size = inputs.size();
    const bool validation = !options.validationIn.empty();
    brain::TrainResult result;
    brain::TrainState state;

    // check inputs
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    } else if (!this->_lossFunction.second && this->_lossFunction.first == brain::LossFunction::None) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "no loss function set (lossFunction / internalLossFunction)");
    }
    this->checkDataset_(inputs, expected_outputs);
    if (validation) this->checkDataset_(options.validationIn, options.validationOut);
    else if (!options.validationOut.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "validation outputs without validation inputs");
    }
    if (options.stop == brain::StopCondition::TargetLoss && !(options.targetLoss >= 0)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the target loss should be above or equal to 0");
    } else if (options.stop == brain::StopCondition::Patience && (!validation || options.patience == 0)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the patience rule needs a validation set and a patience above 0");
    } else if (options.epochs == 0 && !options.onEpoch && options.stop == brain::StopCondition::Epochs) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "no epoch limit and no stop rule: it would never stop");
    }

    // setup: the mini-batch mode of the network for the training (put back at the end), the order of the samples
    const brain::UIType batchSize = options.batchSize == 0 ? size : std::min<brain::UIType>(options.batchSize, size);
    const bool grouped = batchSize != 1;
    const bool before = this->_network->batch();
    std::vector<std::size_t> order(size);
    std::mt19937_64 engine(options.seed != 0 ? static_cast<std::uint64_t>(options.seed) : static_cast<std::uint64_t>(brain::tools::rand_unsigned_integer()));
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    std::iota(order.begin(), order.end(), 0);
    this->_network->batch(grouped);
    try {
        for (brain::UIType epoch = 0; options.epochs == 0 || epoch < options.epochs; ++epoch) {
            brain::FType total = 0.0;
            brain::UIType inBatch = 0;
            bool go = true;

            // one pass over the samples: the weights change after each one, or after each group of batchSize (their mean gradient)
            if (options.shuffle) std::shuffle(order.begin(), order.end(), engine);
            for (const std::size_t i: order) {
                this->inputs(inputs[i]);
                const brain::Values outputs = this->apply();

                total += this->loss(outputs, expected_outputs[i]).second;
                if (options.mean) this->learn_<true>(outputs, expected_outputs[i]);
                else this->learn_<false>(outputs, expected_outputs[i]);
                if (!grouped) {++state.updates; continue;}
                if (++inBatch == batchSize) {
                    this->_network->applyBatch();
                    ++state.updates;
                    inBatch = 0;
                }
            }
            if (inBatch > 0) {this->_network->applyBatch(); ++state.updates;} // the last group, smaller

            // the state after the epoch: the loss, the validation (nothing updated by it)
            state.epoch = epoch + 1;
            state.loss = total / static_cast<brain::FType>(size);
            if (validation) {
                brain::FType validationLoss = 0.0;

                for (std::size_t i = 0; i < options.validationIn.size(); ++i)
                    validationLoss += this->loss(this->_network->apply(options.validationIn[i]), options.validationOut[i]).second;
                state.validationLoss = validationLoss / static_cast<brain::FType>(options.validationIn.size());
                if (epoch == 0 || state.validationLoss < state.bestValidation) {
                    state.bestValidation = state.validationLoss;
                    state.sinceBest = 0;
                } else {
                    ++state.sinceBest;
                }
            }
            state.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            result = {state.epoch, state.updates, state.loss, brain::StopReason::Epochs};

            // the stop rule: the one of the user when it is given, else the built-in one
            if (options.onEpoch) {
                go = options.onEpoch(options, state);
                if (!go) result.reason = brain::StopReason::Callback;
            } else if (options.stop == brain::StopCondition::TargetLoss && state.loss <= options.targetLoss) {
                go = false;
                result.reason = brain::StopReason::TargetLoss;
            } else if (options.stop == brain::StopCondition::Patience && state.sinceBest >= options.patience) {
                go = false;
                result.reason = brain::StopReason::Patience;
            }
            if (!go) break;
        }
    } catch (...) {
        this->_network->batch(before);
        throw;
    }
    this->_network->batch(before);
    return result;
}

_hot brain::FType brain::NeuralNetwork::loss(const brain::FType output, const brain::FType expected) const
{
    if (this->_lossFunction.second) return this->_lossFunction.second(output, expected);
    switch (this->_lossFunction.first) {
        case brain::LossFunction::MSE:          return brain::tools::mse(output, expected);
        case brain::LossFunction::CrossEntropy: return brain::tools::cross_entropy(output, expected);
        default: throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "no loss function set (lossFunction / internalLossFunction)");
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
            default: throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "no loss function set (lossFunction / internalLossFunction)");
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
            default: throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "no loss function set (lossFunction / internalLossFunction)");
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
    auto compute = [&](const auto& fn) { // xstyle: ignore CPP-AUTO (generic lambda: one instantiation per loss function, inlined)
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
            default: throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "no loss function set (lossFunction / internalLossFunction)");
        }
    }
    return {std::move(values), total / static_cast<brain::FType>(size)};
}

_hot void brain::NeuralNetwork::backpropagation(const brain::FType coef)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    } else if (!std::isfinite(coef)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficient should be a finite number");
    }
    this->_network->backpropagation(coef);
}

_hot void brain::NeuralNetwork::backpropagation(const brain::Values& coefs)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->backpropagation(coefs);
}

_cold void brain::NeuralNetwork::weightInit(const brain::FType fallback)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->weightInit(fallback);
}

_cold void brain::NeuralNetwork::biasInit(const brain::FType fallback)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->biasInit(fallback);
}

_hot void brain::NeuralNetwork::inputs(const brain::Values& inputs)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    this->_network->inputs(inputs);
}

_hot brain::Values brain::NeuralNetwork::apply(void)
{
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->apply();
}

_cold std::pair<brain::UIType, brain::UIType> brain::NeuralNetwork::neurons(void) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->neurons();
}

_cold brain::INetwork& brain::NeuralNetwork::get(void)
{
    // a reference that can change the network: refused when frozen, like every other change
    this->requireUnfrozen();
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return *this->_network;
}

_cold const brain::INetwork& brain::NeuralNetwork::get(void) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return *this->_network;
}

_hot brain::Values brain::NeuralNetwork::exec(const brain::Values& inputs) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->apply(inputs);
}

_hot brain::Dataset brain::NeuralNetwork::execAll(const brain::Dataset& inputs) const
{
    if (!this->_network) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::NoNetwork);
    }
    return this->_network->applyAll(inputs);
}

// Warn (and continue) when the extension of the file is not a known one
_cold static void checkExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char c) {return static_cast<char>(std::tolower(c));});
    if (extension == BRAIN_EXTENSION_BRAIN || extension == BRAIN_EXTENSION_RAW || extension == BRAIN_EXTENSION_LAYER || extension == BRAIN_EXTENSION_NETWORK) return;
    std::cerr << utils::exception::WarningException(utils::exception::ExternalCode::InvalidNetworkFile, "unknown extension '" + path.extension().string() + "' for " + path.string() + " (known: " BRAIN_EXTENSION_BRAIN ", " BRAIN_EXTENSION_RAW ", " BRAIN_EXTENSION_LAYER ", " BRAIN_EXTENSION_NETWORK ")").formated() << std::endl;
}

_cold void brain::NeuralNetwork::load(const std::filesystem::path& path)
{
    std::unique_ptr<brain::INetwork> network;
    std::ifstream file(path, std::ios::binary);
    brain::Bytes data;

    // read the file
    this->requireUnfrozen();
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
        case BRAIN_MAGIC_BRAIN: network = std::make_unique<brain::BrainNetwork>(1); break; // placeholder, deserialize() rebuilds it
        case BRAIN_MAGIC_RAW:   network = std::make_unique<brain::RawBrainNetwork>(1); break;
        case BRAIN_MAGIC_LAYER: network = std::make_unique<brain::LayerNetwork>(1, 1); break;
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
    this->requireUnfrozen();

    // network
    this->_network.reset();

    // training
    this->_learningRateDecayed = this->_learningRate;
    this->_steps = 0;
}
