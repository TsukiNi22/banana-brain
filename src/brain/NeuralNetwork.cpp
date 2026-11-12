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
##  @file NeuralNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/Network.hpp"
#include "brain/NeuralNetwork.hpp"
#include <filesystem>
#include <optional>
#include <ostream>
#include <utility>
#include <memory>

_cold void brain::NeuralNetwork::check_(const brain::TrainOptions& options) const
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::decay_(const brain::TrainOptions& options)
{
    /* Nothing */
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Layered>(const brain::UIType layer, const brain::UIType neuron_per_layer, const brain::LayerOptions& options)
{
    /* Nothing */
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Layered>(const brain::Layers& layers, const brain::LayerOptions& options)
{
    /* Nothing */
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Brain>(const brain::UIType neurons, const brain::BrainOptions& options)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::weightInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::biasInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::size(const brain::UIType input, const brain::UIType output)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::input(const brain::UIType input)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::output(const brain::UIType output)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::rand(const brain::FType min, const brain::FType max)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::rand(const brain::Rand& rand)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::activation(const brain::Activation& fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::backend(std::shared_ptr<brain::IBackend> backend)
{
    /* Nothing */
}

_cold const std::pair<brain::UIType, brain::UIType>& brain::NeuralNetwork::size(void) const
{
    /* Nothing */
    return this->_network->size();
}

_cold brain::UIType brain::NeuralNetwork::input(void) const
{
    /* Nothing */
    return {};
}

_cold brain::UIType brain::NeuralNetwork::output(void) const
{
    /* Nothing */
    return {};
}

_cold const brain::Rand& brain::NeuralNetwork::rand(void) const
{
    /* Nothing */
    return this->_network->rand();
}

_cold const std::pair<brain::FType, brain::FType>& brain::NeuralNetwork::range(void) const
{
    /* Nothing */
    return this->_network->range();
}

_cold const brain::Activation& brain::NeuralNetwork::activation(void) const
{
    /* Nothing */
    return this->_network->activation();
}

_cold std::pair<brain::UIType, brain::UIType> brain::NeuralNetwork::neurons(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::NeuralNetwork::options(const brain::TrainOptions& options)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::loss(const brain::Loss& fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::score(const brain::Score& fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::learningRate(const brain::FType learning_rate)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::decay(const brain::UIType every, const std::pair<bool, brain::FType>& coef)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::stepLimit(const brain::FType limit)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::seed(const brain::IType seed)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::epochs(const brain::UIType epochs)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::batchSize(const brain::UIType size)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::shuffle(const bool shuffle)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::validation(const brain::FType share)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::validation(const brain::Dataset& inputs, const brain::Dataset& expected)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::patience(const brain::UIType epochs)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::targetLoss(const brain::FType loss)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::keepBest(const bool keep)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::autosave(const std::optional<std::filesystem::path>& path)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::hook(std::shared_ptr<brain::ITrainHook> hook)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::optimizer(std::unique_ptr<brain::IOptimizer> optimizer)
{
    /* Nothing */
}

_hot void brain::NeuralNetwork::train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs)
{
    /* Nothing */
}

_hot void brain::NeuralNetwork::inputs(const brain::Values& inputs)
{
    /* Nothing */
}

_hot brain::Values brain::NeuralNetwork::apply(void)
{
    /* Nothing */
    return {};
}

_hot void brain::NeuralNetwork::backpropagation(const brain::FType coef)
{
    /* Nothing */
}

_hot void brain::NeuralNetwork::backpropagation(const brain::Values& coefs)
{
    /* Nothing */
}

_hot brain::FType brain::NeuralNetwork::loss(const brain::Value output, const brain::Value expected) const
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::NeuralNetwork::loss(const brain::Values& outputs, const brain::Values& expected) const
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::NeuralNetwork::exec(const brain::Values& inputs) const
{
    /* Nothing */
    return {};
}

_hot brain::Dataset brain::NeuralNetwork::execAll(const brain::Dataset& inputs) const
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::NeuralNetwork::evaluate(const brain::Dataset& inputs, const brain::Dataset& expected) const
{
    /* Nothing */
    return {};
}

_cold brain::Metrics brain::NeuralNetwork::metrics(const brain::Dataset& inputs, const brain::Dataset& expected) const
{
    /* Nothing */
    return {};
}

_cold void brain::NeuralNetwork::load(const std::filesystem::path& path)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::save(const std::filesystem::path& path) const
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::loadState(const std::filesystem::path& path)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::saveState(const std::filesystem::path& path) const
{
    /* Nothing */
}

_cold std::unique_ptr<brain::NeuralNetwork> brain::NeuralNetwork::from(const std::filesystem::path& path)
{
    /* Nothing */
    return {};
}

_cold void brain::NeuralNetwork::clear(void)
{
    /* Nothing */
}

_cold std::unique_ptr<brain::NeuralNetwork> brain::NeuralNetwork::clone(void) const
{
    /* Nothing */
    return {};
}

_cold std::ostream& brain::operator<<(std::ostream& os, const brain::NeuralNetwork& nn)
{
    /* Nothing */
    return os;
}
