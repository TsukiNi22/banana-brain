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
#include <utility>

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Layered>(const brain::UIType layer, const brain::UIType neuron_per_layer)
{
    /* Nothing */
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Brain>(const brain::UIType neurons, const std::pair<brain::FType, brain::FType>& size)
{
    /* Nothing */
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Banana>(void)
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

_cold void brain::NeuralNetwork::learningRate(const brain::FType learning_rate, const brain::UIType learning_rate_decay, const std::pair<bool, brain::FType>& coef)
{
    /* Nothing */
}

_cold std::pair<brain::UIType, brain::UIType> brain::NeuralNetwork::neurons(void) const
{
    /* Nothing */
    return {};
}

_hot void brain::NeuralNetwork::train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationFn& backpropagation)
{
    /* Nothing */
}

_hot void brain::NeuralNetwork::train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationsFn& backpropagation)
{
    /* Nothing */
}

_hot void brain::NeuralNetwork::train(const brain::FType loss, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationFn& backpropagation)
{
    /* Nothing */
}

_hot void brain::NeuralNetwork::train(const brain::FType loss, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationsFn& backpropagation)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::load(void)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::unload(void)
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

_hot brain::FType brain::NeuralNetwork::loss(const brain::FType output, const brain::FType expected) const
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

_cold void brain::NeuralNetwork::load(const std::filesystem::path& path)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::save(const std::filesystem::path& path) const
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::clear(void)
{
    /* Nothing */
}
