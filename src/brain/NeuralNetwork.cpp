/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 10/11/2026 by @author Tsukini

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
_cold void brain::NeuralNetwork::init<brain::NetworkType::Layered>(const brain::Layers& layers)
{
    /* Nothing */
}

template<>
_cold void brain::NeuralNetwork::init<brain::NetworkType::Brain>(const brain::UIType neurons, const std::pair<brain::FType, brain::FType>& size)
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

_cold void brain::NeuralNetwork::rand(const brain::RandFn& fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::activationFunction(const brain::ActivationFn& fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::internalActivationFunction(const brain::ActivationFunction fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::lossFunction(const brain::LossFn& fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::internalLossFunction(const brain::LossFunction fn)
{
    /* Nothing */
}

_cold void brain::NeuralNetwork::learningRate(const brain::FType learning_rate)
{
    /* Nothing */
}

_cold std::pair<brain::UIType, brain::UIType> brain::NeuralNetwork::neurons(void) const
{
    /* Nothing */
    return {};
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
