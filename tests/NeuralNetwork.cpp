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
##  Unit tests of the NeuralNetwork (configuration, training & usage)
\**************************************************************/

#include "brain/NeuralNetwork.hpp"
#include <gtest/gtest.h>
#include <utility>
#include <cstdint>

/* ------------------------------- NeuralNetwork ------------------------------- */
// The round / loss of train must be exactly a UIType / FType: any other type is refused at compile time
template<typename T, typename Fn = brain::BackpropagationFn>
constexpr bool trainAccepts = requires(brain::NeuralNetwork n, brain::Dataset d, Fn fn) {n.train(T{}, d, d, fn);};
static_assert(trainAccepts<brain::UIType>);
static_assert(trainAccepts<brain::FType>);
static_assert(trainAccepts<brain::UIType, brain::BackpropagationsFn>);
static_assert(trainAccepts<brain::FType, brain::BackpropagationsFn>);
static_assert(!trainAccepts<int>);
static_assert(!trainAccepts<unsigned int>);
static_assert(!trainAccepts<double>);
static_assert(!trainAccepts<std::int64_t>);

TEST(NeuralNetwork, DefaultState) {
    brain::NeuralNetwork network;
    EXPECT_FALSE(network.isFrozen());
    EXPECT_EQ(network.seed(), 0);
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{0, 0}));
    EXPECT_EQ(network.input(), 0u);
    EXPECT_EQ(network.output(), 0u);
    EXPECT_EQ(network.learningRate(), 1.0);
    EXPECT_EQ(network.activationFunction().first, brain::ActivationFunction::None);
    EXPECT_EQ(network.activationFunction().second, nullptr);
    EXPECT_EQ(network.lossFunction().first, brain::LossFunction::None);
    EXPECT_EQ(network.lossFunction().second, nullptr);
    EXPECT_EQ(network.rand().first, (std::pair<brain::FType, brain::FType>{-1.0, 1.0}));
    EXPECT_EQ(network.rand().second, nullptr);
}

TEST(NeuralNetwork, Setters) {
    brain::NeuralNetwork network;
    network.seed(42);
    network.size(3, 2);
    network.rand(0.0, 0.5);
    network.internalActivationFunction(brain::ActivationFunction::ReLU);
    network.internalLossFunction(brain::LossFunction::MSE);
    EXPECT_EQ(network.seed(), 42);
    EXPECT_EQ(network.input(), 3u);
    EXPECT_EQ(network.output(), 2u);
    EXPECT_EQ(network.rand().first, (std::pair<brain::FType, brain::FType>{0.0, 0.5}));
    EXPECT_EQ(network.activationFunction().first, brain::ActivationFunction::ReLU);
    EXPECT_EQ(network.lossFunction().first, brain::LossFunction::MSE);

    network.input(5);
    network.output(1);
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{5, 1}));
}

TEST(NeuralNetwork, ExternalFunctionsResetTheInternalOne) {
    brain::NeuralNetwork network;
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    network.activationFunction([](const brain::FType value) {return value;});
    EXPECT_EQ(network.activationFunction().first, brain::ActivationFunction::None);
    EXPECT_NE(network.activationFunction().second, nullptr);

    network.internalLossFunction(brain::LossFunction::CrossEntropy);
    network.lossFunction([](const brain::FType output, const brain::FType expected) {return output - expected;});
    EXPECT_EQ(network.lossFunction().first, brain::LossFunction::None);
    EXPECT_NE(network.lossFunction().second, nullptr);

    network.rand([](const brain::IType seed) {return static_cast<brain::FType>(seed);});
    EXPECT_NE(network.rand().second, nullptr);
}

TEST(NeuralNetwork, Freeze) {
    brain::NeuralNetwork network;
    network.freeze();
    EXPECT_TRUE(network.isFrozen());
}
