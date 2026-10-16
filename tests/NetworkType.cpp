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
##  @file NetworkType.cpp

File Description:
##  Unit tests of the value types, the enums & the neuron structs
\**************************************************************/

#include "brain/NetworkDefine.hpp"
#include "brain/NetworkType.hpp"
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>
#include <new>

/* ------------------------------- Types ------------------------------- */
// The value types keep the constraints of their static_assert
static_assert(std::is_signed_v<brain::IType> && std::is_integral_v<brain::IType>);
static_assert(std::is_unsigned_v<brain::UIType> && std::is_integral_v<brain::UIType>);
static_assert(std::is_floating_point_v<brain::FType>);

// The data aliases are built on FType
static_assert(std::is_same_v<brain::Values::value_type, brain::FType>);
static_assert(std::is_same_v<brain::Dataset::value_type, brain::Values>);

/* ------------------------------- Enums ------------------------------- */
TEST(NetworkDefine, NoneValues) {
    EXPECT_NE(brain::ActivationFunction::None, brain::ActivationFunction::Sigmoid);
    EXPECT_NE(brain::LossFunction::None, brain::LossFunction::MSE);
    EXPECT_NE(brain::NetworkType::Layered, brain::NetworkType::Brain);
    EXPECT_NE(brain::NetworkType::Brain, brain::NetworkType::Banana);
}

/* ------------------------------- Neurons ------------------------------- */
TEST(LayerNeuron, DefaultState) {
    brain::LayerNeuron neuron;
    EXPECT_TRUE(neuron.incoming.empty());
    EXPECT_EQ(neuron.bias, 0.0);
}

TEST(LayerNeuron, CacheLineAligned) {
    EXPECT_EQ(alignof(brain::LayerNeuron), std::hardware_destructive_interference_size);
    EXPECT_EQ(sizeof(brain::LayerNeuron) % std::hardware_destructive_interference_size, 0u);
}

TEST(BrainNeuron, DefaultState) {
    brain::BrainNeuron neuron;
    EXPECT_TRUE(neuron.incoming.empty());
    EXPECT_TRUE(neuron.ordered.empty());
    EXPECT_EQ(neuron.id, 0u);
    EXPECT_EQ(neuron.bias, 0.0);
}

TEST(BrainNeuron, EqualityOnIdsOnly) {
    brain::BrainNeuron a;
    brain::BrainNeuron b;
    a.id = 1; a.pos = {0.0, 0.0};
    b.id = 1; b.pos = {1.0, 1.0}; b.bias = 0.5;
    EXPECT_TRUE(a == b); // same id, different position / bias
    b.id = 2;
    EXPECT_FALSE(a == b);
}

// The squared distance is exposed by distance2() and by operator| (same result, an FType)
static_assert(std::is_same_v<decltype(std::declval<const brain::BrainNeuron&>() | std::declval<const brain::BrainNeuron&>()), brain::FType>);
static_assert(std::is_same_v<decltype(std::declval<const brain::BrainNeuron&>().distance2(std::declval<const brain::BrainNeuron&>())), brain::FType>);

TEST(BrainNeuron, SquaredDistanceOperator) {
    brain::BrainNeuron a;
    brain::BrainNeuron b;
    a.pos = {0.0, 0.0};
    b.pos = {3.0, 4.0};
    EXPECT_EQ(a | b, a.distance2(b)); // the operator forwards to distance2 (squared: 25 once implemented, never 5)
}

TEST(BrainNeuron, CacheLineAligned) {
    EXPECT_EQ(alignof(brain::BrainNeuron), std::hardware_destructive_interference_size);
    EXPECT_EQ(sizeof(brain::BrainNeuron) % std::hardware_destructive_interference_size, 0u);
}
