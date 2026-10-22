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
}

/* ------------------------------- Neurons ------------------------------- */
TEST(LayerNeuron, DefaultState) {
    brain::LayerNeuron neuron;
    EXPECT_EQ(neuron.incoming, 0u);
    EXPECT_EQ(neuron.count, 0u);
    EXPECT_EQ(neuron.bias, 0.0);
}

TEST(BrainNeuron, ReducedNeuronIsOnlyTheBias) {
    brain::BrainNeuron neuron;
    brain::BrainLink link;

    EXPECT_EQ(neuron.bias, 0.0);
    EXPECT_EQ(sizeof(brain::BrainNeuron), sizeof(brain::FType)); // nothing else, no padding: the smallest neuron
    EXPECT_EQ(link.id, 0u);
    EXPECT_EQ(link.weight, 0.0); // a free slot is all at 0
}

TEST(RawBrainNeuron, DefaultState) {
    brain::RawBrainNeuron neuron;
    brain::RawBrainLink link;

    EXPECT_EQ(neuron.bias, 0.0);
    EXPECT_EQ(link.id, 0u);
    EXPECT_EQ(link.weight, 0.0);
    EXPECT_FALSE(link.inward);
}

// The squared distance is exposed by distance2() and by operator| (same result, an FType)
static_assert(std::is_same_v<decltype(std::declval<const brain::RawBrainNeuron&>() | std::declval<const brain::RawBrainNeuron&>()), brain::FType>);
static_assert(std::is_same_v<decltype(std::declval<const brain::RawBrainNeuron&>().distance2(std::declval<const brain::RawBrainNeuron&>())), brain::FType>);

TEST(RawBrainNeuron, SquaredDistanceOperator) {
    brain::RawBrainNeuron a;
    brain::RawBrainNeuron b;
    a.pos = {0.0, 0.0};
    b.pos = {3.0, 4.0};
    EXPECT_EQ(a | b, a.distance2(b));
    EXPECT_FLOAT_EQ(a | b, 25.0); // squared: 25, never 5
}

// The neurons and the links are stored in a utils::system::HybridMemory: bytes copied to the gpu
static_assert(std::is_trivially_copyable_v<brain::LayerNeuron>);
static_assert(std::is_trivially_copyable_v<brain::BrainNeuron>);
static_assert(std::is_trivially_copyable_v<brain::BrainLink>);
static_assert(std::is_trivially_copyable_v<brain::RawBrainNeuron>);
static_assert(std::is_trivially_copyable_v<brain::RawBrainLink>);
