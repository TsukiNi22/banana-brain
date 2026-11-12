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
    EXPECT_TRUE(neuron.pos.empty()); // 0D until the network places it (one value per axis)
    EXPECT_EQ(link.id, 0u);
    EXPECT_EQ(link.weight, 0.0);
    EXPECT_FALSE(link.inward);
}
