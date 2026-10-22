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
##  Unit tests of the LayerNetwork
\**************************************************************/

#define _Exception
#include "brain/network/Network.hpp"
#include "brain/tools/bytes.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <memory>

static constexpr brain::UIType HEADER = 28; // magic (2) + activation (1) + sizes (16) + random range (8) + random mode (1)

static bool throwsCode(const std::function<void()>& fn, const utils::exception::ExternalCode code)
{
    try {fn();} catch (const utils::exception::IException& e) {return e.getCode() == static_cast<utils::exception::InternalCode>(code);}
    return false;
}

// A valid file of a network of two layers of two neurons: 4 neurons and 6 weights stored
static brain::Bytes layerData(void)
{
    brain::Bytes data = brain::LayerNetwork(1, 1).serialize();

    data.resize(HEADER);
    brain::tools::put<std::uint64_t>(data, 2);
    brain::tools::put<std::uint64_t>(data, 2);
    brain::tools::put<std::uint64_t>(data, 2);
    brain::tools::put<std::uint64_t>(data, 4);
    for (std::uint64_t i = 0; i < 4; ++i) {
        brain::tools::put<std::uint64_t>(data, i * 2);
        brain::tools::put<std::uint64_t>(data, 2);
        brain::tools::put<brain::FType>(data, 0.5f * static_cast<brain::FType>(i));
    }
    brain::tools::put<std::uint64_t>(data, 6);
    for (std::uint64_t i = 0; i < 6; ++i)
        brain::tools::put<brain::FType>(data, 0.25f * static_cast<brain::FType>(i));
    return data;
}

/* ------------------------------- LayerNetwork ------------------------------- */
TEST(LayerNetwork, Construct) {
    EXPECT_NO_THROW(brain::LayerNetwork network(2, 3););
    EXPECT_NO_THROW(brain::LayerNetwork network(brain::Layers{4, 3, 1}););
    EXPECT_NO_THROW(brain::LayerNetwork network(2, 3, utils::system::MemoryMode::Ram););
}

TEST(LayerNetwork, ConstructRejectsEmptyLayers) {
    EXPECT_THROW(brain::LayerNetwork network(0, 3), utils::exception::IException);
    EXPECT_THROW(brain::LayerNetwork network(2, 0), utils::exception::IException);
    EXPECT_THROW(brain::LayerNetwork network(brain::Layers{}), utils::exception::IException);
    EXPECT_THROW(brain::LayerNetwork network(brain::Layers{3, 0, 1}), utils::exception::IException);
}

TEST(LayerNetwork, NeuronsAreTheLayersAndTheTotal) {
    EXPECT_EQ(brain::LayerNetwork(2, 3).neurons(), (std::pair<brain::UIType, brain::UIType>{2, 6}));
    EXPECT_EQ(brain::LayerNetwork(brain::Layers{4, 3, 1}).neurons(), (std::pair<brain::UIType, brain::UIType>{3, 8}));
    EXPECT_EQ(brain::LayerNetwork(brain::Layers{4, 3, 1}).layers(), (brain::Layers{4, 3, 1}));
}

TEST(LayerNetwork, UsableAsINetwork) {
    std::unique_ptr<brain::INetwork> network = std::make_unique<brain::LayerNetwork>(2, 3);
    ASSERT_NE(network, nullptr);
    EXPECT_NO_THROW(network->inputs({0.0, 1.0}));
}

TEST(LayerNetwork, ExecutionChecksItsInputs) {
    brain::LayerNetwork network(2, 3);

    EXPECT_TRUE(throwsCode([&] {(void)network.apply(brain::Values{});}, utils::exception::ExternalCode::InvalidInputSize));
    network.input(2);
    EXPECT_TRUE(throwsCode([&] {(void)network.apply(brain::Values{1.0, 2.0, 3.0});}, utils::exception::ExternalCode::InvalidInputSize));
    EXPECT_NO_THROW((void)network.apply(brain::Values{1.0, 2.0}));
    EXPECT_TRUE(throwsCode([&] {(void)network.apply();}, utils::exception::ExternalCode::InvalidInputSize)); // training: no inputs loaded
}

TEST(LayerNetwork, SettersAreChecked) {
    brain::LayerNetwork network(2, 3);

    EXPECT_THROW(network.rand(2.0, 1.0), utils::exception::IException);
    EXPECT_THROW(network.rand(brain::RandFn{}), utils::exception::IException);
    EXPECT_THROW(network.activationFunction(brain::ActivationFn{}), utils::exception::IException);
    EXPECT_THROW(network.weightInit(0.0), utils::exception::IException);
    EXPECT_THROW(network.biasInit(std::nanf("")), utils::exception::IException);
    EXPECT_THROW(network.backpropagation(std::nanf("")), utils::exception::IException);
    EXPECT_THROW(network.backpropagation(brain::Values{}), utils::exception::IException);
}

TEST(LayerNetwork, SerializeStartsWithTheMagicBytesAndTheActivation) {
    brain::LayerNetwork network(2, 3);
    brain::Bytes data = network.serialize();

    ASSERT_GE(data.size(), 3u);
    EXPECT_EQ(data[0], BRAIN_MAGIC);
    EXPECT_EQ(data[1], BRAIN_MAGIC_LAYER);
    EXPECT_EQ(data[2], BRAIN_ACTIVATION_NONE);
    network.internalActivationFunction(brain::ActivationFunction::ReLU);
    EXPECT_EQ(network.serialize()[2], static_cast<std::uint8_t>(brain::ActivationFunction::ReLU));
    network.activationFunction([](const brain::FType value) {return value;});
    EXPECT_EQ(network.serialize()[2], BRAIN_ACTIVATION_CUSTOM);
}

TEST(LayerNetwork, SerializeRoundTrip) {
    const brain::Bytes data = layerData();
    brain::LayerNetwork network(1, 1);

    ASSERT_NO_THROW(network.deserialize(data));
    EXPECT_EQ(network.layers(), (brain::Layers{2, 2}));
    EXPECT_EQ(network.serialize(), data);
}

TEST(LayerNetwork, SerializeKeepsTheSettings) {
    brain::LayerNetwork network(2, 3);
    brain::LayerNetwork other(1, 1);

    network.size(4, 2);
    network.rand(-0.5, 0.25);
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    ASSERT_NO_THROW(other.deserialize(network.serialize()));
    EXPECT_EQ(other.size(), (std::pair<brain::UIType, brain::UIType>{4, 2}));
    EXPECT_EQ(other.rand().first, (std::pair<brain::FType, brain::FType>{-0.5, 0.25}));
    EXPECT_EQ(other.activationFunction().first, brain::ActivationFunction::Sigmoid);
}

TEST(LayerNetwork, DeserializeRefusesAnInvalidFile) {
    brain::LayerNetwork network(1, 1);
    brain::Bytes data = layerData();
    brain::Bytes bad = data;

    bad[1] = BRAIN_MAGIC_BRAIN; // the file of an other kind of network
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    bad = data; bad[0] = 0x00;
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    EXPECT_TRUE(throwsCode([&] {network.deserialize({});}, utils::exception::ExternalCode::InvalidNetworkFile));
    bad = data; bad.push_back(0x00); // data after the weights
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    bad = data; bad.resize(bad.size() - 3); // truncated weights
    EXPECT_THROW(network.deserialize(bad), utils::exception::IException);
    bad = data; bad[2] = 0x7A; // unknown activation
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    EXPECT_EQ(network.layers(), (brain::Layers{1})); // nothing edited by a refused file
}

TEST(LayerNetwork, DeserializeRefusesAHugeCount) {
    brain::Bytes data = brain::LayerNetwork(1, 1).serialize();

    data.resize(HEADER);
    brain::tools::put<std::uint64_t>(data, ~std::uint64_t{0}); // billions of layers announced, nothing behind
    EXPECT_TRUE(throwsCode([&] {brain::LayerNetwork(1, 1).deserialize(data);}, utils::exception::ExternalCode::InvalidNetworkFile));
}
