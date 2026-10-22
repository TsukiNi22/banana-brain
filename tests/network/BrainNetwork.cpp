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
##  @file BrainNetwork.cpp

File Description:
##  Unit tests of the BrainNetwork
\**************************************************************/

#define _Exception
#include "brain/network/Network.hpp"
#include "brain/tools/bytes.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <memory>

static constexpr brain::UIType HEADER = 28; // magic (2) + activation (1) + sizes (16) + random range (8) + random mode (1)

static bool throwsCode(const std::function<void()>& fn, const utils::exception::ExternalCode code)
{
    try {fn();} catch (const utils::exception::IException& e) {return e.getCode() == static_cast<utils::exception::InternalCode>(code);}
    return false;
}

// A valid file of a reduced network of 3 neurons with 2 link slots each
static brain::Bytes brainData(void)
{
    brain::Bytes data = brain::BrainNetwork(1).serialize();

    data.resize(HEADER);
    brain::tools::put<std::uint64_t>(data, 3);
    brain::tools::put<std::uint64_t>(data, 2);
    brain::tools::put<std::uint64_t>(data, 3);
    for (std::uint64_t i = 0; i < 3; ++i)
        brain::tools::put<brain::FType>(data, 0.5f * static_cast<brain::FType>(i));
    brain::tools::put<std::uint64_t>(data, 6);
    for (std::uint64_t i = 0; i < 6; ++i) {
        brain::tools::put<std::uint64_t>(data, i % 3);
        brain::tools::put<brain::FType>(data, 0.25f * static_cast<brain::FType>(i));
    }
    return data;
}

/* ------------------------------- BrainNetwork ------------------------------- */
TEST(BrainNetwork, Construct) {
    EXPECT_NO_THROW(brain::BrainNetwork network(10););
    EXPECT_NO_THROW(brain::BrainNetwork network(10, 3););
    EXPECT_NO_THROW(brain::BrainNetwork network(10, 3, utils::system::MemoryMode::Ram););
    EXPECT_THROW(brain::BrainNetwork network(0), utils::exception::IException);
}

TEST(BrainNetwork, NeuronsAndLinks) {
    brain::BrainNetwork network(10, 3);

    EXPECT_EQ(network.neurons(), (std::pair<brain::UIType, brain::UIType>{0, 10}));
    EXPECT_EQ(network.links(), 3u);
}

TEST(BrainNetwork, UsableAsINetwork) {
    std::unique_ptr<brain::INetwork> network = std::make_unique<brain::BrainNetwork>(10);
    ASSERT_NE(network, nullptr);
    EXPECT_NO_THROW(network->inputs({0.0, 1.0}));
    EXPECT_NO_THROW((void)network->apply(brain::Values{0.0, 1.0}));
}

TEST(BrainNetwork, TheTrainingIsRefused) {
    brain::BrainNetwork network(10);

    EXPECT_TRUE(throwsCode([&] {(void)network.apply();}, utils::exception::ExternalCode::ReadOnlyNetwork));
    EXPECT_TRUE(throwsCode([&] {network.backpropagation(0.5);}, utils::exception::ExternalCode::ReadOnlyNetwork));
    EXPECT_TRUE(throwsCode([&] {network.backpropagation(brain::Values{0.5});}, utils::exception::ExternalCode::ReadOnlyNetwork));
    EXPECT_TRUE(throwsCode([&] {network.weightInit();}, utils::exception::ExternalCode::ReadOnlyNetwork));
    EXPECT_TRUE(throwsCode([&] {network.biasInit();}, utils::exception::ExternalCode::ReadOnlyNetwork));
}

TEST(BrainNetwork, SerializeStartsWithTheMagicBytes) {
    const brain::Bytes data = brain::BrainNetwork(10, 3).serialize();

    ASSERT_GE(data.size(), 3u);
    EXPECT_EQ(data[0], BRAIN_MAGIC);
    EXPECT_EQ(data[1], BRAIN_MAGIC_BRAIN);
    EXPECT_EQ(data[2], BRAIN_ACTIVATION_NONE);
}

TEST(BrainNetwork, SerializeRoundTrip) {
    const brain::Bytes data = brainData();
    brain::BrainNetwork network(1);

    ASSERT_NO_THROW(network.deserialize(data));
    EXPECT_EQ(network.neurons().second, 3u);
    EXPECT_EQ(network.links(), 2u);
    EXPECT_EQ(network.serialize(), data);
}

TEST(BrainNetwork, DeserializeRefusesAnInvalidFile) {
    brain::BrainNetwork network(1);
    const brain::Bytes data = brainData();
    brain::Bytes bad = data;

    bad[1] = BRAIN_MAGIC_LAYER; // the file of an other kind of network
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    bad = data; bad.push_back(0x00);
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    bad = data; bad.resize(bad.size() - 2);
    EXPECT_THROW(network.deserialize(bad), utils::exception::IException);
    bad = data; // a link to a neuron that does not exist
    bad[bad.size() - 4 - 8] = 9;
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    EXPECT_EQ(network.neurons().second, 1u); // nothing edited by a refused file
}

TEST(BrainNetwork, InitAllocatesFreeNeuronsAndSlots) {
    const brain::Bytes data = brain::BrainNetwork(5, 3).serialize();
    brain::UIType offset = HEADER;

    EXPECT_EQ(brain::tools::get<std::uint64_t>(data, offset), 5u);
    EXPECT_EQ(brain::tools::get<std::uint64_t>(data, offset), 3u);
    EXPECT_EQ(brain::tools::get<std::uint64_t>(data, offset), 5u); // a neuron each
    for (brain::UIType i = 0; i < 5; ++i)
        EXPECT_EQ(brain::tools::get<brain::FType>(data, offset), 0.0f);
    EXPECT_EQ(brain::tools::get<std::uint64_t>(data, offset), 15u); // 3 slots of links for each neuron, all free
    for (brain::UIType i = 0; i < 15; ++i) {
        EXPECT_EQ(brain::tools::get<std::uint64_t>(data, offset), 0u);
        EXPECT_EQ(brain::tools::get<brain::FType>(data, offset), 0.0f);
    }
    EXPECT_EQ(offset, data.size());
}

TEST(BrainNetwork, TooManySlotsOverflowTheSize) {
    EXPECT_THROW(brain::BrainNetwork network(std::numeric_limits<brain::UIType>::max() / 2, 4), utils::exception::IException);
}
