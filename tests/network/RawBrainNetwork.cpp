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
##  @file RawBrainNetwork.cpp

File Description:
##  Unit tests of the RawBrainNetwork
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

// A valid file of a raw network of 3 neurons with 2 link slots each
static brain::Bytes rawData(void)
{
    brain::Bytes data = brain::RawBrainNetwork(1).serialize();

    data.resize(HEADER);
    brain::tools::put<std::uint64_t>(data, 3);
    brain::tools::put<std::uint64_t>(data, 2);
    brain::tools::put<brain::FType>(data, 2.0f);
    brain::tools::put<brain::FType>(data, 3.0f);
    brain::tools::put<brain::FType>(data, 0.5f);
    brain::tools::put<std::uint8_t>(data, BRAIN_ACTIVATION_NONE);
    brain::tools::put<std::uint64_t>(data, 3);
    for (std::uint64_t i = 0; i < 3; ++i) {
        brain::tools::put<brain::FType>(data, static_cast<brain::FType>(i));
        brain::tools::put<brain::FType>(data, 0.5f * static_cast<brain::FType>(i));
        brain::tools::put<brain::FType>(data, 0.1f * static_cast<brain::FType>(i));
    }
    brain::tools::put<std::uint64_t>(data, 6);
    for (std::uint64_t i = 0; i < 6; ++i) {
        brain::tools::put<std::uint64_t>(data, (i + 1) % 3);
        brain::tools::put<brain::FType>(data, 0.25f * static_cast<brain::FType>(i));
        brain::tools::put<bool>(data, i % 2 == 0);
    }
    return data;
}

/* ------------------------------- RawBrainNetwork ------------------------------- */
TEST(RawBrainNetwork, Construct) {
    EXPECT_NO_THROW(brain::RawBrainNetwork network(10););
    EXPECT_NO_THROW(brain::RawBrainNetwork network(10, {2.0, 3.0}););
    EXPECT_NO_THROW(brain::RawBrainNetwork network(10, {2.0, 3.0}, utils::system::MemoryMode::Ram););
    EXPECT_THROW(brain::RawBrainNetwork network(0), utils::exception::IException);
    EXPECT_THROW(brain::RawBrainNetwork network(10, {0.0, 1.0}), utils::exception::IException);
    EXPECT_THROW(brain::RawBrainNetwork network(10, {1.0, -1.0}), utils::exception::IException);
}

TEST(RawBrainNetwork, IsABrainNetworkThatTrains) {
    std::unique_ptr<brain::BrainNetwork> network = std::make_unique<brain::RawBrainNetwork>(10);

    EXPECT_EQ(network->neurons(), (std::pair<brain::UIType, brain::UIType>{0, 10}));
    EXPECT_NO_THROW(network->inputs({0.0, 1.0}));
    EXPECT_NO_THROW((void)network->apply());
    EXPECT_NO_THROW(network->backpropagation(0.5));
    EXPECT_NO_THROW(network->weightInit());
}

TEST(RawBrainNetwork, LinksCapIsThePercentageOfTheNeurons) {
    EXPECT_EQ(brain::RawBrainNetwork(100).links(), 10u); // 10% of 100, the default hook gives more
    EXPECT_EQ(brain::RawBrainNetwork(2).links(), 1u); // at least one link
    EXPECT_EQ(brain::RawBrainNetwork(1).links(), 0u); // nobody to link to
}

TEST(RawBrainNetwork, LinksCapSetters) {
    brain::RawBrainNetwork network(100);

    network.linksCap(0.5f);
    EXPECT_EQ(network.links(), 50u);
    network.linksCap(1.0f);
    EXPECT_EQ(network.links(), 99u); // never more than the other neurons
    network.linksCap([](const brain::UIType n) {return n / 50;}); // the hook is the smallest here
    EXPECT_EQ(network.links(), 2u);
    network.resetLinksCap();
    EXPECT_EQ(network.links(), 10u);
    EXPECT_THROW(network.linksCap(0.0f), utils::exception::IException);
    EXPECT_THROW(network.linksCap(1.5f), utils::exception::IException);
    EXPECT_THROW(network.linksCap(std::nanf("")), utils::exception::IException);
    EXPECT_THROW(network.linksCap(brain::LinksCapFn{}), utils::exception::IException);
}

TEST(RawBrainNetwork, DefaultHookStaysFiniteOnABigNetwork) {
    // log1.1(x^(2log2(x))) written without the pow: finite even where the pow overflows a double
    EXPECT_TRUE(std::isfinite(BRAIN_LINKS_CAP_COMPUTING(1e9)));
    EXPECT_GT(BRAIN_LINKS_CAP_COMPUTING(100), 100.0); // over the neurons on a small network: the percentage is the limit
    EXPECT_LT(BRAIN_LINKS_CAP_COMPUTING(1e6), 0.1 * 1e6); // under 10% on a big one: the hook is the limit
}

TEST(RawBrainNetwork, SerializeStartsWithTheMagicBytes) {
    const brain::Bytes data = brain::RawBrainNetwork(10).serialize();

    ASSERT_GE(data.size(), 3u);
    EXPECT_EQ(data[0], BRAIN_MAGIC);
    EXPECT_EQ(data[1], BRAIN_MAGIC_RAW);
}

TEST(RawBrainNetwork, SerializeRoundTrip) {
    const brain::Bytes data = rawData();
    brain::RawBrainNetwork network(1);

    ASSERT_NO_THROW(network.deserialize(data));
    EXPECT_EQ(network.neurons().second, 3u);
    EXPECT_EQ(network.links(), 2u);
    EXPECT_EQ(network.serialize(), data);
}

TEST(RawBrainNetwork, DeserializeRefusesAnInvalidFile) {
    brain::RawBrainNetwork network(1);
    const brain::Bytes data = rawData();
    brain::Bytes bad = data;

    bad[1] = BRAIN_MAGIC_BRAIN; // the reduced file: other data
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    bad = data; bad.push_back(0x00);
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
    bad = data; bad.resize(bad.size() - 2);
    EXPECT_THROW(network.deserialize(bad), utils::exception::IException);
    EXPECT_EQ(network.neurons().second, 1u); // nothing edited by a refused file
}

TEST(RawBrainNetwork, ReduceGivesAnExecutionOnlyNetwork) {
    brain::RawBrainNetwork raw(1);
    std::unique_ptr<brain::BrainNetwork> reduced;

    ASSERT_NO_THROW(raw.deserialize(rawData()));
    reduced = raw.reduce();
    ASSERT_NE(reduced, nullptr);
    EXPECT_EQ(reduced->neurons(), raw.neurons());
    EXPECT_EQ(reduced->links(), raw.links());
    EXPECT_EQ(reduced->serialize()[1], BRAIN_MAGIC_BRAIN);
    EXPECT_THROW(reduced->backpropagation(0.5), utils::exception::IException);
}

TEST(RawBrainNetwork, ReducedFileIsSmallerThanTheRawOne) {
    brain::RawBrainNetwork raw(1);

    ASSERT_NO_THROW(raw.deserialize(rawData()));
    EXPECT_LT(raw.reduce()->serialize().size(), raw.serialize().size());
}
