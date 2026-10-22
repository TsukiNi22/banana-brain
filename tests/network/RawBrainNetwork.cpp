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
#include "brain/tools/random.hpp"
#include "brain/tools/bytes.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>

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

struct ParsedRaw {
    std::vector<brain::RawBrainNeuron> neurons;
    std::vector<brain::RawBrainLink> links;
    brain::UIType cap = 0;
};

// What a network stores, read from its file
static ParsedRaw parseRaw(const brain::Bytes& data)
{
    ParsedRaw parsed;
    brain::UIType offset = HEADER;
    brain::UIType count;

    (void)brain::tools::get<std::uint64_t>(data, offset); // neurons
    parsed.cap = brain::tools::get<std::uint64_t>(data, offset);
    offset += 3 * sizeof(brain::FType) + sizeof(std::uint8_t); // the size of the space and the limit
    count = brain::tools::get<std::uint64_t>(data, offset);
    for (brain::UIType i = 0; i < count; ++i) {
        brain::RawBrainNeuron neuron;
        neuron.pos.x = brain::tools::get<brain::FType>(data, offset);
        neuron.pos.y = brain::tools::get<brain::FType>(data, offset);
        neuron.bias = brain::tools::get<brain::FType>(data, offset);
        parsed.neurons.push_back(neuron);
    }
    count = brain::tools::get<std::uint64_t>(data, offset);
    for (brain::UIType i = 0; i < count; ++i) {
        brain::RawBrainLink link;
        link.id = brain::tools::get<std::uint64_t>(data, offset);
        link.weight = brain::tools::get<brain::FType>(data, offset);
        link.inward = brain::tools::get<bool>(data, offset);
        parsed.links.push_back(link);
    }
    return parsed;
}

// Every inward link of a neuron has its outward mirror on the sender, with the same weight
static void expectMirrors(const ParsedRaw& parsed)
{
    for (brain::UIType receiver = 0; receiver < parsed.neurons.size(); ++receiver) {
        for (brain::UIType slot = 0; slot < parsed.cap; ++slot) {
            const brain::RawBrainLink& link = parsed.links[receiver * parsed.cap + slot];
            bool found = false;

            if (!link.inward) continue;
            EXPECT_NE(link.id, receiver); // never a link to itself
            for (brain::UIType other = 0; other < parsed.cap; ++other) {
                const brain::RawBrainLink& mirror = parsed.links[link.id * parsed.cap + other];
                found = found || (!mirror.inward && mirror.id == receiver && mirror.weight == link.weight);
            }
            EXPECT_TRUE(found);
        }
    }
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

/* ------------------------------- init ------------------------------- */
TEST(RawBrainNetwork, InitPlacesTheNeuronsInTheSpace) {
    const ParsedRaw parsed = parseRaw(brain::RawBrainNetwork(30, {10.0, 5.0}).serialize());

    ASSERT_EQ(parsed.neurons.size(), 30u);
    for (const brain::RawBrainNeuron& neuron: parsed.neurons) {
        EXPECT_GE(neuron.pos.x, 0.0f);
        EXPECT_LE(neuron.pos.x, 10.0f);
        EXPECT_GE(neuron.pos.y, 0.0f);
        EXPECT_LE(neuron.pos.y, 5.0f);
        EXPECT_EQ(neuron.bias, 0.0f);
    }
}

TEST(RawBrainNetwork, InitLinksEachNeuronToItsNearest) {
    const ParsedRaw parsed = parseRaw(brain::RawBrainNetwork(30).serialize());
    brain::UIType inward = 0;

    ASSERT_EQ(parsed.cap, 3u);
    ASSERT_EQ(parsed.links.size(), 90u); // 3 slots for each neuron
    for (const brain::RawBrainLink& link: parsed.links) {
        inward += link.inward;
        EXPECT_LT(link.id, 30u);
        if (link.inward || link.weight != 0.0f) EXPECT_EQ(link.weight, 1.0f); // 1.0 by default
    }
    EXPECT_GT(inward, 0u);
    expectMirrors(parsed);
}

TEST(RawBrainNetwork, InitNeverLinksTwiceTheSamePair) {
    const ParsedRaw parsed = parseRaw(brain::RawBrainNetwork(30).serialize());

    for (brain::UIType neuron = 0; neuron < 30; ++neuron) {
        for (brain::UIType a = 0; a < parsed.cap; ++a) {
            for (brain::UIType b = a + 1; b < parsed.cap; ++b) {
                const brain::RawBrainLink& first = parsed.links[neuron * parsed.cap + a];
                const brain::RawBrainLink& second = parsed.links[neuron * parsed.cap + b];
                if (first.weight != 0.0f && second.weight != 0.0f) EXPECT_NE(first.id, second.id);
            }
        }
    }
}

TEST(RawBrainNetwork, InitUsesTheGeneratorOfTheLibrary) {
    brain::tools::rand(42);
    const brain::Bytes first = brain::RawBrainNetwork(30).serialize();
    brain::tools::rand(42);
    const brain::Bytes same = brain::RawBrainNetwork(30).serialize();
    brain::tools::rand(43);
    const brain::Bytes other = brain::RawBrainNetwork(30).serialize();

    EXPECT_EQ(first, same);
    EXPECT_NE(first, other);
}

TEST(RawBrainNetwork, InitSetsTheReducedCopyUp) {
    brain::RawBrainNetwork raw(30);
    const ParsedRaw parsed = parseRaw(raw.serialize());
    const brain::Bytes reduced = raw.reduce()->serialize();
    brain::UIType offset = HEADER + 2 * sizeof(std::uint64_t);
    brain::UIType inward = 0;
    brain::UIType used = 0;
    brain::UIType count = brain::tools::get<std::uint64_t>(reduced, offset);

    for (const brain::RawBrainLink& link: parsed.links)
        inward += link.inward;
    offset += count * sizeof(brain::FType); // the biases
    count = brain::tools::get<std::uint64_t>(reduced, offset);
    EXPECT_EQ(count, 90u);
    for (brain::UIType i = 0; i < count; ++i) {
        (void)brain::tools::get<std::uint64_t>(reduced, offset);
        used += brain::tools::get<brain::FType>(reduced, offset) != 0.0f;
    }
    EXPECT_EQ(used, inward); // only the inward links are kept
}

TEST(RawBrainNetwork, ChangingTheLinksCapSetsTheLinksUpAgain) {
    brain::RawBrainNetwork network(30);

    network.linksCap(0.5f);
    EXPECT_EQ(parseRaw(network.serialize()).links.size(), 30u * 15u);
    expectMirrors(parseRaw(network.serialize()));
}

TEST(RawBrainNetwork, WeightInitGivesTheSameWeightToBothSidesOfALink) {
    brain::RawBrainNetwork network(30);

    network.rand(0.25, 0.5);
    network.weightInit();
    const ParsedRaw parsed = parseRaw(network.serialize());
    expectMirrors(parsed);
    for (const brain::RawBrainLink& link: parsed.links) {
        if (link.weight == 0.0f) continue; // a free slot
        EXPECT_GE(link.weight, 0.25f);
        EXPECT_LE(link.weight, 0.5f);
    }
}

TEST(RawBrainNetwork, BiasInitUsesTheRandomRange) {
    brain::RawBrainNetwork network(10);

    network.rand(0.5, 1.0);
    network.biasInit();
    for (const brain::RawBrainNeuron& neuron: parseRaw(network.serialize()).neurons) {
        EXPECT_GE(neuron.bias, 0.5f);
        EXPECT_LE(neuron.bias, 1.0f);
    }
}
