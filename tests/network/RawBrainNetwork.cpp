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
#include "brain/tools/neuron.hpp"
#include "brain/tools/random.hpp"
#include "brain/tools/bytes.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <cmath>

static constexpr brain::UIType HEADER = 28; // magic (2) + activation (1) + sizes (16) + random range (8) + random mode (1)

static bool throwsCode(const std::function<void()>& fn, const utils::exception::ExternalCode code)
{
    try {fn();} catch (const utils::exception::IException& e) {return e.getCode() == static_cast<utils::exception::InternalCode>(code);}
    return false;
}

static bool throwsCode(const std::function<void()>& fn, const utils::exception::InternalCode code)
{
    try {fn();} catch (const utils::exception::IException& e) {return e.getCode() == code;}
    return false;
}

// A valid file of a raw network of 3 neurons with 2 link slots each: 0 receives from 1 and sends to 2, 1 receives from 2, each link and its clone have the same weight (clone: the weight written in the file of the slot is changed)
static brain::Bytes rawData(const brain::UIType broken = 6)
{
    // <id, weight, inward> of the 6 slots
    const brain::RawBrainLink slots[6] = {{1, 0.5f, true}, {2, 0.75f, false}, {0, 0.5f, false}, {2, 0.25f, true}, {0, 0.75f, true}, {1, 0.25f, false}};

    brain::Bytes data = brain::RawBrainNetwork(1).serialize();

    data.resize(HEADER);
    brain::tools::put<std::uint64_t>(data, 3);
    brain::tools::put<std::uint64_t>(data, 2);
    brain::tools::put<brain::FType>(data, 2.0f);
    brain::tools::put<brain::FType>(data, 3.0f);
    brain::tools::put<brain::FType>(data, 0.5f);
    brain::tools::put<std::uint8_t>(data, BRAIN_ACTIVATION_NONE);
    brain::tools::put<brain::FType>(data, 1.0f); // time steps limit
    brain::tools::put<std::uint8_t>(data, BRAIN_ACTIVATION_NONE);
    brain::tools::put<std::uint64_t>(data, 4);
    brain::tools::put<brain::FType>(data, 0.0f); // signal threshold
    brain::tools::put<brain::FType>(data, 0.01f); // weight min
    brain::tools::put<brain::FType>(data, 2.0f); // weight max
    brain::tools::put<std::uint8_t>(data, 0); // orphans
    brain::tools::put<std::uint8_t>(data, 0); // left to right
    brain::tools::put<std::uint64_t>(data, 0); // far links
    brain::tools::put<std::uint64_t>(data, 0); // path at the init
    brain::tools::put<brain::FType>(data, 0); // momentum
    brain::tools::put<brain::FType>(data, 0); // step limit
    brain::tools::put<std::uint8_t>(data, 0); // clamp of the weights
    brain::tools::put<std::uint8_t>(data, 0); // low new links
    brain::tools::put<std::uint8_t>(data, 0); // mini-batch
    brain::tools::put<std::uint8_t>(data, 0); // direct output
    brain::tools::put<std::uint8_t>(data, 0); // keep the paths
    brain::tools::put<brain::FType>(data, 0); // Adam
    brain::tools::put<brain::FType>(data, 0); // rate decay
    brain::tools::put<std::uint8_t>(data, 0); // inverted links
    brain::tools::put<std::uint8_t>(data, 0); // random links
    brain::tools::put<std::uint64_t>(data, 0); // minimum of links
    brain::tools::put<std::uint8_t>(data, 0); // input paths
    brain::tools::put<std::uint8_t>(data, 0); // move the last link
    brain::tools::put<std::uint64_t>(data, 3);
    for (std::uint64_t i = 0; i < 3; ++i) {
        brain::tools::put<brain::FType>(data, static_cast<brain::FType>(i));
        brain::tools::put<brain::FType>(data, 0.5f * static_cast<brain::FType>(i));
        brain::tools::put<brain::FType>(data, 0.1f * static_cast<brain::FType>(i));
    }
    brain::tools::put<std::uint64_t>(data, 6);
    for (std::uint64_t i = 0; i < 6; ++i) {
        brain::tools::put<std::uint64_t>(data, slots[i].id);
        brain::tools::put<brain::FType>(data, i == broken ? slots[i].weight + 1.0f : slots[i].weight);
        brain::tools::put<bool>(data, slots[i].inward);
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
    offset += 3 * sizeof(brain::FType) + sizeof(std::uint8_t) + sizeof(brain::FType) + sizeof(std::uint8_t) + sizeof(std::uint64_t) + sizeof(brain::FType) + 2 * sizeof(brain::FType) + 2 * sizeof(std::uint8_t) + 2 * sizeof(std::uint64_t) + 3 * sizeof(brain::FType) + 5 * sizeof(std::uint8_t) + sizeof(brain::FType) + 4 * sizeof(std::uint8_t) + sizeof(std::uint64_t); // the rate decay, the input paths and the move of the last link, the inverted and random links modes, the minimum of links (after Adam); the size of the space, the links limit, the time steps limit, the signal threshold, the limits of the weights, the orphans and left to right modes, the far links, the path, the momentum, the step limit, the clamp, the new links, the mini-batch, the direct output and the paths modes, the rate of Adam
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
    network->size(2, 1);
    EXPECT_NO_THROW(network->inputs({0.0, 1.0}));
    EXPECT_NO_THROW((void)network->apply());
    EXPECT_NO_THROW(network->backpropagation(0.5));
    EXPECT_NO_THROW(network->weightInit());
}

TEST(RawBrainNetwork, LinksCapIsThePercentageOfTheNeurons) {
    EXPECT_EQ(brain::RawBrainNetwork(100).links(), 30u); // 100 % of 100, the default hook (3 sqrt(100) = 30) is the smallest
    EXPECT_EQ(brain::RawBrainNetwork(2).links(), 1u); // at least one link
    EXPECT_EQ(brain::RawBrainNetwork(1).links(), 0u); // nobody to link to
}

TEST(RawBrainNetwork, LinksCapSetters) {
    brain::RawBrainNetwork network(100);

    network.linksCap(0.1f);
    EXPECT_EQ(network.links(), 10u); // the percentage is the smallest here
    network.linksCap(1.0f);
    EXPECT_EQ(network.links(), 30u); // the default hook: 3 sqrt(100)
    network.linksCap([](const brain::UIType n) {return n * 10;});
    EXPECT_EQ(network.links(), 99u); // never more than the other neurons
    network.linksCap([](const brain::UIType n) {return n / 50;}); // the hook is the smallest here
    EXPECT_EQ(network.links(), 2u);
    network.resetLinksCap();
    EXPECT_EQ(network.links(), 30u);
    EXPECT_THROW(network.linksCap(0.0f), utils::exception::IException);
    EXPECT_THROW(network.linksCap(1.5f), utils::exception::IException);
    EXPECT_THROW(network.linksCap(std::nanf("")), utils::exception::IException);
    EXPECT_THROW(network.linksCap(brain::LinksCapFn{}), utils::exception::IException);
}

TEST(RawBrainNetwork, DefaultHookStaysFiniteOnABigNetwork) {
    // 3 sqrt(n): finite on a big network, under the neurons as soon as n > 9 (the hook is the limit)
    EXPECT_TRUE(std::isfinite(BRAIN_LINKS_CAP_COMPUTING(1e9)));
    EXPECT_DOUBLE_EQ(BRAIN_LINKS_CAP_COMPUTING(100), 30.0);
    EXPECT_LT(BRAIN_LINKS_CAP_COMPUTING(1e6), 0.01 * 1e6); // 3000 links at a million neurons
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

TEST(RawBrainNetwork, DeserializeRefusesALinkWithoutItsClone) {
    brain::RawBrainNetwork network(1);

    for (brain::UIType slot = 0; slot < 6; ++slot) // whichever side of whichever link has an other weight
        EXPECT_TRUE(throwsCode([&] {network.deserialize(rawData(slot));}, utils::exception::ExternalCode::InvalidNetworkFile)) << "slot " << slot;
    EXPECT_NO_THROW(network.deserialize(rawData()));
}

TEST(RawBrainNetwork, DeserializeRefusesABoolThatIsNotOneOrZero) {
    brain::Bytes data = rawData();
    brain::RawBrainNetwork network(1);

    data[data.size() - 1] = 2; // the inward flag of the last slot
    EXPECT_TRUE(throwsCode([&] {network.deserialize(data);}, utils::exception::ExternalCode::InvalidNetworkFile)); // read as true (no undefined bool): the link has no clone any more
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

    ASSERT_EQ(parsed.cap, 16u); // 3 sqrt(30) = 16.4
    ASSERT_EQ(parsed.links.size(), 480u); // 16 slots for each neuron
    for (const brain::RawBrainLink& link: parsed.links) {
        inward += link.inward;
        EXPECT_LT(link.id, 30u);
        if (link.inward || link.weight != 0.0f) EXPECT_EQ(link.weight, 1.0f); // 1.0 by default (weightInit gives the random ones)
    }
    EXPECT_GT(inward, 0u);
    expectMirrors(parsed);
}

TEST(RawBrainNetwork, InitFillsTheCapWithLinksInBothDirections) {
    brain::RawBrainNetwork network(60, {10.0, 10.0});

    network.randomLinks(false); // the default gives each neuron a random number of links

    brain::tools::rand(7);
    network.size(3, 2);
    const ParsedRaw parsed = parseRaw(network.serialize());
    std::vector<brain::UIType> counts;
    brain::UIType inward = 0;
    brain::UIType outward = 0;

    for (brain::UIType neuron = 3; neuron < 58; ++neuron) { // the neurons that are not inputs or outputs
        brain::UIType used = 0;

        for (brain::UIType slot = 0; slot < parsed.cap; ++slot) {
            const brain::RawBrainLink& link = parsed.links[neuron * parsed.cap + slot];
            if (link.weight == 0.0f) continue;
            ++used;
            inward += link.inward;
            outward += !link.inward;
        }
        EXPECT_GE(used, 1u); // at least one link
        EXPECT_LE(used, parsed.cap);
        counts.push_back(used);
    }
    EXPECT_GT(std::count(counts.begin(), counts.end(), parsed.cap), static_cast<std::ptrdiff_t>(counts.size() / 2)); // most neurons are full: as many links as the cap allows
    EXPECT_GT(inward, 0u);
    EXPECT_GT(outward, 0u);
    expectMirrors(parsed);
}

TEST(RawBrainNetwork, InitLinksTheNearestNeurons) {
    brain::RawBrainNetwork network(60, {10.0, 10.0});

    brain::tools::rand(7);
    network.size(3, 2);
    const ParsedRaw parsed = parseRaw(network.serialize());
    brain::UIType nearestLinked = 0;

    for (brain::UIType neuron = 0; neuron < 60; ++neuron) {
        const brain::UIType nearest = brain::tools::sort(neuron, parsed.neurons, 1)[0];

        for (brain::UIType slot = 0; slot < parsed.cap; ++slot)
            nearestLinked += parsed.links[neuron * parsed.cap + slot].weight != 0.0f && parsed.links[neuron * parsed.cap + slot].id == nearest;
    }
    EXPECT_GE(nearestLinked, 45u); // most of the neurons are linked to their nearest (not when it is full or when both are inputs / outputs)
}

TEST(RawBrainNetwork, InitInputsOnlySendAndOutputsOnlyReceive) {
    brain::RawBrainNetwork network(40, {10.0, 10.0});

    network.size(4, 3);
    const ParsedRaw parsed = parseRaw(network.serialize());

    for (brain::UIType neuron = 0; neuron < 40; ++neuron) {
        for (brain::UIType slot = 0; slot < parsed.cap; ++slot) {
            const brain::RawBrainLink& link = parsed.links[neuron * parsed.cap + slot];

            if (link.weight == 0.0f) continue;
            if (neuron < 4) EXPECT_FALSE(link.inward) << "input " << neuron; // an input receives nothing
            if (neuron >= 37) EXPECT_TRUE(link.inward) << "output " << neuron; // an output sends nothing
        }
    }
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
    brain::UIType offset = HEADER + 3 * sizeof(std::uint64_t) + sizeof(brain::FType); // after the neurons, the slots, the time steps and the signal threshold
    brain::UIType inward = 0;
    brain::UIType used = 0;
    brain::UIType count = brain::tools::get<std::uint64_t>(reduced, offset);

    for (const brain::RawBrainLink& link: parsed.links)
        inward += link.inward;
    offset += count * sizeof(brain::FType); // the biases
    count = brain::tools::get<std::uint64_t>(reduced, offset);
    EXPECT_EQ(count, 480u);
    for (brain::UIType i = 0; i < count; ++i) {
        (void)brain::tools::get<std::uint64_t>(reduced, offset);
        used += brain::tools::get<brain::FType>(reduced, offset) != 0.0f;
    }
    EXPECT_EQ(used, inward); // only the inward links are kept
}

TEST(RawBrainNetwork, ANetworkAfterItsInitAndItsWeightInitIsAValidFile) {
    brain::RawBrainNetwork network(30);
    brain::RawBrainNetwork loaded(1);

    ASSERT_NO_THROW(loaded.deserialize(network.serialize())); // the clones have the same weights
    EXPECT_EQ(loaded.serialize(), network.serialize());
    network.rand(0.25, 0.5);
    network.weightInit();
    ASSERT_NO_THROW(loaded.deserialize(network.serialize()));
    EXPECT_EQ(loaded.serialize(), network.serialize());
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

/* ------------------------------- inputs, outputs, execution ------------------------------- */
TEST(RawBrainNetwork, InitPlacesTheInputsAndTheOutputsOnTheEdges) {
    brain::RawBrainNetwork network(20, {10.0, 4.0});
    ParsedRaw parsed;

    network.size(3, 2);
    parsed = parseRaw(network.serialize());
    ASSERT_EQ(parsed.neurons.size(), 20u);
    for (brain::UIType i = 0; i < 3; ++i) { // the first neurons: x min, evenly spread on the height
        EXPECT_FLOAT_EQ(parsed.neurons[i].pos.x, 0.0f);
        EXPECT_FLOAT_EQ(parsed.neurons[i].pos.y, (static_cast<float>(i) + 0.5f) * 4.0f / 3.0f);
    }
    for (brain::UIType j = 0; j < 2; ++j) { // the last neurons: x max
        EXPECT_FLOAT_EQ(parsed.neurons[18 + j].pos.x, 10.0f);
        EXPECT_FLOAT_EQ(parsed.neurons[18 + j].pos.y, (static_cast<float>(j) + 0.5f) * 4.0f / 2.0f);
    }
}

TEST(RawBrainNetwork, InputsReceiveNothingAndOutputsSendNothing) {
    brain::RawBrainNetwork network(30);
    ParsedRaw parsed;

    network.size(4, 3);
    parsed = parseRaw(network.serialize());
    for (brain::UIType neuron = 0; neuron < 30; ++neuron) {
        for (brain::UIType slot = 0; slot < parsed.cap; ++slot) {
            const brain::RawBrainLink& link = parsed.links[neuron * parsed.cap + slot];

            if (neuron < 4) EXPECT_FALSE(link.inward); // an input receives nothing
            if (neuron >= 27 && link.weight != 0.0f) EXPECT_TRUE(link.inward); // an output sends nothing
        }
    }
    expectMirrors(parsed);
}

TEST(RawBrainNetwork, ChangingTheSizesPlacesTheNeuronsAgain) {
    brain::RawBrainNetwork network(20, {10.0, 4.0});
    ParsedRaw parsed;

    network.size(2, 2);
    network.size(5, 1);
    parsed = parseRaw(network.serialize());
    for (brain::UIType i = 0; i < 5; ++i)
        EXPECT_FLOAT_EQ(parsed.neurons[i].pos.x, 0.0f);
    EXPECT_FLOAT_EQ(parsed.neurons[19].pos.x, 10.0f);
    EXPECT_NE(parsed.neurons[5].pos.x, 0.0f); // not an input any more
    EXPECT_THROW(network.size(10, 11), utils::exception::IException);
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{5, 1}));
}

TEST(RawBrainNetwork, TrainingForwardGivesTheOutputsOfTheReducedNetwork) {
    brain::RawBrainNetwork network(12, {10.0, 10.0});
    const brain::Values inputs = {0.1, 0.5, 0.9};
    brain::Values trained;

    network.size(3, 2);
    network.rand(0.2, 0.8);
    network.weightInit();
    network.biasInit();
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    network.inputs(inputs);
    trained = network.apply();
    ASSERT_EQ(trained.size(), 2u);
    EXPECT_EQ(trained, network.reduce()->apply(inputs)); // same time steps, same links: the same outputs
    EXPECT_EQ(trained, network.BrainNetwork::apply(inputs));
}

TEST(RawBrainNetwork, StepsCapIsThePercentageOfTheNeurons) {
    brain::RawBrainNetwork network(10);

    EXPECT_EQ(network.steps(), 3u); // min(10 * 100 %, log2(10) = 3)
    network.stepsCap(0.2f);
    EXPECT_EQ(network.steps(), 2u); // 20 % of 10, under the hook
    network.stepsCap(1.0f);
    EXPECT_EQ(network.steps(), 3u); // the hook of the default is the smallest
    network.stepsCap([](const brain::UIType) {return 7;}); // the hook is the smallest only under the percentage
    EXPECT_EQ(network.steps(), 7u);
    network.stepsCap(0.5f);
    EXPECT_EQ(network.steps(), 5u);
    network.stepsCap([](const brain::UIType) {return 0;});
    EXPECT_EQ(network.steps(), 1u); // at least one step
    network.resetStepsCap();
    EXPECT_EQ(network.steps(), 3u);
    EXPECT_THROW(network.stepsCap(0.0f), utils::exception::IException);
    EXPECT_THROW(network.stepsCap(1.5f), utils::exception::IException);
    EXPECT_THROW(network.stepsCap(brain::StepsCapFn{}), utils::exception::IException);
}

TEST(RawBrainNetwork, StepsAreNeverUnderTheFarthestPathPlusTenPercent) {
    // 60 neurons, 2 inputs and 1 output spread on a wide space: the paths are several links long, one step cannot carry the signal
    brain::RawBrainNetwork network(60, {20.0, 2.0});

    network.stepsCap([](const brain::UIType) {return 1;});
    network.linksCap([](const brain::UIType) {return 3;});
    network.setRules(brain::LEFT_TO_RIGHT | brain::INPUT_PATHS);
    brain::tools::rand(4);
    network.size(2, 1);
    const std::vector<brain::RawBrainLink> links = network.rawLinks();
    const brain::UIType cap = network.links();
    brain::UIType farthest = 0;

    for (brain::UIType from = 0; from < 2; ++from) { // shortest path along the outward links to the output
        std::vector<brain::IType> distance(60, -1);
        std::vector<brain::UIType> queue = {from};

        distance[from] = 0;
        for (brain::UIType head = 0; head < queue.size(); ++head) {
            for (brain::UIType slot = 0; slot < cap; ++slot) {
                const brain::RawBrainLink& link = links[queue[head] * cap + slot];

                if (link.inward || (link.id == 0 && link.weight == 0) || distance[link.id] >= 0) continue;
                distance[link.id] = distance[queue[head]] + 1;
                queue.push_back(link.id);
            }
        }
        if (distance[59] > 0) farthest = std::max(farthest, static_cast<brain::UIType>(distance[59]));
    }
    ASSERT_GT(farthest, 1u);
    EXPECT_EQ(network.steps(), static_cast<brain::UIType>(std::ceil(static_cast<double>(farthest) * 1.1))); // the hook said 1: the floor wins
}

TEST(RawBrainNetwork, StepsAreSerializedAndTheCustomHookIsWarned) {
    brain::RawBrainNetwork network(10);
    brain::RawBrainNetwork other(1);

    network.stepsCap(0.3f);
    other.deserialize(network.serialize());
    EXPECT_EQ(other.steps(), 3u);
    EXPECT_EQ(other.serialize(), network.serialize());
    network.stepsCap([](const brain::UIType) {return 2;}); // a function can't be saved: warned at the load, the number of steps is kept
    other.deserialize(network.serialize());
    EXPECT_EQ(other.steps(), 2u);
}

TEST(RawBrainNetwork, LimitsHaveGetters) {
    brain::RawBrainNetwork network(100);

    EXPECT_FLOAT_EQ(network.linksCap().first, static_cast<brain::FType>(BRAIN_LINKS_CAP_PERCENTAGE));
    EXPECT_TRUE(static_cast<bool>(network.linksCap().second));
    EXPECT_FLOAT_EQ(network.stepsCap().first, static_cast<brain::FType>(BRAIN_STEPS_CAP_PERCENTAGE));
    EXPECT_TRUE(static_cast<bool>(network.stepsCap().second));
    network.linksCap(0.5f);
    network.stepsCap(0.25f);
    EXPECT_FLOAT_EQ(network.linksCap().first, 0.5f);
    EXPECT_FLOAT_EQ(network.stepsCap().first, 0.25f);
    network.linksCap([](const brain::UIType n) {return n / 10;});
    EXPECT_EQ(network.linksCap().second(100), 10u);
}

TEST(RawBrainNetwork, AWeakSignalIsCancelledInTheTrainingForwardToo) {
    brain::RawBrainNetwork network(12, {10.0, 10.0});
    const brain::Values inputs = {0.1, 0.5, 0.9};

    network.size(3, 2);
    network.rand(0.2, 0.8);
    network.weightInit();
    network.threshold(0.9);
    network.inputs(inputs);
    EXPECT_EQ(network.apply(), network.reduce()->apply(inputs)); // the reduced copy has the same threshold
    network.threshold(1e9); // everything computed is cancelled
    network.inputs(inputs);
    for (const brain::FType output: network.apply())
        EXPECT_EQ(output, 0.0f);
    EXPECT_FLOAT_EQ(network.reduce()->threshold(), 1e9f);
}

TEST(RawBrainNetwork, MemoryInfo) {
    brain::RawBrainNetwork ram(10, {10.0, 10.0}, utils::system::MemoryMode::Ram);
    brain::RawBrainNetwork any(10);

    EXPECT_FALSE(ram.onGpu());
    EXPECT_EQ(ram.memoryMode(), utils::system::MemoryMode::Ram);
    EXPECT_EQ(ram.requestedMode(), utils::system::MemoryMode::Ram);
    EXPECT_EQ(ram.gpuMemory(), 0u);
    EXPECT_EQ(ram.gpuName(), "[None]");
    EXPECT_GT(ram.memory(), ram.reduce()->memory()); // the raw network keeps its own memories too
    EXPECT_EQ(any.requestedMode(), utils::system::MemoryMode::Auto);
    EXPECT_EQ(any.onGpu(), any.memoryMode() != utils::system::MemoryMode::Ram);
}

/* ------------------------------- backpropagation ------------------------------- */
// A chain input -> hidden -> output with one link each way, the weights 1.0 (the neurons are at the same place: the order of the ids breaks the ties)
static brain::RawBrainNetwork& chainNetwork(brain::RawBrainNetwork& network)
{
    network.size(1, 1);
    network.internalActivationFunction(brain::ActivationFunction::Linear);
    network.weightLimits(0.01, 100.0);
    return network;
}

TEST(RawBrainNetwork, BackpropagationNeedsAnApplyAndOutputs) {
    brain::RawBrainNetwork network(5, {10.0, 10.0});

    EXPECT_THROW(network.backpropagation(0.5), utils::exception::IException); // outputs not set
    network.size(1, 1);
    EXPECT_THROW(network.backpropagation(0.5), utils::exception::IException); // no apply() yet
    network.inputs(brain::Values{1.0});
    (void)network.apply();
    EXPECT_NO_THROW(network.backpropagation(0.5));
    EXPECT_THROW(network.backpropagation(std::nanf("")), utils::exception::IException);
    EXPECT_THROW(network.backpropagation(brain::Values{0.1, 0.2}), utils::exception::IException); // 2 coefficients for 1 output
    EXPECT_THROW(network.backpropagation(brain::Values{}), utils::exception::IException);
}

TEST(RawBrainNetwork, APositiveScoreStrengthensAndANegativeOneWeakens) {
    for (const brain::FType score: {0.1f, -0.1f}) {
        brain::RawBrainNetwork network(6, {10.0, 10.0});
        const brain::Values inputs = {1.0};

        brain::tools::rand(5);
        chainNetwork(network);
        network.weightInit();
        network.weightLimits(0.0001, 1000.0);
        network.inputs(inputs);
        const brain::Values before = network.apply();
        const brain::Bytes saved = network.serialize();
        network.backpropagation(score);
        network.inputs(inputs);
        const brain::Values after = network.apply();

        EXPECT_NE(network.serialize(), saved) << "score " << score; // the weights moved
        if (before[0] > 0 && after[0] != before[0]) EXPECT_EQ(after[0] > before[0], score > 0) << "score " << score;
        expectMirrors(parseRaw(network.serialize())); // the links and their clones keep the same weight
    }
}

TEST(RawBrainNetwork, AWeakLinkIsRemovedUnlessItIsTheLastOne) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});

    brain::tools::rand(3);
    network.size(2, 2);
    network.weightInit();
    network.weightLimits(1e9, 2e9); // every farthest link is too weak
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    const ParsedRaw before = parseRaw(network.serialize());
    network.backpropagation(0.0);
    const ParsedRaw after = parseRaw(network.serialize());
    brain::UIType linksBefore = 0;
    brain::UIType linksAfter = 0;

    for (const brain::RawBrainLink& link: before.links) linksBefore += link.inward || link.weight != 0.0f;
    for (const brain::RawBrainLink& link: after.links) linksAfter += link.inward || link.weight != 0.0f;
    EXPECT_LT(linksAfter, linksBefore);
    expectMirrors(after);
    for (brain::UIType neuron = 0; neuron < 30; ++neuron) {
        brain::UIType in[2] = {0, 0}; // before, after
        brain::UIType out[2] = {0, 0};

        for (brain::UIType slot = 0; slot < after.cap; ++slot) {
            const brain::RawBrainLink& was = before.links[neuron * after.cap + slot];
            const brain::RawBrainLink& now = after.links[neuron * after.cap + slot];

            if (was.inward) ++in[0];
            else if (was.weight != 0.0f) ++out[0];
            if (now.inward) ++in[1];
            else if (now.weight != 0.0f) ++out[1];
        }
        if (in[0] + out[0] >= 1) EXPECT_GE(in[1] + out[1], 1u) << "neuron " << neuron; // never left alone
        if (neuron < 2 && out[0] >= 1) EXPECT_GE(out[1], 1u) << "input " << neuron; // an input keeps an outward link
        if (neuron >= 2 && in[0] >= 1) EXPECT_GE(in[1], 1u) << "neuron " << neuron; // the others keep an inward link
    }
}

TEST(RawBrainNetwork, OrphansCanBeAllowed) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});

    EXPECT_FALSE(network.orphans());
    network.allowOrphans();
    network.minLinks(0); // the default minimum of 1 keeps the last link of a neuron
    EXPECT_TRUE(network.orphans());
    network.disallowOrphans();
    EXPECT_FALSE(network.orphans());

    brain::tools::rand(3);
    network.size(2, 2);
    network.weightInit();
    network.weightLimits(1e9, 2e9);
    network.allowOrphans();
    for (brain::UIType round = 0; round < 20; ++round) {
        network.inputs(brain::Values{0.5, 0.5});
        (void)network.apply();
        network.backpropagation(0.0);
    }
    const ParsedRaw after = parseRaw(network.serialize());
    brain::UIType empty = 0;

    for (brain::UIType neuron = 0; neuron < 30; ++neuron) {
        brain::UIType used = 0;

        for (brain::UIType slot = 0; slot < after.cap; ++slot)
            used += after.links[neuron * after.cap + slot].inward || after.links[neuron * after.cap + slot].weight != 0.0f;
        empty += used == 0;
    }
    EXPECT_GT(empty, 0u); // some neurons lost every link
    expectMirrors(after);
}

TEST(RawBrainNetwork, AStrongLinkLinksTheNearestFreeNeuron) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});

    brain::tools::rand(3);
    network.size(2, 2);
    network.weightInit();
    network.weightLimits(1e9, 2e9); // a removal pass first: the init fills the cap, the growth needs free slots
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    network.backpropagation(0.0);
    network.weightLimits(0.0, 1e-6); // every link is too strong
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    const ParsedRaw before = parseRaw(network.serialize());
    network.backpropagation(0.0);
    const ParsedRaw after = parseRaw(network.serialize());
    brain::UIType linksBefore = 0;
    brain::UIType linksAfter = 0;

    for (const brain::RawBrainLink& link: before.links) linksBefore += link.inward || link.weight != 0.0f;
    for (const brain::RawBrainLink& link: after.links) linksAfter += link.inward || link.weight != 0.0f;
    EXPECT_GT(linksAfter, linksBefore); // some neurons got a new link
    EXPECT_EQ(after.cap, before.cap); // never over the cap
    for (brain::UIType neuron = 0; neuron < 30; ++neuron) {
        for (brain::UIType slot = 0; slot < after.cap; ++slot) {
            const brain::RawBrainLink& link = after.links[neuron * after.cap + slot];

            if (neuron < 2) EXPECT_FALSE(link.inward) << "input " << neuron;
            if (neuron >= 28) EXPECT_TRUE(link.inward || link.weight == 0.0f) << "output " << neuron;
        }
    }
    expectMirrors(after);
}

TEST(RawBrainNetwork, WeightLimitsAreCheckedAndSerialized) {
    brain::RawBrainNetwork network(10);
    brain::RawBrainNetwork other(10);

    EXPECT_FLOAT_EQ(network.weightMin(), static_cast<brain::FType>(BRAIN_WEIGHT_MIN));
    EXPECT_FLOAT_EQ(network.weightMax(), static_cast<brain::FType>(BRAIN_WEIGHT_MAX));
    network.weightLimits(0.1, 3.0);
    network.weightMin(0.2);
    network.weightMax(4.0);
    EXPECT_FLOAT_EQ(network.weightMin(), 0.2f);
    EXPECT_FLOAT_EQ(network.weightMax(), 4.0f);
    EXPECT_THROW(network.weightMin(-0.1), utils::exception::IException);
    EXPECT_THROW(network.weightMin(5.0), utils::exception::IException); // over the max
    EXPECT_THROW(network.weightMax(0.1), utils::exception::IException); // under the min
    EXPECT_THROW(network.weightLimits(1.0, 1.0), utils::exception::IException);
    EXPECT_THROW(network.weightLimits(std::nanf(""), 1.0), utils::exception::IException);
    EXPECT_FLOAT_EQ(network.weightMin(), 0.2f);
    network.allowOrphans();
    other.deserialize(network.serialize());
    EXPECT_FLOAT_EQ(other.weightMin(), 0.2f);
    EXPECT_FLOAT_EQ(other.weightMax(), 4.0f);
    EXPECT_TRUE(other.orphans());
}

TEST(RawBrainNetwork, ANewLinkIsSmallerThanTheNearestOnesAndInsideTheLimits) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});

    brain::tools::rand(11);
    network.size(2, 2);
    network.weightInit();
    network.weightLimits(1e9, 2e9); // a removal pass first: the init fills the cap, the growth needs free slots
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    network.backpropagation(0.0);
    network.weightLimits(0.0, 1e-6); // every link is too strong: no other link to look at, the reference is the middle of the limits
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    const ParsedRaw before = parseRaw(network.serialize());
    network.backpropagation(0.0);
    const ParsedRaw after = parseRaw(network.serialize());
    brain::UIType created = 0;

    for (brain::UIType neuron = 0; neuron < 30; ++neuron) {
        for (brain::UIType slot = 0; slot < after.cap; ++slot) {
            const brain::RawBrainLink& link = after.links[neuron * after.cap + slot];
            const brain::RawBrainLink& was = before.links[neuron * after.cap + slot];

            if (link.weight == 0.0f || (was.weight != 0.0f && was.id == link.id)) continue; // a free slot, or an old link
            ++created;
            EXPECT_LT(std::abs(link.weight), 1e-6f); // under the max
            EXPECT_GT(std::abs(link.weight), 0.0f); // over the min
        }
    }
    EXPECT_GT(created, 0u);
    expectMirrors(after);
}

TEST(RawBrainNetwork, AnyWeakLinkIsRemovedNotOnlyTheFarthest) {
    brain::RawBrainNetwork network(40, {10.0, 10.0});

    brain::tools::rand(2);
    network.size(2, 2);
    network.weightInit();
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    const ParsedRaw before = parseRaw(network.serialize());

    // make the weights of the nearest links weak: only they trigger
    network.weightLimits(0.5, 1e9);
    network.backpropagation(0.0);
    const ParsedRaw after = parseRaw(network.serialize());
    brain::UIType weak = 0;
    brain::UIType gone = 0;

    for (brain::UIType i = 0; i < before.links.size(); ++i) {
        weak += before.links[i].weight != 0.0f && std::abs(before.links[i].weight) <= 0.5f;
        gone += before.links[i].weight != 0.0f && after.links[i].weight == 0.0f && !after.links[i].inward;
    }
    EXPECT_GT(weak, 0u);
    EXPECT_GT(gone, 0u); // some links were removed, whatever their place in the list
    expectMirrors(after);
}

// The file of a network with other sizes in its header (the input and output sizes: 2 x u64 after the magic bytes and the activation byte)
static brain::Bytes withSizes(brain::Bytes data, const std::uint64_t input, const std::uint64_t output)
{
    for (brain::UIType i = 0; i < 8; ++i) {
        data[3 + i] = static_cast<std::uint8_t>(input >> (8 * i));
        data[11 + i] = static_cast<std::uint8_t>(output >> (8 * i));
    }
    return data;
}

TEST(RawBrainNetwork, AFileWhereAnInputReceivesOrAnOutputSendsIsRefused) {
    brain::RawBrainNetwork network(20, {10.0, 10.0}); // no inputs nor outputs: every neuron sends and receives
    brain::RawBrainNetwork other(2);
    const brain::Bytes data = network.serialize();
    const ParsedRaw parsed = parseRaw(data);
    bool firstReceives = false;
    bool lastSends = false;

    for (brain::UIType slot = 0; slot < parsed.cap; ++slot) {
        const brain::RawBrainLink& first = parsed.links[slot];
        const brain::RawBrainLink& last = parsed.links[19 * parsed.cap + slot];
        firstReceives = firstReceives || first.inward;
        lastSends = lastSends || (!last.inward && (last.id != 0 || last.weight != 0.0f));
    }
    ASSERT_NO_THROW(other.deserialize(data));
    if (firstReceives) EXPECT_TRUE(throwsCode([&] {other.deserialize(withSizes(data, 1, 0));}, utils::exception::ExternalCode::InvalidNetworkFile));
    if (lastSends) EXPECT_TRUE(throwsCode([&] {other.deserialize(withSizes(data, 0, 1));}, utils::exception::ExternalCode::InvalidNetworkFile));
    EXPECT_TRUE(firstReceives || lastSends); // the case is really tested
    network.size(2, 1); // set up again for these sizes: always valid
    EXPECT_NO_THROW(other.deserialize(network.serialize()));
}

// Every inward link comes from a neuron strictly on the left of its receiver
static bool onlyLeftToRight(const ParsedRaw& parsed)
{
    for (brain::UIType owner = 0; owner < parsed.neurons.size(); ++owner) {
        for (brain::UIType slot = 0; slot < parsed.cap; ++slot) {
            const brain::RawBrainLink& link = parsed.links[owner * parsed.cap + slot];
            if (link.inward && !(parsed.neurons[link.id].pos.x < parsed.neurons[owner].pos.x)) return false;
        }
    }
    return true;
}

TEST(RawBrainNetwork, LeftToRightLinksOnlyGoToTheRight) {
    brain::RawBrainNetwork network(40, {10.0, 10.0});

    EXPECT_FALSE(network.leftToRight());
    network.size(3, 2);
    brain::tools::rand(5);
    EXPECT_FALSE(onlyLeftToRight(parseRaw(network.serialize()))); // random directions by default
    network.leftToRight(true);
    EXPECT_TRUE(network.leftToRight());
    const ParsedRaw parsed = parseRaw(network.serialize());
    brain::UIType links = 0;

    EXPECT_TRUE(onlyLeftToRight(parsed));
    for (const brain::RawBrainLink& link: parsed.links) links += link.inward;
    EXPECT_GT(links, 0u);
    expectMirrors(parsed);
    network.leftToRight(false);
    EXPECT_FALSE(network.leftToRight());
}

TEST(RawBrainNetwork, LeftToRightIsSavedAndCheckedAtTheLoad) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});
    brain::RawBrainNetwork other(2);

    brain::tools::rand(9);
    network.size(2, 1);
    network.leftToRight(true);
    other.deserialize(network.serialize());
    EXPECT_TRUE(other.leftToRight());
    EXPECT_EQ(other.serialize(), network.serialize());

    // a network with random directions read as a left to right one: refused
    network.leftToRight(false);
    brain::Bytes data = network.serialize();
    ASSERT_FALSE(onlyLeftToRight(parseRaw(data)));
    data[HEADER + 55] = 1; // the left to right byte, after the neurons, the slots, the space, the limits, the steps, the threshold, the weight limits and the orphans mode
    EXPECT_TRUE(throwsCode([&] {other.deserialize(data);}, utils::exception::ExternalCode::InvalidNetworkFile));
}

TEST(RawBrainNetwork, LeftToRightNewLinksComeFromTheLeft) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});

    brain::tools::rand(4);
    network.size(2, 2);
    network.leftToRight(true);
    network.weightInit();
    network.weightLimits(1e9, 2e9); // a removal pass first: the init fills the cap, the growth needs free slots
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    network.backpropagation(0.0);
    const brain::UIType before = parseRaw(network.serialize()).links.size();
    network.weightLimits(0.0, 1e-6); // every link is too strong: the neurons link the nearest free ones on their left
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    network.backpropagation(0.0);
    const ParsedRaw after = parseRaw(network.serialize());

    EXPECT_EQ(after.links.size(), before);
    EXPECT_TRUE(onlyLeftToRight(after));
    expectMirrors(after);
}

// The neurons each input reaches through inward links (-1: not reached)
static std::vector<brain::IType> reachedFrom(const ParsedRaw& parsed, const brain::UIType from)
{
    std::vector<brain::IType> distance(parsed.neurons.size(), -1);
    std::vector<brain::UIType> queue = {from};

    distance[from] = 0;
    for (brain::UIType head = 0; head < queue.size(); ++head) {
        for (brain::UIType owner = 0; owner < parsed.neurons.size(); ++owner) {
            for (brain::UIType slot = 0; slot < parsed.cap; ++slot) {
                const brain::RawBrainLink& link = parsed.links[owner * parsed.cap + slot];

                if (!link.inward || link.id != queue[head] || distance[owner] >= 0) continue;
                distance[owner] = distance[queue[head]] + 1;
                queue.push_back(owner);
            }
        }
    }
    return distance;
}

// A network whose links never change (limits out of reach), trained on the same input
static std::unique_ptr<brain::RawBrainNetwork> trainingNetwork(const brain::IType seed)
{
    std::unique_ptr<brain::RawBrainNetwork> network = std::make_unique<brain::RawBrainNetwork>(12, std::pair<brain::FType, brain::FType>{4.0, 4.0});

    brain::tools::rand(seed);
    network->size(2, 1);
    network->weightInit();
    network->biasInit();
    network->weightLimits(0.0, 1e9);
    return network;
}

TEST(RawBrainNetwork, MomentumKeepsThePreviousStep) {
    std::unique_ptr<brain::RawBrainNetwork> plain_ = trainingNetwork(3);
    std::unique_ptr<brain::RawBrainNetwork> moving_ = trainingNetwork(3);
    brain::RawBrainNetwork& plain = *plain_;
    brain::RawBrainNetwork& moving = *moving_;

    EXPECT_EQ(moving.momentum(), 0);
    moving.momentum(0.5);
    EXPECT_EQ(moving.momentum(), 0.5);
    EXPECT_TRUE(throwsCode([&] {moving.momentum(1.0);}, utils::exception::InternalCode::InvalidArgument));
    EXPECT_TRUE(throwsCode([&] {moving.momentum(-0.1);}, utils::exception::InternalCode::InvalidArgument));
    for (brain::RawBrainNetwork* network: {&plain, &moving}) {
        network->inputs(brain::Values{0.3, 0.8});
        (void)network->apply();
        network->backpropagation(1.0);
    }
    EXPECT_EQ(plain.serialize().size(), moving.serialize().size());
    EXPECT_EQ(parseRaw(plain.serialize()).neurons[11].bias, parseRaw(moving.serialize()).neurons[11].bias); // the first step: no previous one
    for (brain::RawBrainNetwork* network: {&plain, &moving}) {
        network->inputs(brain::Values{0.3, 0.8});
        (void)network->apply();
        network->backpropagation(1.0);
    }
    EXPECT_NE(parseRaw(plain.serialize()).neurons[11].bias, parseRaw(moving.serialize()).neurons[11].bias);
}

TEST(RawBrainNetwork, StepLimitAndClampKeepTheWeightsInBounds) {
    std::unique_ptr<brain::RawBrainNetwork> owned = trainingNetwork(5);
    brain::RawBrainNetwork& network = *owned;
    ParsedRaw before = parseRaw(network.serialize());

    EXPECT_TRUE(throwsCode([&] {network.stepLimit(-1.0);}, utils::exception::InternalCode::InvalidArgument));
    network.stepLimit(0.01);
    EXPECT_EQ(network.stepLimit(), static_cast<brain::FType>(0.01));
    network.inputs(brain::Values{1.0, 1.0});
    (void)network.apply();
    network.backpropagation(1000.0);
    ParsedRaw after = parseRaw(network.serialize());
    for (brain::UIType k = 0; k < after.links.size(); ++k)
        EXPECT_LE(std::abs(after.links[k].weight - before.links[k].weight), 0.01 + 1e-6);
    for (brain::UIType i = 0; i < after.neurons.size(); ++i)
        EXPECT_LE(std::abs(after.neurons[i].bias - before.neurons[i].bias), 0.01 + 1e-6);

    // clamp: a huge score can't push a weight or a bias over weightMax
    network.stepLimit(0.0);
    network.weightLimits(0.0, 2.0);
    network.clampWeights(true);
    EXPECT_TRUE(network.clampWeights());
    for (int round = 0; round < 5; ++round) {
        network.inputs(brain::Values{1.0, 1.0});
        (void)network.apply();
        network.backpropagation(1000.0);
    }
    after = parseRaw(network.serialize());
    for (const brain::RawBrainLink& link: after.links)
        EXPECT_LE(std::abs(link.weight), 2.0);
    for (const brain::RawBrainNeuron& neuron: after.neurons)
        EXPECT_LE(std::abs(neuron.bias), 2.0);
    expectMirrors(after);
}

TEST(RawBrainNetwork, BatchAppliesTheMeanOfThePendingGradients) {
    std::unique_ptr<brain::RawBrainNetwork> single_ = trainingNetwork(8);
    std::unique_ptr<brain::RawBrainNetwork> batched_ = trainingNetwork(8);
    brain::RawBrainNetwork& single = *single_;
    brain::RawBrainNetwork& batched = *batched_;

    batched.batch(true);
    EXPECT_TRUE(batched.batch());
    for (int i = 0; i < 2; ++i) {
        batched.inputs(brain::Values{0.2, 0.9});
        (void)batched.apply();
        batched.backpropagation(0.5);
    }
    EXPECT_EQ(batched.pending(), 2u);
    batched.batch(false);
    batched.batch(true); // a change of mode drops what is pending
    EXPECT_EQ(batched.pending(), 0u);
    for (int i = 0; i < 2; ++i) {
        batched.inputs(brain::Values{0.2, 0.9});
        (void)batched.apply();
        batched.backpropagation(0.5);
    }
    batched.applyBatch();
    EXPECT_EQ(batched.pending(), 0u);
    batched.applyBatch(); // nothing pending: nothing done
    single.inputs(brain::Values{0.2, 0.9});
    (void)single.apply();
    single.backpropagation(0.5);
    const ParsedRaw a = parseRaw(single.serialize());
    const ParsedRaw b = parseRaw(batched.serialize());
    for (brain::UIType k = 0; k < a.links.size(); ++k)
        EXPECT_NEAR(a.links[k].weight, b.links[k].weight, 1e-5);
    for (brain::UIType i = 0; i < a.neurons.size(); ++i)
        EXPECT_NEAR(a.neurons[i].bias, b.neurons[i].bias, 1e-5);
}

TEST(RawBrainNetwork, LowNewLinksStartAtFivePercentOfTheLimits) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});

    brain::tools::rand(4);
    network.size(2, 2);
    network.weightInit();
    network.lowNewLinks(true);
    EXPECT_TRUE(network.lowNewLinks());
    network.weightLimits(1e9, 2e9); // a removal pass first: the init fills the cap, the growth needs free slots
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    network.backpropagation(0.0);
    const ParsedRaw before = parseRaw(network.serialize());
    network.weightLimits(0.0, 1e-3); // every link is too strong: new links, at 5 % of the limits
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    network.backpropagation(0.0);
    const ParsedRaw after = parseRaw(network.serialize());
    brain::UIType added = 0;

    for (brain::UIType k = 0; k < after.links.size(); ++k) {
        const brain::RawBrainLink& link = after.links[k];

        if (!link.inward || (before.links[k].inward && before.links[k].id == link.id)) continue;
        EXPECT_NEAR(std::abs(link.weight), 0.05e-3, 1e-9);
        ++added;
    }
    EXPECT_GT(added, 0u);
}

TEST(RawBrainNetwork, PathInitLinksEveryInputToEveryOutput) {
    brain::RawBrainNetwork network(80, {20.0, 4.0});

    brain::tools::rand(2);
    network.linksCap([](const brain::UIType) {return static_cast<brain::UIType>(3);});
    network.size(3, 2);
    EXPECT_EQ(network.pathInit(), 0u);
    EXPECT_TRUE(throwsCode([&] {network.pathInit(1);}, utils::exception::InternalCode::InvalidArgument));
    network.pathInit(4);
    EXPECT_EQ(network.pathInit(), 4u);
    const ParsedRaw parsed = parseRaw(network.serialize());

    for (brain::UIType from = 0; from < 3; ++from) {
        const std::vector<brain::IType> distance = reachedFrom(parsed, from);

        EXPECT_GE(distance[78], 1);
        EXPECT_GE(distance[79], 1);
    }
    expectMirrors(parsed);
}

TEST(RawBrainNetwork, FarLinksAndTheTrainingModesAreSaved) {
    brain::RawBrainNetwork network(40, {10.0, 10.0});
    brain::RawBrainNetwork other(2);

    brain::tools::rand(6);
    network.size(2, 1);
    network.farLinks(2);
    EXPECT_EQ(network.farLinks(), 2u);
    expectMirrors(parseRaw(network.serialize()));
    network.pathInit(3);
    network.momentum(0.9);
    network.stepLimit(0.3);
    network.clampWeights(true);
    network.lowNewLinks(true);
    network.batch(true);
    network.directOutput(true);
    network.keepPaths(true);
    network.adam(0.01);
    other.deserialize(network.serialize());
    EXPECT_EQ(other.farLinks(), 2u);
    EXPECT_EQ(other.pathInit(), 3u);
    EXPECT_EQ(other.momentum(), static_cast<brain::FType>(0.9));
    EXPECT_EQ(other.stepLimit(), static_cast<brain::FType>(0.3));
    EXPECT_TRUE(other.clampWeights());
    EXPECT_TRUE(other.lowNewLinks());
    EXPECT_TRUE(other.batch());
    EXPECT_TRUE(other.directOutput());
    EXPECT_TRUE(other.keepPaths());
    EXPECT_EQ(other.adam(), static_cast<brain::FType>(0.01));
    EXPECT_EQ(other.serialize(), network.serialize());

    // a momentum of 1 is refused at the load
    brain::Bytes data = network.serialize();
    const brain::FType one = 1.0;
    std::memcpy(data.data() + HEADER + 56 + 2 * sizeof(std::uint64_t), &one, sizeof(brain::FType)); // the momentum, after the left to right byte, the far links and the path
    EXPECT_TRUE(throwsCode([&] {other.deserialize(data);}, utils::exception::ExternalCode::InvalidNetworkFile));
}

TEST(RawBrainNetwork, DirectOutputStillLearnsWhenTheOutputIsSaturated) {
    std::unique_ptr<brain::RawBrainNetwork> plain_ = trainingNetwork(9);
    std::unique_ptr<brain::RawBrainNetwork> direct_ = trainingNetwork(9);
    brain::RawBrainNetwork& plain = *plain_;
    brain::RawBrainNetwork& direct = *direct_;

    direct.directOutput(true);
    EXPECT_TRUE(direct.directOutput());
    plain.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    direct.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    // the output pushed to a saturated value by a huge negative score first (its slope is then ~0)
    for (brain::RawBrainNetwork* network: {&plain, &direct}) {
        for (int i = 0; i < 3; ++i) {
            network->inputs(brain::Values{0.5, 0.5});
            (void)network->apply();
            network->backpropagation(-1e4);
        }
    }
    const brain::FType plainBefore = parseRaw(plain.serialize()).neurons[11].bias;
    const brain::FType directBefore = parseRaw(direct.serialize()).neurons[11].bias;
    for (brain::RawBrainNetwork* network: {&plain, &direct}) {
        network->inputs(brain::Values{0.5, 0.5});
        (void)network->apply();
        network->backpropagation(1.0);
    }
    EXPECT_LT(std::abs(parseRaw(plain.serialize()).neurons[11].bias - plainBefore), 1e-6); // the slope of a saturated sigmoid: nothing moves
    EXPECT_NEAR(parseRaw(direct.serialize()).neurons[11].bias - directBefore, 1.0, 1e-4); // the score itself
}

TEST(RawBrainNetwork, KeepPathsNeverCutsTheLastPathOfAnInput) {
    for (const bool keep: {false, true}) {
        brain::RawBrainNetwork network(30, {10.0, 10.0});
        brain::UIType joined = 0;

        brain::tools::rand(4);
        network.size(2, 2);
        network.allowOrphans(); // only the paths hold the links back
        network.keepPaths(keep);
        EXPECT_EQ(network.keepPaths(), keep);
        network.weightLimits(1e9, 2e9); // every link is too weak: all of them would be removed
        network.inputs(brain::Values{0.5, 0.5});
        (void)network.apply();
        network.backpropagation(0.0);
        const ParsedRaw parsed = parseRaw(network.serialize());
        for (brain::UIType from = 0; from < 2; ++from) {
            const std::vector<brain::IType> distance = reachedFrom(parsed, from);
            joined += (distance[28] >= 0) + (distance[29] >= 0);
        }
        EXPECT_EQ(joined, keep ? 4u : 0u);
        expectMirrors(parsed);
    }
}

TEST(RawBrainNetwork, AdamStepsByItsRateWhateverTheScore) {
    std::unique_ptr<brain::RawBrainNetwork> big_ = trainingNetwork(12);
    std::unique_ptr<brain::RawBrainNetwork> small_ = trainingNetwork(12);
    brain::RawBrainNetwork& big = *big_;
    brain::RawBrainNetwork& small = *small_;
    const ParsedRaw start = parseRaw(big.serialize());
    brain::UIType moved = 0;

    EXPECT_TRUE(throwsCode([&] {big.adam(-0.1);}, utils::exception::InternalCode::InvalidArgument));
    big.adam(0.1);
    small.adam(0.1);
    big.momentum(0.9); // replaced by Adam
    EXPECT_EQ(big.adam(), static_cast<brain::FType>(0.1));
    big.inputs(brain::Values{0.3, 0.8});
    small.inputs(brain::Values{0.3, 0.8});
    (void)big.apply();
    (void)small.apply();
    big.backpropagation(1000.0);
    small.backpropagation(0.001);
    const ParsedRaw a = parseRaw(big.serialize());
    const ParsedRaw b = parseRaw(small.serialize());

    // the first step of Adam is the rate in the direction of the gradient: the same move for both scores
    for (brain::UIType k = 0; k < a.links.size(); ++k) {
        if (!a.links[k].inward || a.links[k].weight == start.links[k].weight) continue;
        EXPECT_NEAR(std::abs(a.links[k].weight - start.links[k].weight), 0.1, 1e-3);
        EXPECT_NEAR(a.links[k].weight, b.links[k].weight, 1e-3);
        ++moved;
    }
    EXPECT_GT(moved, 0u);
    expectMirrors(a);
}

TEST(RawBrainNetwork, ReviveResetsASaturatedNeuron) {
    std::unique_ptr<brain::RawBrainNetwork> owned = trainingNetwork(9);
    brain::RawBrainNetwork& network = *owned;

    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    EXPECT_EQ(network.revive(), 0u); // nothing executed yet
    for (int i = 0; i < 3; ++i) { // a huge negative score saturates the output at 0
        network.inputs(brain::Values{0.5, 0.5});
        (void)network.apply();
        network.backpropagation(-1e4);
    }
    (void)network.revive(); // the watch starts after the saturation
    for (const brain::Values& inputs: {brain::Values{0.0, 1.0}, brain::Values{1.0, 0.0}, brain::Values{0.5, 0.5}}) {
        network.inputs(inputs);
        (void)network.apply();
    }
    EXPECT_GT(network.revive(), 0u);
    const ParsedRaw parsed = parseRaw(network.serialize());
    EXPECT_EQ(parsed.neurons[11].bias, 0.0f);
    expectMirrors(parsed);
    EXPECT_EQ(network.revive(), 0u); // the watch starts again after each call
}

TEST(RawBrainNetwork, SequenceGivesTheInputsOfEachStep) {
    std::unique_ptr<brain::RawBrainNetwork> owned = trainingNetwork(4);
    brain::RawBrainNetwork& network = *owned;
    const brain::UIType steps = network.steps();
    const brain::Values constant = {0.3, 0.8};

    // the same inputs at every step: the same as inputs()
    network.inputs(constant);
    const brain::Values plain = network.apply();
    network.sequence(brain::Dataset(steps + 1, constant));
    ASSERT_EQ(plain.size(), 1u);
    EXPECT_NEAR(network.apply()[0], plain[0], 1e-4); // the sequence runs on the host, the plain call on the gpu when there is one
    EXPECT_NEAR(network.applySequence(brain::Dataset(steps + 1, constant))[0], plain[0], 1e-4);

    // a pulse: the inputs at the first step only, then 0 (what the network kept)
    const brain::Dataset pulse = {constant};
    network.sequence(pulse);
    const brain::Values trained = network.apply();
    ASSERT_EQ(network.states().size(), steps + 1);
    EXPECT_EQ(network.states()[1][0], 0.0f); // the inputs are silent after the sequence
    EXPECT_EQ(network.states()[0][0], constant[0]);
    EXPECT_EQ(network.applySequence(pulse), trained);
    EXPECT_NE(trained, plain);
    network.backpropagation(0.5); // the gradient of a sequence goes back through its steps
    network.inputs(constant); // cancels the sequence
    (void)network.apply();
    EXPECT_EQ(network.states()[1][0], constant[0]);
    EXPECT_THROW(network.sequence({}), utils::exception::IException);
    EXPECT_THROW(network.sequence({{0.1}}), utils::exception::IException);
}

TEST(RawBrainNetwork, ManyInputsAndOutputsInit) {
    // 35 inputs and 10 outputs on 85 neurons (the digits of the lab): every path asked, kept, the links of the inputs full
    for (const brain::IType seed: {1, 2, 3, 7, 42, 48282}) {
        brain::RawBrainNetwork network(85, {8.0f, 5.0f});

        network.weightLimits(0.01f, 5.0f);
        network.rand(-3.0f, 3.0f);
        network.allowOrphans();
        network.keepPaths(true);
        network.lowNewLinks(true);
        network.pathInit(3);
        network.adam(0.03f);
        network.batch(true);
        brain::tools::rand(seed);
        ASSERT_NO_THROW(network.size(35, 10));
        network.weightInit();
        network.biasInit();
        network.inputs(brain::Values(35, 1.0f));
        EXPECT_EQ(network.apply().size(), 10u);
        network.backpropagation(brain::Values(10, 0.1f));
        network.applyBatch();
    }
}

// ---------- rate decay, inverted links, random links, minimum of links ----------

static bool sameLinks(const std::vector<brain::RawBrainLink>& a, const std::vector<brain::RawBrainLink>& b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const brain::RawBrainLink& x, const brain::RawBrainLink& y) {return x.id == y.id && x.weight == y.weight && x.inward == y.inward;});
}

TEST(RawBrainNetwork, RateDecayShrinksTheStepsAndIsSerialized) {
    std::unique_ptr<brain::RawBrainNetwork> constant_ = trainingNetwork(3);
    std::unique_ptr<brain::RawBrainNetwork> decayed_ = trainingNetwork(3);
    brain::RawBrainNetwork& constant = *constant_;
    brain::RawBrainNetwork& decayed = *decayed_;
    const brain::Values inputs = {0.4, 0.9};
    const std::function<brain::FType(brain::RawBrainNetwork&, const std::vector<brain::RawBrainLink>&)> moved = [&](brain::RawBrainNetwork& network, const std::vector<brain::RawBrainLink>& start) { // the distance the weights went since the start
        const std::vector<brain::RawBrainLink> now = network.rawLinks();
        brain::FType sum = 0;
        for (brain::UIType k = 0; k < now.size(); ++k) sum += now[k].id == start[k].id && now[k].inward == start[k].inward ? std::abs(now[k].weight - start[k].weight) : 0; // the links that stayed
        return sum;
    };

    EXPECT_EQ(decayed.rateDecay(), 0);
    EXPECT_TRUE(throwsCode([&] {decayed.rateDecay(-1.0);}, utils::exception::InternalCode::InvalidArgument));
    decayed.rateDecay(1.0);
    EXPECT_EQ(decayed.rateDecay(), 1.0f);
    const std::vector<brain::RawBrainLink> start = constant.rawLinks();
    ASSERT_TRUE(sameLinks(start, decayed.rawLinks()));
    // the first update is the same, the next ones are divided by 1 + decay * updates (2, then 3...)
    for (brain::UIType round = 0; round < 4; ++round) {
        for (brain::RawBrainNetwork* network: {&constant, &decayed}) {
            brain::tools::rand(77); // the same draws for the links that change
            network->inputs(inputs);
            (void)network->apply();
            network->backpropagation(brain::Values{0.01});
        }
        if (round == 0) EXPECT_NEAR(moved(constant, start), moved(decayed, start), 1e-3); // the same first update (the decayed one is applied on the host, the other on the gpu when there is one)
    }
    EXPECT_GT(moved(constant, start), 0.0f);
    EXPECT_LT(moved(decayed, start), moved(constant, start));
    brain::RawBrainNetwork loaded(12, {4.0, 4.0});
    loaded.size(2, 1);
    loaded.deserialize(decayed.serialize());
    EXPECT_EQ(loaded.rateDecay(), 1.0f);
}

TEST(RawBrainNetwork, MinLinksAndRandomLinksAreSerializedAndShapeTheInit) {
    brain::RawBrainNetwork network(30, {8.0, 5.0});
    const std::function<std::vector<brain::UIType>()> count = [&] {
        std::vector<brain::UIType> out;
        for (brain::UIType owner = 0; owner < 30; ++owner) {
            brain::UIType used = 0;
            for (brain::UIType slot = 0; slot < network.links(); ++slot) used += network.rawLinks()[owner * network.links() + slot].weight != 0 ? 1 : 0;
            out.push_back(used);
        }
        return out;
    };

    EXPECT_TRUE(network.randomLinks()); // the defaults: a random number of links, one at least
    EXPECT_EQ(network.minLinks(), 1u);
    network.randomLinks(false);
    brain::tools::rand(5);
    network.size(2, 1);
    const std::vector<brain::UIType> full = count();
    EXPECT_FALSE(network.randomLinks());
    network.minLinks(3);
    EXPECT_EQ(network.minLinks(), 3u);
    EXPECT_EQ(count(), full); // no change without the random init
    network.randomLinks(true);
    EXPECT_TRUE(network.randomLinks());
    const std::vector<brain::UIType> random = count();
    EXPECT_NE(random, full); // fewer links than the cap on some neurons
    for (const brain::UIType used: random) EXPECT_LE(used, network.links());
    brain::RawBrainNetwork loaded(30, {8.0, 5.0});
    loaded.size(2, 1);
    loaded.deserialize(network.serialize());
    EXPECT_TRUE(loaded.randomLinks());
    EXPECT_EQ(loaded.minLinks(), 3u);
}

TEST(RawBrainNetwork, MinLinksKeepsTheLinksOfAWeakNeuron) {
    for (const brain::UIType minimum: {0u, 4u}) {
        brain::RawBrainNetwork network(12, {4.0, 4.0});

        network.allowOrphans();
        network.weightLimits(0.5f, 5.0f);
        network.minLinks(minimum);
        brain::tools::rand(4);
        network.size(2, 1);
        network.weightInit();
        network.inputs({0.5, 0.5});
        (void)network.apply();
        // every link under weightMin: the ones the minimum protects stay
        network.backpropagation(brain::Values{0.0});
        std::vector<brain::RawBrainLink> links = network.rawLinks();
        brain::UIType before = 0;
        for (const brain::RawBrainLink& link: links) before += link.weight != 0 ? 1 : 0;
        EXPECT_GT(before, 0u);
        EXPECT_EQ(network.minLinks(), minimum);
    }
}

TEST(RawBrainNetwork, InvertedLinksSwapsTheLimits) {
    // the same network and the same training: the links that change at the limits are not the same in the two modes
    std::vector<std::vector<brain::RawBrainLink>> result;

    for (const bool inverted: {false, true}) {
        brain::RawBrainNetwork network(14, {4.0, 4.0});

        network.allowOrphans();
        network.weightLimits(0.01f, 1.0f);
        network.invertedLinks(inverted);
        EXPECT_EQ(network.invertedLinks(), inverted);
        brain::tools::rand(9);
        network.size(2, 1);
        network.weightInit();
        network.biasInit();
        brain::tools::rand(11);
        for (brain::UIType round = 0; round < 40; ++round) {
            network.inputs({1.0, 0.0});
            (void)network.apply();
            network.backpropagation(brain::Values{1.0});
        }
        result.push_back(network.rawLinks());
    }
    EXPECT_FALSE(sameLinks(result[0], result[1]));
    brain::RawBrainNetwork loaded(14, {4.0, 4.0});
    loaded.size(2, 1);
    loaded.invertedLinks(true);
    brain::RawBrainNetwork copy(14, {4.0, 4.0});
    copy.size(2, 1);
    copy.deserialize(loaded.serialize());
    EXPECT_TRUE(copy.invertedLinks());
}

// ---------- the rules as flags, the paths of the inputs, the move of the last link ----------

static brain::UIType inputsReaching(brain::RawBrainNetwork& network) // inputs with a path (along the outward links) to at least one output
{
    const std::vector<brain::RawBrainLink> links = network.rawLinks();
    const brain::UIType count = network.neurons().second;
    const brain::UIType cap = network.links();
    brain::UIType reaching = 0;

    for (brain::UIType from = 0; from < network.input(); ++from) {
        std::vector<bool> seen(count, false);
        std::vector<brain::UIType> queue = {from};
        bool found = false;

        seen[from] = true;
        for (brain::UIType head = 0; head < queue.size() && !found; ++head) {
            for (brain::UIType slot = 0; slot < cap; ++slot) {
                const brain::RawBrainLink& link = links[queue[head] * cap + slot];

                if (link.inward || (link.id == 0 && link.weight == 0) || seen[link.id]) continue;
                seen[link.id] = true;
                queue.push_back(link.id);
                found = found || link.id >= count - network.output();
            }
        }
        reaching += found ? 1 : 0;
    }
    return reaching;
}

TEST(RawBrainNetwork, RulesAreFlagsLikeTheCli) {
    brain::RawBrainNetwork network(12, {4.0, 4.0});

    EXPECT_EQ(network.getRules(), brain::rules::DEFAULT); // the defaults of a new network
    EXPECT_EQ(brain::rules::DEFAULT, static_cast<std::uint32_t>(brain::RANDOM_LINKS));
    network.addRules(brain::KEEP_PATHS | brain::ORPHANS);
    EXPECT_EQ(network.getRules(), brain::rules::DEFAULT | brain::KEEP_PATHS | brain::ORPHANS);
    EXPECT_TRUE(network.keepPaths());
    EXPECT_TRUE(network.orphans());
    network.subRules(brain::ORPHANS);
    EXPECT_FALSE(network.orphans());
    network.allowOrphans(); // the single setters are the same bits
    EXPECT_NE(network.getRules() & brain::ORPHANS, 0u);
    network.clearRules(); // everything off
    EXPECT_EQ(network.getRules(), 0u);
    EXPECT_FALSE(network.randomLinks());
    network.setRules(brain::rules::LAB); // the rules of the lab
    EXPECT_EQ(network.getRules(), brain::rules::LAB);
    EXPECT_TRUE(network.batch());
    EXPECT_TRUE(network.directOutput());
    EXPECT_FALSE(network.randomLinks());
    network.resetRules(); // back to the defaults of a new network
    EXPECT_EQ(network.getRules(), brain::rules::DEFAULT);
    EXPECT_FALSE(network.batch());
    network.setRules(brain::rules::ALL);
    EXPECT_EQ(network.getRules(), brain::rules::ALL);
    EXPECT_THROW(network.setRules(1u << 20), utils::exception::IException); // a bit that is not a rule
    EXPECT_EQ(network.getRules(), brain::rules::ALL); // nothing changed by a refused set

    // saved and loaded: the two newest rules too
    network.setRules(brain::INPUT_PATHS | brain::MOVE_LAST_LINK | brain::BATCH);
    brain::RawBrainNetwork loaded(12, {4.0, 4.0});
    loaded.deserialize(network.serialize());
    EXPECT_EQ(loaded.getRules(), brain::INPUT_PATHS | brain::MOVE_LAST_LINK | brain::BATCH);
}

TEST(RawBrainNetwork, InputPathsGivesEveryInputAWayToAnOutput) {
    for (const brain::IType seed: {1, 2, 3, 4, 5, 6}) {
        brain::RawBrainNetwork network(40, {10.0, 10.0});

        network.linksCap([](const brain::UIType) {return 3;}); // few links: some inputs are cut off from the outputs
        network.setRules(brain::INPUT_PATHS);
        brain::tools::rand(seed);
        network.size(6, 2);
        EXPECT_EQ(inputsReaching(network), 6u) << "seed " << seed;
    }
}

TEST(RawBrainNetwork, InputPathsAreNeverCutByTheTraining) {
    // every weight is under the minimum: everything that may be removed is
    brain::UIType with = 0;
    brain::UIType without = 0;

    for (const bool rule: {false, true}) {
        brain::RawBrainNetwork network(40, {10.0, 10.0});

        network.linksCap([](const brain::UIType) {return 4;});
        network.setRules(brain::ORPHANS | (rule ? brain::INPUT_PATHS : 0u));
        network.minLinks(0);
        network.weightLimits(1e9f, 2e9f);
        brain::tools::rand(3);
        network.size(6, 2);
        network.weightInit();
        for (brain::UIType round = 0; round < 20; ++round) {
            network.inputs(brain::Values(6, 0.5f));
            (void)network.apply();
            network.backpropagation(brain::Values{0.0, 0.0});
        }
        (rule ? with : without) = inputsReaching(network);
    }
    EXPECT_EQ(with, 6u); // the rule keeps a path for every input
    EXPECT_LT(without, 6u); // without it the training cuts them
}

TEST(RawBrainNetwork, MoveLastLinkMovesTheLinkInsteadOfKeepingIt) {
    // every weight is under the minimum, the orphans are refused: the last link of a neuron is kept, or moved with the rule
    std::vector<std::vector<brain::RawBrainLink>> result;

    for (const bool rule: {false, true}) {
        brain::RawBrainNetwork network(30, {10.0, 10.0});

        network.linksCap([](const brain::UIType) {return 4;});
        network.setRules(rule ? brain::MOVE_LAST_LINK : 0u);
        network.weightLimits(1e9f, 2e9f);
        brain::tools::rand(5);
        network.size(3, 2);
        network.weightInit();
        for (brain::UIType round = 0; round < 10; ++round) {
            network.inputs(brain::Values(3, 0.5f));
            (void)network.apply();
            network.backpropagation(brain::Values{0.0, 0.0});
        }
        result.push_back(network.rawLinks());
        expectMirrors(parseRaw(network.serialize())); // every link still has its clone with the same weight
        // no neuron ends without a link, whichever the rule
        const brain::UIType cap = network.links();
        for (brain::UIType owner = 0; owner < 30; ++owner) {
            brain::UIType used = 0;
            for (brain::UIType slot = 0; slot < cap; ++slot) used += (result.back()[owner * cap + slot].inward || result.back()[owner * cap + slot].weight != 0) ? 1 : 0;
            EXPECT_GE(used, 1u) << "neuron " << owner << " rule " << rule;
        }
    }
    EXPECT_FALSE(sameLinks(result[0], result[1])); // the last links were moved
    brain::UIType moved = 0;
    for (brain::UIType k = 0; k < result[0].size(); ++k) moved += (result[0][k].id != result[1][k].id || result[0][k].inward != result[1][k].inward) ? 1 : 0;
    EXPECT_GT(moved, 0u);
}

TEST(RawBrainNetwork, SettledStepsAreCopiesAndTheResultIsTheSame) {
    // a network without loop settles after its depth: the steps after that are the same to the bit (copied, not computed again)
    brain::RawBrainNetwork network(40, {10.0, 10.0}, utils::system::MemoryMode::Ram); // the host steps (the gpu has its own kernels)

    network.setRules(brain::LEFT_TO_RIGHT);
    network.stepsCap([](const brain::UIType) {return 30;}); // more steps than the depth
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    network.rand(-1.0f, 1.0f);
    brain::tools::rand(8);
    network.size(3, 2);
    network.weightInit();
    network.biasInit();
    const brain::Values inputs = {0.2f, 0.9f, 0.5f};

    network.inputs(inputs);
    const brain::Values trained = network.apply();
    const std::vector<brain::Values> states = network.states();

    ASSERT_EQ(states.size(), network.steps() + 1);
    EXPECT_EQ(states[states.size() - 2], states.back()); // settled
    EXPECT_EQ(network.apply(inputs), trained); // the execution that stores nothing
    EXPECT_EQ(network.reduce()->apply(inputs), trained); // the reduced network
    EXPECT_EQ(network.apply(), trained); // again: the buffers of the last call are reused
    EXPECT_EQ(network.states(), states);
}

TEST(RawBrainNetwork, TheSignalThresholdIsTheToleranceOfTheEarlyStop) {
    // small weights: the states converge a little more at every step, a threshold stops the steps when nothing moves by more than it
    brain::RawBrainNetwork exact(30, {10.0, 10.0}, utils::system::MemoryMode::Ram); // the host steps (the gpu has its own kernels)
    brain::RawBrainNetwork early(30, {10.0, 10.0}, utils::system::MemoryMode::Ram);
    const brain::Values inputs = {0.3f, 0.8f};

    for (brain::RawBrainNetwork* network: {&exact, &early}) {
        network->stepsCap([](const brain::UIType) {return 40;});
        network->internalActivationFunction(brain::ActivationFunction::Sigmoid);
        network->rand(-0.15f, 0.15f);
        brain::tools::rand(6);
        network->size(2, 1);
        network->weightInit();
        network->biasInit();
    }
    early.threshold(2e-2f);
    exact.inputs(inputs);
    early.inputs(inputs);
    const brain::Values slow = exact.apply();
    const brain::Values fast = early.apply();
    const std::vector<brain::Values> exactStates = exact.states();
    const std::vector<brain::Values> earlyStates = early.states();

    const brain::UIType steps = early.steps();

    ASSERT_EQ(earlyStates.size(), steps + 1);
    ASSERT_EQ(exactStates.size(), steps + 1);
    ASSERT_EQ(slow.size(), 1u);
    EXPECT_NEAR(fast[0], slow[0], 3e-2f); // the same outputs to about the threshold
    // the early one stopped computing at a step where nothing moved: its last states are copies, the exact one is still moving or settled later
    brain::UIType copiedEarly = 0;
    brain::UIType copiedExact = 0;
    for (brain::UIType step = 1; step <= steps; ++step) {
        copiedEarly += earlyStates[step] == earlyStates[step - 1] ? 1u : 0u;
        copiedExact += exactStates[step] == exactStates[step - 1] ? 1u : 0u;
    }
    EXPECT_GT(copiedEarly, copiedExact);
}

TEST(RawBrainNetwork, ApplyAllGivesTheOutputsOfEverySample) {
    brain::RawBrainNetwork network(50, {10.0, 10.0}, utils::system::MemoryMode::Ram); // the host steps
    brain::Dataset samples;

    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    brain::tools::rand(12);
    network.size(4, 3);
    network.weightInit();
    network.biasInit();
    for (brain::UIType s = 0; s < 7; ++s) samples.push_back({0.1f * static_cast<brain::FType>(s), 0.9f - 0.1f * static_cast<brain::FType>(s), 0.5f, static_cast<brain::FType>(s % 2)});
    const brain::Dataset together = network.applyAll(samples);

    ASSERT_EQ(together.size(), samples.size());
    for (brain::UIType s = 0; s < samples.size(); ++s) {
        const brain::Values alone = network.apply(samples[s]);

        ASSERT_EQ(together[s].size(), 3u);
        for (brain::UIType o = 0; o < 3; ++o) EXPECT_NEAR(together[s][o], alone[o], 1e-6f) << "sample " << s << " output " << o; // the same sums in the same order
    }
    EXPECT_EQ(network.applyAll({}).size(), 0u);
    EXPECT_EQ(network.applyAll({samples[2]}).size(), 1u);
    EXPECT_THROW((void)network.applyAll({samples[0], {0.1f}}), utils::exception::IException); // a sample of the wrong size
    // relu, linear, a function of the user and a threshold: the same
    network.threshold(1e-4f);
    for (const brain::ActivationFunction function: {brain::ActivationFunction::ReLU, brain::ActivationFunction::Linear}) {
        network.internalActivationFunction(function);
        network.rand(-0.05f, 0.05f);
        const brain::Dataset many = network.applyAll(samples);

        for (brain::UIType s = 0; s < samples.size(); ++s) EXPECT_NEAR(many[s][0], network.apply(samples[s])[0], 1e-5f);
    }
    network.activationFunction([](const brain::FType value) {return std::tanh(value);});
    const brain::Dataset custom = network.applyAll(samples);
    for (brain::UIType s = 0; s < samples.size(); ++s) EXPECT_NEAR(custom[s][0], network.apply(samples[s])[0], 1e-5f);
}

TEST(RawBrainNetwork, ThreadsGiveTheSameStatesAsOneThread) {
    brain::RawBrainNetwork network(300, {10.0, 10.0}, utils::system::MemoryMode::Ram); // the host steps, several chunks of neurons
    const brain::Values inputs = {0.3f, 0.6f, 0.9f};

    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    network.stepsCap([](const brain::UIType) {return 12;});
    brain::tools::rand(14);
    network.size(3, 2);
    network.weightInit();
    network.biasInit();
    EXPECT_EQ(network.threads(), 0u); // automatic by default: a small network stays on one thread
    network.threads(1);
    network.inputs(inputs);
    const brain::Values single = network.apply();
    const std::vector<brain::Values> singleStates = network.states();
    const brain::Values singleExec = network.apply(inputs);

    for (const brain::UIType threads: {2u, 3u, 7u}) { // every neuron is computed by one thread with its own sum: the same bits
        network.threads(threads);
        EXPECT_EQ(network.threads(), threads);
        EXPECT_EQ(network.apply(), single) << threads << " threads";
        EXPECT_EQ(network.states(), singleStates) << threads << " threads";
        EXPECT_EQ(network.apply(inputs), singleExec) << threads << " threads";
    }
    // a sequence and a function of the user stay on one thread, and give their result
    network.threads(4);
    network.sequence({inputs, {0.1f, 0.2f, 0.3f}});
    EXPECT_EQ(network.apply().size(), 2u);
    network.activationFunction([](const brain::FType value) {return std::tanh(value);});
    network.inputs(inputs);
    EXPECT_EQ(network.apply().size(), 2u);
}

/* ---------------------- mini-batch (samples together) ---------------------- */
static brain::Dataset batchSamples(const brain::UIType count, const brain::UIType size, const brain::UIType seed)
{
    brain::Dataset out(count, brain::Values(size));

    for (brain::UIType i = 0; i < count; ++i)
        for (brain::UIType j = 0; j < size; ++j) out[i][j] = std::sin(static_cast<brain::FType>(seed + i * 17 + j * 5)); // varied, in [-1, 1]
    return out;
}

static void nearValues(const brain::Values& a, const brain::Values& b)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 2e-4 + 2e-4 * std::abs(b[i])) << "value " << i; // float rounding: the sums are not made in the same order
}

TEST(RawBrainBatch, TheThreadsOfTheHostMatchOneSampleAtATime)
{
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::Sigmoid, brain::ActivationFunction::ReLU, brain::ActivationFunction::None}) {
        for (const brain::UIType threads: {1u, 3u}) {
            brain::RawBrainNetwork together(40, {10.0, 10.0}, utils::system::MemoryMode::Ram);
            brain::RawBrainNetwork one(40, {10.0, 10.0}, utils::system::MemoryMode::Ram);

            together.linksCap(0.1);
            together.size(4, 2);
            together.rand(-1.0, 1.0);
            together.weightInit();
            together.biasInit();
            together.internalActivationFunction(fn);
            together.weightLimits(1e-9, 1e9); // the links stay: only the weights are compared
            together.threads(threads);
            together.batch(true);
            one.deserialize(together.serialize());
            one.internalActivationFunction(fn);
            one.batch(true);
            ASSERT_TRUE(together.batchTraining());
            for (brain::UIType round = 0; round < 3; ++round) {
                const brain::Dataset inputs = batchSamples(11, 4, round);
                brain::Dataset coefs = batchSamples(11, 2, round + 40);
                const brain::Dataset outputs = together.applyTrain(inputs);

                for (brain::Values& sample: coefs) for (brain::FType& coef: sample) coef *= 0.05f;
                ASSERT_EQ(outputs.size(), inputs.size());
                for (std::size_t i = 0; i < inputs.size(); ++i) {
                    one.inputs(inputs[i]);
                    nearValues(outputs[i], one.apply());
                    one.backpropagation(coefs[i]);
                }
                together.backpropagationAll(coefs);
                EXPECT_EQ(together.pending(), inputs.size());
                together.applyBatch();
                one.applyBatch();
            }

            // the same weights after three mini-batches
            const brain::Dataset check = batchSamples(5, 4, 99);
            const brain::Dataset got = together.applyAll(check);
            const brain::Dataset expected = one.applyAll(check);

            for (std::size_t i = 0; i < check.size(); ++i) nearValues(got[i], expected[i]);
        }
    }
}

TEST(RawBrainBatch, ThePendingGradientNeedsTheSamplesOfTheLastApplyTrain)
{
    brain::RawBrainNetwork network(20, {10.0, 10.0}, utils::system::MemoryMode::Ram);

    network.size(3, 2);
    network.rand(-1.0, 1.0);
    network.weightInit();
    network.biasInit();
    network.batch(true);
    (void)network.applyTrain(batchSamples(4, 3, 1));
    EXPECT_ANY_THROW(network.backpropagationAll(batchSamples(3, 2, 1))); // 4 samples were applied
    EXPECT_ANY_THROW(network.backpropagationAll(batchSamples(4, 3, 1))); // 3 coefficients for 2 outputs
}
