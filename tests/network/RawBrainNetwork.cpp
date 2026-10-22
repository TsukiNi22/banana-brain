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
#include <memory>
#include <vector>

static constexpr brain::UIType HEADER = 28; // magic (2) + activation (1) + sizes (16) + random range (8) + random mode (1)

static bool throwsCode(const std::function<void()>& fn, const utils::exception::ExternalCode code)
{
    try {fn();} catch (const utils::exception::IException& e) {return e.getCode() == static_cast<utils::exception::InternalCode>(code);}
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
    offset += 3 * sizeof(brain::FType) + sizeof(std::uint8_t) + sizeof(brain::FType) + sizeof(std::uint8_t) + sizeof(std::uint64_t) + sizeof(brain::FType) + 2 * sizeof(brain::FType) + sizeof(std::uint8_t); // the size of the space, the links limit, the time steps limit, the signal threshold, the limits of the weights and the orphans mode
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

TEST(RawBrainNetwork, InitGivesARandomNumberOfLinksInBothDirections) {
    brain::RawBrainNetwork network(60, {10.0, 10.0});

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
    EXPECT_GT(*std::max_element(counts.begin(), counts.end()), *std::min_element(counts.begin(), counts.end())); // not always the same number
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
    EXPECT_EQ(count, 90u);
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

    EXPECT_EQ(network.steps(), 2u); // min(10 * 25 %, 4 log2(10) = 13)
    network.stepsCap(0.5f);
    EXPECT_EQ(network.steps(), 5u);
    network.stepsCap(1.0f);
    EXPECT_EQ(network.steps(), 10u); // under the hook up to 100 % here
    network.stepsCap(1.0f);
    network.stepsCap([](const brain::UIType) {return 7;}); // the hook is the smallest only under the percentage
    EXPECT_EQ(network.steps(), 7u);
    network.stepsCap([](const brain::UIType) {return 0;});
    EXPECT_EQ(network.steps(), 1u); // at least one step
    network.resetStepsCap();
    EXPECT_EQ(network.steps(), 2u);
    EXPECT_THROW(network.stepsCap(0.0f), utils::exception::IException);
    EXPECT_THROW(network.stepsCap(1.5f), utils::exception::IException);
    EXPECT_THROW(network.stepsCap(brain::StepsCapFn{}), utils::exception::IException);
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

TEST(RawBrainNetwork, AWeakFarthestLinkIsRemovedUnlessItIsTheLastOne) {
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

TEST(RawBrainNetwork, AStrongFarthestLinkLinksTheNextNearestNeuron) {
    brain::RawBrainNetwork network(30, {10.0, 10.0});

    brain::tools::rand(3);
    network.size(2, 2);
    network.weightInit();
    network.weightLimits(0.0, 1e-6); // every farthest link is too strong
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
