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
#include <vector>
#include <cmath>

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
    brain::tools::put<std::uint64_t>(data, 4);
    brain::tools::put<brain::FType>(data, 0.0f); // signal threshold
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

// The file of a reduced network: its neurons, the biases and the slots of links (id, weight) of each neuron
static brain::Bytes brainBytes(const brain::UIType neurons, const brain::UIType cap, const brain::UIType steps, const brain::UIType input, const brain::UIType output,
    const std::vector<brain::FType>& biases, const std::vector<std::pair<brain::UIType, brain::FType>>& links, const brain::FType threshold = 0.0)
{
    brain::BrainNetwork base(neurons, cap);
    brain::Bytes data;

    base.size(input, output);
    data = base.serialize();
    data.resize(HEADER);
    brain::tools::put<std::uint64_t>(data, neurons);
    brain::tools::put<std::uint64_t>(data, cap);
    brain::tools::put<std::uint64_t>(data, steps);
    brain::tools::put<brain::FType>(data, threshold);
    brain::tools::put<std::uint64_t>(data, biases.size());
    for (const brain::FType bias: biases)
        brain::tools::put<brain::FType>(data, bias);
    brain::tools::put<std::uint64_t>(data, links.size());
    for (const std::pair<brain::UIType, brain::FType>& link: links) {
        brain::tools::put<std::uint64_t>(data, link.first);
        brain::tools::put<brain::FType>(data, link.second);
    }
    return data;
}

// A chain: the input 0 feeds 1 (x2), 1 feeds 2 (x3), 2 feeds the output 3 (x0.5), one slot of link each
static brain::Bytes chain(const brain::UIType steps, const brain::FType threshold = 0.0)
{
    return brainBytes(4, 1, steps, 1, 1, {0.0, 0.0, 0.0, 0.0}, {{0, 0.0}, {0, 2.0}, {1, 3.0}, {2, 0.5}}, threshold);
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
    network->size(2, 1);
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
    EXPECT_EQ(brain::tools::get<std::uint64_t>(data, offset), 1u); // time steps (placeholder default of a reduced network built alone)
    EXPECT_EQ(brain::tools::get<brain::FType>(data, offset), static_cast<brain::FType>(BRAIN_SIGNAL_THRESHOLD)); // signal threshold
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

/* ------------------------------- execution ------------------------------- */
TEST(BrainNetwork, EachTimeStepReadsThePreviousOne) {
    brain::BrainNetwork network(1);
    const brain::Values inputs = {1.0};

    // the signal needs 3 steps to cross the chain: 1 * 2 * 3 * 0.5
    network.deserialize(chain(1));
    EXPECT_EQ(network.apply(inputs), (brain::Values{0.0}));
    network.deserialize(chain(2));
    EXPECT_EQ(network.apply(inputs), (brain::Values{0.0}));
    network.deserialize(chain(3));
    EXPECT_EQ(network.apply(inputs), (brain::Values{3.0}));
    network.deserialize(chain(10)); // stable once it crossed
    EXPECT_EQ(network.apply(inputs), (brain::Values{3.0}));
    EXPECT_EQ(network.steps(), 10u);
}

TEST(BrainNetwork, TheInputStaysAtItsValueAndTheBiasIsAdded) {
    brain::BrainNetwork network(1);

    network.deserialize(brainBytes(3, 1, 2, 1, 1, {0.0, 1.0, 0.0}, {{0, 0.0}, {0, 2.0}, {1, 1.0}})); // 1 = 1 + 2 * in, output = 1
    EXPECT_EQ(network.apply(brain::Values{2.0}), (brain::Values{5.0}));
    network.internalActivationFunction(brain::ActivationFunction::ReLU);
    EXPECT_EQ(network.apply(brain::Values{-2.0}), (brain::Values{0.0})); // 1 - 4 under 0
}

TEST(BrainNetwork, ExecutionNeedsTheInputsAndTheOutputs) {
    brain::BrainNetwork network(4, 1);

    EXPECT_TRUE(throwsCode([&] {(void)network.apply(brain::Values{1.0});}, utils::exception::ExternalCode::InvalidInputSize));
    network.input(1);
    EXPECT_TRUE(throwsCode([&] {(void)network.apply(brain::Values{1.0});}, utils::exception::ExternalCode::InvalidOutputSize));
    network.output(1);
    EXPECT_NO_THROW((void)network.apply(brain::Values{1.0}));
}

TEST(BrainNetwork, InputsAndOutputsHaveToFitInTheNeurons) {
    brain::BrainNetwork network(4);

    EXPECT_THROW(network.size(3, 3), utils::exception::IException);
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{0, 0})); // the sizes are restored
    EXPECT_NO_THROW(network.size(2, 2));
    EXPECT_THROW(network.output(3), utils::exception::IException);
    EXPECT_EQ(network.output(), 2u);
}

TEST(BrainNetwork, StepsAreSerialized) {
    brain::BrainNetwork network(1);
    brain::BrainNetwork other(1);

    network.deserialize(chain(7));
    other.deserialize(network.serialize());
    EXPECT_EQ(other.steps(), 7u);
    EXPECT_EQ(other.serialize(), network.serialize());
    EXPECT_THROW(other.deserialize(chain(0)), utils::exception::IException); // at least one step
}

TEST(BrainNetwork, AWeakSignalIsCancelled) {
    brain::BrainNetwork network(1);
    const brain::Values inputs = {1.0};

    network.deserialize(chain(3, 2.5)); // the neuron 1 gets 2 (under 2.5): cancelled, nothing crosses the chain
    EXPECT_EQ(network.apply(inputs), (brain::Values{0.0}));
    network.deserialize(chain(3, 1.0)); // 2, 6 and 3 are over 1
    EXPECT_EQ(network.apply(inputs), (brain::Values{3.0}));
    network.deserialize(chain(3, 2.0)); // the absolute value is compared with "under": 2 is kept
    EXPECT_EQ(network.apply(inputs), (brain::Values{3.0}));
    EXPECT_EQ(network.apply(brain::Values{-1.0}), (brain::Values{-3.0})); // a negative signal too
}

TEST(BrainNetwork, ThresholdSetterAndGetter) {
    brain::BrainNetwork network(1);

    EXPECT_FLOAT_EQ(network.threshold(), static_cast<brain::FType>(BRAIN_SIGNAL_THRESHOLD));
    network.deserialize(chain(3));
    network.threshold(2.5);
    EXPECT_FLOAT_EQ(network.threshold(), 2.5f);
    EXPECT_EQ(network.apply(brain::Values{1.0}), (brain::Values{0.0}));
    network.threshold(0.0);
    EXPECT_EQ(network.apply(brain::Values{1.0}), (brain::Values{3.0}));
    EXPECT_THROW(network.threshold(-0.1), utils::exception::IException);
    EXPECT_THROW(network.threshold(std::nanf("")), utils::exception::IException);
    EXPECT_FLOAT_EQ(network.threshold(), 0.0f);
}

TEST(BrainNetwork, ThresholdIsSerialized) {
    brain::BrainNetwork network(1);
    brain::BrainNetwork other(1);

    network.deserialize(chain(3, 0.75));
    other.deserialize(network.serialize());
    EXPECT_FLOAT_EQ(other.threshold(), 0.75f);
    EXPECT_EQ(other.serialize(), network.serialize());
    EXPECT_TRUE(throwsCode([&] {other.deserialize(chain(3, -1.0));}, utils::exception::ExternalCode::InvalidNetworkFile));
}
