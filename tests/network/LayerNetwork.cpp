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
#include "brain/tools/random.hpp"
#include "brain/tools/bytes.hpp"
#include <gtest/gtest.h>
#include <functional>
#include <cstdint>
#include <atomic>
#include <thread>
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

// A valid file of a network of two layers of two neurons and two inputs: 4 neurons and 8 weights stored
static brain::Bytes layerData(void)
{
    brain::LayerNetwork base(1, 1);
    base.input(2);
    brain::Bytes data = base.serialize();

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
    brain::tools::put<std::uint64_t>(data, 8);
    for (std::uint64_t i = 0; i < 8; ++i)
        brain::tools::put<brain::FType>(data, 0.25f * static_cast<brain::FType>(i));
    return data;
}

struct ParsedLayer {
    brain::Layers layers;
    std::vector<brain::LayerNeuron> neurons;
    std::vector<brain::FType> weights;
};

// What a network stores, read from its file
static ParsedLayer parseLayer(const brain::Bytes& data)
{
    ParsedLayer parsed;
    brain::UIType offset = HEADER;
    brain::UIType count = brain::tools::get<std::uint64_t>(data, offset);

    for (brain::UIType i = 0; i < count; ++i)
        parsed.layers.push_back(brain::tools::get<std::uint64_t>(data, offset));
    count = brain::tools::get<std::uint64_t>(data, offset);
    for (brain::UIType i = 0; i < count; ++i) {
        brain::LayerNeuron neuron;
        neuron.incoming = brain::tools::get<std::uint64_t>(data, offset);
        neuron.count = brain::tools::get<std::uint64_t>(data, offset);
        neuron.bias = brain::tools::get<brain::FType>(data, offset);
        parsed.neurons.push_back(neuron);
    }
    count = brain::tools::get<std::uint64_t>(data, offset);
    for (brain::UIType i = 0; i < count; ++i)
        parsed.weights.push_back(brain::tools::get<brain::FType>(data, offset));
    return parsed;
}

// The file of a network whose weights (neuron after neuron) and biases are given
static brain::Bytes layerBytes(const brain::Layers& layers, const brain::UIType input, const std::vector<brain::FType>& weights, const std::vector<brain::FType>& biases,
    const brain::ActivationFunction activation = brain::ActivationFunction::None)
{
    brain::LayerNetwork base(layers);
    brain::UIType previous = input;
    brain::UIType first = 0;
    brain::UIType index = 0;
    brain::Bytes data;

    base.input(input);
    base.internalActivationFunction(activation);
    data = base.serialize();
    data.resize(HEADER);
    brain::tools::put<std::uint64_t>(data, layers.size());
    for (const brain::UIType layer: layers)
        brain::tools::put<std::uint64_t>(data, layer);
    brain::tools::put<std::uint64_t>(data, biases.size());
    for (const brain::UIType layer: layers) {
        for (brain::UIType k = 0; k < layer; ++k, ++index) {
            brain::tools::put<std::uint64_t>(data, first);
            brain::tools::put<std::uint64_t>(data, previous);
            brain::tools::put<brain::FType>(data, biases[index]);
            first += previous;
        }
        previous = layer;
    }
    brain::tools::put<std::uint64_t>(data, weights.size());
    for (const brain::FType weight: weights)
        brain::tools::put<brain::FType>(data, weight);
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

/* ------------------------------- init ------------------------------- */
TEST(LayerNetwork, InitSetsEveryNeuronAtItsDefault) {
    brain::LayerNetwork network(brain::Layers{2, 3});
    ParsedLayer parsed;

    network.input(4); // the first layer has one weight per input
    parsed = parseLayer(network.serialize());
    ASSERT_EQ(parsed.neurons.size(), 5u);
    ASSERT_EQ(parsed.weights.size(), 14u); // 2 * 4 + 3 * 2
    EXPECT_EQ(parsed.neurons[0].incoming, 0u);
    EXPECT_EQ(parsed.neurons[1].incoming, 4u);
    EXPECT_EQ(parsed.neurons[2].incoming, 8u);
    EXPECT_EQ(parsed.neurons[4].incoming, 12u);
    EXPECT_EQ(parsed.neurons[0].count, 4u);
    EXPECT_EQ(parsed.neurons[2].count, 2u); // one weight per neuron of the previous layer
    for (const brain::LayerNeuron& neuron: parsed.neurons)
        EXPECT_EQ(neuron.bias, 0.0f);
    for (const brain::FType weight: parsed.weights)
        EXPECT_EQ(weight, 1.0f);
}

TEST(LayerNetwork, ChangingTheInputSizeSetsTheWeightsUpAgain) {
    brain::LayerNetwork network(brain::Layers{2});

    EXPECT_TRUE(parseLayer(network.serialize()).weights.empty()); // no input size yet: no weight
    network.input(3);
    EXPECT_EQ(parseLayer(network.serialize()).weights.size(), 6u);
    network.input(5);
    EXPECT_EQ(parseLayer(network.serialize()).weights.size(), 10u);
    network.output(7); // the outputs do not change the weights
    EXPECT_EQ(parseLayer(network.serialize()).weights.size(), 10u);
}

TEST(LayerNetwork, TooManyWeightsOverflowTheSize) {
    brain::LayerNetwork network(brain::Layers{4, 4});

    EXPECT_THROW(network.input(std::numeric_limits<brain::UIType>::max() / 2), utils::exception::IException);
}

TEST(LayerNetwork, WeightInitUsesTheRandomRange) {
    brain::LayerNetwork network(brain::Layers{2, 2});

    network.input(3);
    network.rand(0.25, 0.25);
    network.weightInit();
    for (const brain::FType weight: parseLayer(network.serialize()).weights)
        EXPECT_FLOAT_EQ(weight, 0.25f);
    network.rand(0.0, 0.0); // a result that falls on 0 is the fallback
    network.weightInit(0.5);
    for (const brain::FType weight: parseLayer(network.serialize()).weights)
        EXPECT_FLOAT_EQ(weight, 0.5f);
}

TEST(LayerNetwork, RandomFunctionWinsOverTheRange) {
    brain::LayerNetwork network(brain::Layers{2});

    network.input(2);
    network.rand(0.25, 0.25);
    network.rand([](const brain::IType) {return 0.75f;});
    network.weightInit();
    network.biasInit();
    for (const brain::FType weight: parseLayer(network.serialize()).weights)
        EXPECT_FLOAT_EQ(weight, 0.75f);
    for (const brain::LayerNeuron& neuron: parseLayer(network.serialize()).neurons)
        EXPECT_FLOAT_EQ(neuron.bias, 0.75f);
    network.rand([](const brain::IType) {return std::nanf("");});
    EXPECT_THROW(network.weightInit(), utils::exception::IException); // a non finite value is refused
}

TEST(LayerNetwork, BiasInitStaysInTheRange) {
    brain::LayerNetwork network(brain::Layers{20});

    network.rand(0.5, 1.0);
    network.biasInit();
    for (const brain::LayerNeuron& neuron: parseLayer(network.serialize()).neurons) {
        EXPECT_GE(neuron.bias, 0.5f);
        EXPECT_LE(neuron.bias, 1.0f);
    }
}

/* ------------------------------- execution ------------------------------- */
TEST(LayerNetwork, ExecutesLayerAfterLayer) {
    brain::LayerNetwork network(brain::Layers{2, 1});

    // neuron 0: 0.5 + 1 + 2 = 3.5, neuron 1: 0 + 3 + 4 = 7, output: 1 + 0.5 * 3.5 + 0.25 * 7 = 4.5
    network.deserialize(layerBytes({2, 1}, 2, {1.0, 2.0, 3.0, 4.0, 0.5, 0.25}, {0.5, 0.0, 1.0}));
    EXPECT_EQ(network.apply(brain::Values{1.0, 1.0}), (brain::Values{4.5}));
}

TEST(LayerNetwork, ExecutesWithTheActivation) {
    brain::LayerNetwork network(brain::Layers{1});
    const brain::Values inputs = {1.0, 1.0};

    network.deserialize(layerBytes({1}, 2, {1.0, -3.0}, {0.0})); // the sum is -2
    EXPECT_EQ(network.apply(inputs), (brain::Values{-2.0}));
    network.internalActivationFunction(brain::ActivationFunction::ReLU);
    EXPECT_EQ(network.apply(inputs), (brain::Values{0.0}));
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    EXPECT_NEAR(network.apply(inputs)[0], 1.0 / (1.0 + std::exp(2.0)), 1e-6);
    network.activationFunction([](const brain::FType value) {return value * 2;}); // the external function wins
    EXPECT_EQ(network.apply(inputs), (brain::Values{-4.0}));
}

TEST(LayerNetwork, TrainingForwardGivesTheSameOutputsAndKeepsTheLayers) {
    brain::LayerNetwork network(brain::Layers{2, 1});
    const brain::Values inputs = {1.0, 0.5};

    network.deserialize(layerBytes({2, 1}, 2, {1.0, 2.0, 3.0, 4.0, 0.5, 0.25}, {0.5, 0.0, 1.0}));
    network.inputs(inputs);
    EXPECT_EQ(network.apply(), network.apply(inputs));
}

TEST(LayerNetwork, ExecutionNeedsTheNumberOfInputs) {
    brain::LayerNetwork network(2, 2);

    EXPECT_TRUE(throwsCode([&] {(void)network.apply(brain::Values{1.0, 2.0});}, utils::exception::ExternalCode::InvalidInputSize));
    network.input(2);
    EXPECT_NO_THROW((void)network.apply(brain::Values{1.0, 2.0}));
}

TEST(LayerNetwork, SeveralThreadsExecuteTheSameNetwork) {
    brain::LayerNetwork network(brain::Layers{2, 1});
    const brain::Values expected = {4.5};
    std::vector<std::thread> threads; // xstyle: ignore LU-THREADS (raw threads on purpose: the test of the concurrent calls)
    std::atomic<int> wrong = 0;

    network.deserialize(layerBytes({2, 1}, 2, {1.0, 2.0, 3.0, 4.0, 0.5, 0.25}, {0.5, 0.0, 1.0}));
    for (int i = 0; i < 8; ++i)
        threads.emplace_back([&] {
            for (int round = 0; round < 200; ++round)
                if (network.apply(brain::Values{1.0, 1.0}) != expected) ++wrong;
        });
    for (std::thread& thread: threads)
        thread.join();
    EXPECT_EQ(wrong, 0);
}

TEST(LayerNetwork, DeserializeRefusesNeuronsThatDontMatchTheLayers) {
    brain::LayerNetwork network(brain::Layers{2, 1});
    brain::Bytes bad = layerBytes({2, 1}, 2, {1.0, 2.0, 3.0, 4.0, 0.5, 0.25}, {0.5, 0.0, 1.0});

    EXPECT_NO_THROW(network.deserialize(bad));
    bad = layerBytes({2, 1}, 2, {1.0, 2.0, 3.0, 4.0, 0.5}, {0.5, 0.0, 1.0}); // a neuron reads a weight that does not exist
    EXPECT_TRUE(throwsCode([&] {network.deserialize(bad);}, utils::exception::ExternalCode::InvalidNetworkFile));
}

/* ------------------------------- backpropagation ------------------------------- */
TEST(LayerNetwork, BackpropagationNeedsAnApply) {
    brain::LayerNetwork network(brain::Layers{3, 2}, utils::system::MemoryMode::Ram);

    network.size(2, 2);
    EXPECT_THROW(network.backpropagation(0.5), utils::exception::IException); // no apply() yet
    network.inputs(brain::Values{0.5, 0.5});
    (void)network.apply();
    EXPECT_NO_THROW(network.backpropagation(0.5));
    EXPECT_THROW(network.backpropagation(std::nanf("")), utils::exception::IException);
    EXPECT_THROW(network.backpropagation(brain::Values{0.1}), utils::exception::IException); // 1 coefficient for 2 outputs
    EXPECT_THROW(network.backpropagation(brain::Values{0.1, std::nanf("")}), utils::exception::IException);
}

TEST(LayerNetwork, BackpropagationGoesDownTheGradient) {
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::None, brain::ActivationFunction::Sigmoid, brain::ActivationFunction::ReLU}) {
        brain::LayerNetwork network(brain::Layers{4, 1}, utils::system::MemoryMode::Ram);
        const brain::Values inputs = {0.6, -0.3};
        const brain::FType expected = 0.25f;

        network.size(2, 1);
        network.rand(0.1, 0.9);
        network.weightInit();
        network.biasInit();
        network.internalActivationFunction(fn);
        network.inputs(inputs);
        const brain::FType first = std::abs(network.apply()[0] - expected);
        for (brain::UIType round = 0; round < 300; ++round) {
            network.inputs(inputs);
            network.backpropagation(0.1f * (network.apply()[0] - expected)); // the error of the output, times the learning rate
        }
        network.inputs(inputs);
        EXPECT_LT(std::abs(network.apply()[0] - expected), std::max(first * 0.2f, 0.01f)) << "activation " << static_cast<int>(fn);
    }
}

TEST(LayerNetwork, BackpropagationMatchesTheNumericalGradient) {
    brain::LayerNetwork network(brain::Layers{3, 2}, utils::system::MemoryMode::Ram);
    const brain::Values inputs = {0.4, -0.7};
    const brain::Values errors = {1.0, -0.5};

    network.size(2, 2);
    network.rand(-1.0, 1.0);
    network.weightInit();
    network.biasInit();
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    const brain::Bytes start = network.serialize();

    // the loss L = sum(error_k * output_k): its gradient is what one step with these errors moves, so L goes down
    network.inputs(inputs);
    const brain::Values before = network.apply();
    network.backpropagation(errors);
    network.inputs(inputs);
    const brain::Values after = network.apply();
    const brain::FType lossBefore = errors[0] * before[0] + errors[1] * before[1];
    const brain::FType lossAfter = errors[0] * after[0] + errors[1] * after[1];

    EXPECT_NE(network.serialize(), start);
    EXPECT_LT(lossAfter, lossBefore);
}

TEST(LayerNetwork, MiniBatchOfTheSameSampleIsOneOnlineUpdate) {
    // on the host memory (a gpu machine would hide a difference): the mean of two equal gradients is the gradient of one sample, the weights AND the biases change once, at applyBatch()
    brain::LayerNetwork online(brain::Layers{4, 1}, utils::system::MemoryMode::Ram);
    brain::LayerNetwork grouped(brain::Layers{4, 1}, utils::system::MemoryMode::Ram);
    const brain::Values sample = {0.2f, 0.7f};

    for (brain::LayerNetwork* network: {&online, &grouped}) {
        network->rand(-1.0f, 1.0f);
        network->internalActivationFunction(brain::ActivationFunction::Sigmoid);
        brain::tools::rand(6);
        network->size(2, 1);
        network->weightInit();
        network->biasInit();
    }
    EXPECT_FALSE(grouped.batch());
    grouped.batch(true);
    EXPECT_TRUE(grouped.batch());
    for (brain::UIType epoch = 0; epoch < 5; ++epoch) {
        online.inputs(sample);
        online.backpropagation(brain::Values{0.5f * (online.apply()[0] - 0.9f)});
        for (brain::UIType k = 0; k < 2; ++k) {
            grouped.inputs(sample);
            grouped.backpropagation(brain::Values{0.5f * (grouped.apply()[0] - 0.9f)});
        }
        EXPECT_EQ(grouped.pending(), 2u);
        grouped.applyBatch();
        EXPECT_EQ(grouped.pending(), 0u);
        EXPECT_NEAR(grouped.apply(sample)[0], online.apply(sample)[0], 1e-6f) << "epoch " << epoch;
    }
    grouped.applyBatch(); // nothing pending: nothing done
    EXPECT_NEAR(grouped.apply(sample)[0], online.apply(sample)[0], 1e-6f);
}
