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
##  @file Compute.cpp

File Description:
##  Unit tests of the computing on the gpu: the same results as on the host
\**************************************************************/

#define _Exception
#include "brain/network/RawBrainNetwork.hpp"
#include "brain/network/LayerNetwork.hpp"
#include "brain/network/BrainNetwork.hpp"
#include "brain/tools/compute.hpp"
#include <gtest/gtest.h>
#include <cmath>

// The tests that need a CUDA gpu are skipped without one (or when the driver can't be used, ex: under a sanitizer)
#define REQUIRE_CUDA() if (!hasCuda()) GTEST_SKIP() << "no CUDA gpu"

static bool hasCuda(void)
{
    try {return brain::tools::gpu::available(utils::system::make_shared_gpu(utils::system::MemoryMode::Cuda).get());}
    catch (const utils::exception::IException& e) {return false;}
}

static void expectNear(const brain::Values& a, const brain::Values& b)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a[i], b[i], 1e-4 + 1e-5 * std::abs(b[i])) << "value " << i; // float rounding: relative on the big values
}

/* ------------------------------- gpu ------------------------------- */
TEST(Gpu, LayersMatchTheHost) {
    REQUIRE_CUDA();
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::None, brain::ActivationFunction::Sigmoid, brain::ActivationFunction::Linear, brain::ActivationFunction::ReLU}) {
        brain::LayerNetwork gpu(brain::Layers{5, 7, 3}, utils::system::MemoryMode::Cuda);
        brain::LayerNetwork ram(brain::Layers{5, 7, 3}, utils::system::MemoryMode::Ram);
        const brain::Values inputs = {0.3, -0.8, 1.5, 0.1, 2.0};
        gpu.size(5, 3);
        ram.size(5, 3);
        gpu.rand(-1.0, 1.0);
        gpu.weightInit();
        gpu.biasInit();
        gpu.internalActivationFunction(fn);
        ram.deserialize(gpu.serialize());
        ASSERT_TRUE(gpu.onGpu());
        ASSERT_FALSE(ram.onGpu());
        expectNear(gpu.apply(inputs), ram.apply(inputs));

        // the training forward keeps the same sums and outputs
        gpu.inputs(inputs);
        ram.inputs(inputs);
        expectNear(gpu.apply(), ram.apply());
    }
}

// A pair of the same layered network: on the gpu and on the host, from the same weights
struct Twins {
    brain::LayerNetwork gpu;
    brain::LayerNetwork ram;

    Twins(const brain::Layers& layers, const brain::UIType inputs, const brain::ActivationFunction fn): gpu(layers, utils::system::MemoryMode::Cuda), ram(layers, utils::system::MemoryMode::Ram)
    {
        gpu.size(inputs, layers.back());
        ram.size(inputs, layers.back());
        gpu.rand(-0.5, 0.5);
        gpu.weightInit();
        gpu.biasInit();
        gpu.internalActivationFunction(fn);
        ram.deserialize(gpu.serialize());
        ram.internalActivationFunction(fn);
    }
};

static brain::Dataset samples(const brain::UIType count, const brain::UIType size, const brain::UIType seed)
{
    brain::Dataset out(count, brain::Values(size));

    for (brain::UIType i = 0; i < count; ++i)
        for (brain::UIType j = 0; j < size; ++j) out[i][j] = std::sin(static_cast<brain::FType>(seed + i * 31 + j * 7)); // varied, in [-1, 1]
    return out;
}

TEST(Gpu, LayersAllMatchTheHost) {
    REQUIRE_CUDA();
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::None, brain::ActivationFunction::Sigmoid, brain::ActivationFunction::Linear, brain::ActivationFunction::ReLU}) {
        for (const brain::GpuProfile profile: {brain::GpuProfile::Fast, brain::GpuProfile::Light}) {
            Twins twins({70, 33, 3}, 101, fn); // sizes that are not a multiple of the tiles
            const brain::Dataset inputs = samples(131, 101, 3);

            twins.gpu.gpuProfile(profile);
            ASSERT_TRUE(twins.gpu.onGpu());
            const brain::Dataset got = twins.gpu.applyAll(inputs);
            const brain::Dataset expected = twins.ram.applyAll(inputs);

            ASSERT_EQ(got.size(), expected.size());
            for (std::size_t i = 0; i < got.size(); ++i) expectNear(got[i], expected[i]);
        }
    }
}

TEST(Gpu, LayersAllRefuseTheWrongSizes) {
    REQUIRE_CUDA();
    Twins twins({8, 3}, 5, brain::ActivationFunction::ReLU);

    EXPECT_ANY_THROW((void)twins.gpu.applyAll(samples(4, 6, 1)));
    EXPECT_TRUE(twins.gpu.applyAll({}).empty());
}

TEST(Gpu, MiniBatchTrainingMatchesTheHost) {
    REQUIRE_CUDA();
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::None, brain::ActivationFunction::Sigmoid, brain::ActivationFunction::Linear, brain::ActivationFunction::ReLU}) {
        Twins twins({20, 9, 3}, 37, fn);

        twins.gpu.batch(true);
        twins.ram.batch(true);
        ASSERT_TRUE(twins.gpu.batchTraining());
        for (brain::UIType round = 0; round < 3; ++round) {
            const brain::Dataset inputs = samples(70 + round, 37, round);
            const brain::Dataset expected = samples(70 + round, 3, round + 50);
            const brain::Dataset outputs = twins.gpu.applyTrain(inputs);
            brain::Dataset coefs = outputs;

            ASSERT_EQ(outputs.size(), inputs.size());
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                twins.ram.inputs(inputs[i]);
                expectNear(outputs[i], twins.ram.apply());
                for (std::size_t k = 0; k < 3; ++k) coefs[i][k] = 0.05f * (outputs[i][k] - expected[i][k]);
                twins.ram.backpropagation(coefs[i]);
            }
            twins.gpu.backpropagationAll(coefs);
            EXPECT_EQ(twins.gpu.pending(), inputs.size());
            twins.gpu.applyBatch();
            twins.ram.applyBatch();
            EXPECT_EQ(twins.gpu.pending(), 0u);
        }

        // the same weights after three mini-batches (up to the float rounding of the sums)
        twins.gpu.batch(false);
        const brain::Dataset check = samples(9, 37, 77);

        for (std::size_t i = 0; i < check.size(); ++i) expectNear(twins.gpu.apply(check[i]), twins.ram.apply(check[i]));
        EXPECT_GT(twins.gpu.trainingMemory(), 0u);
    }
}

TEST(Gpu, TheLightProfileTrainsOnTheHost) {
    REQUIRE_CUDA();
    Twins twins({12, 3}, 10, brain::ActivationFunction::Sigmoid);

    twins.gpu.batch(true);
    twins.gpu.gpuProfile(brain::GpuProfile::Light);
    EXPECT_FALSE(twins.gpu.batchTraining());
    const brain::Dataset inputs = samples(6, 10, 2);
    const brain::Dataset outputs = twins.gpu.applyTrain(inputs);

    twins.gpu.backpropagationAll(outputs);
    EXPECT_EQ(twins.gpu.pending(), inputs.size());
    twins.gpu.applyBatch();
    EXPECT_EQ(twins.gpu.trainingMemory(), 0u); // nothing kept on the device
}

TEST(Gpu, BatchBackpropagationNeedsTheSamplesOfTheLastApplyTrain) {
    brain::LayerNetwork ram({4, 3}, utils::system::MemoryMode::Ram);

    ram.size(5, 3);
    ram.batch(true);
    EXPECT_ANY_THROW(ram.backpropagationAll(samples(2, 3, 1))); // nothing applied yet
}

TEST(Gpu, BrainApplyAllMatchesTheHost) {
    REQUIRE_CUDA();
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::Sigmoid, brain::ActivationFunction::ReLU, brain::ActivationFunction::None}) {
        for (const brain::GpuProfile profile: {brain::GpuProfile::Fast, brain::GpuProfile::Light}) {
            brain::RawBrainNetwork raw(60, {10.0, 10.0}, utils::system::MemoryMode::Ram);

            raw.linksCap(0.1);
            raw.size(5, 3);
            raw.rand(-1.0, 1.0);
            raw.weightInit();
            raw.biasInit();
            raw.internalActivationFunction(fn);
            brain::BrainNetwork gpu(1, 1, utils::system::MemoryMode::Cuda);
            brain::BrainNetwork ram(1, 1, utils::system::MemoryMode::Ram);
            const brain::Bytes data = raw.reduce()->serialize();

            gpu.deserialize(data);
            ram.deserialize(data);
            gpu.gpuProfile(profile);
            ASSERT_TRUE(gpu.onGpu());
            const brain::Dataset inputs = samples(70, 5, 4);
            const brain::Dataset got = gpu.applyAll(inputs);
            const brain::Dataset expected = ram.applyAll(inputs);

            ASSERT_EQ(got.size(), expected.size());
            for (std::size_t i = 0; i < got.size(); ++i) expectNear(got[i], expected[i]);
        }
    }
}

TEST(Gpu, BrainMiniBatchTrainingMatchesTheHost) {
    REQUIRE_CUDA();
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::Sigmoid, brain::ActivationFunction::ReLU, brain::ActivationFunction::None}) {
        for (const bool direct: {false, true}) {
            brain::RawBrainNetwork gpu(40, {10.0, 10.0}, utils::system::MemoryMode::Cuda);
            brain::RawBrainNetwork ram(40, {10.0, 10.0}, utils::system::MemoryMode::Ram);

            gpu.linksCap(0.1); // without activation a dense network grows past the floats over the steps
            gpu.size(4, 2);
            gpu.rand(-1.0, 1.0);
            gpu.weightInit();
            gpu.biasInit();
            gpu.internalActivationFunction(fn);
            gpu.weightLimits(1e-9, 1e9); // the links stay: only the weights are compared
            gpu.directOutput(direct);
            gpu.batch(true);
            ram.deserialize(gpu.serialize());
            ram.internalActivationFunction(fn);
            ram.directOutput(direct);
            ram.batch(true);
            ASSERT_TRUE(gpu.onGpu());
            ASSERT_TRUE(gpu.batchTraining());
            for (brain::UIType round = 0; round < 3; ++round) {
                const brain::Dataset inputs = samples(9, 4, round);
                brain::Dataset coefs = samples(9, 2, round + 40);
                const brain::Dataset outputs = gpu.applyTrain(inputs);

                for (brain::Values& sample: coefs) for (brain::FType& coef: sample) coef *= 0.05f;
                ASSERT_EQ(outputs.size(), inputs.size());
                for (std::size_t i = 0; i < inputs.size(); ++i) {
                    ram.inputs(inputs[i]);
                    expectNear(outputs[i], ram.apply());
                    ram.backpropagation(coefs[i]);
                }
                gpu.backpropagationAll(coefs);
                EXPECT_EQ(gpu.pending(), inputs.size());
                gpu.applyBatch();
                ram.applyBatch();
            }

            // the same weights after three mini-batches (up to the float rounding of the sums)
            const brain::Dataset check = samples(6, 4, 77);

            for (std::size_t i = 0; i < check.size(); ++i) expectNear(gpu.apply(check[i]), ram.apply(check[i]));
            EXPECT_GT(gpu.trainingMemory(), 0u) << "the states of the batch are kept on the device";
        }
    }
}

TEST(Gpu, BrainStepsMatchTheHost) {
    REQUIRE_CUDA();
    brain::BrainNetwork gpu(20, 4, utils::system::MemoryMode::Cuda);
    brain::BrainNetwork ram(20, 4, utils::system::MemoryMode::Ram);
    const brain::Values inputs = {0.5, -0.25, 1.0};

    // a reduced network is only executed: its data comes from a raw network
    brain::RawBrainNetwork raw(20, {10.0, 10.0}, utils::system::MemoryMode::Ram);

    raw.size(3, 2);
    raw.rand(-1.0, 1.0);
    raw.weightInit();
    raw.biasInit();
    raw.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    raw.threshold(0.05);
    gpu.deserialize(raw.reduce()->serialize());
    ram.deserialize(raw.reduce()->serialize());
    ASSERT_TRUE(gpu.onGpu());
    expectNear(gpu.apply(inputs), ram.apply(inputs));
}

TEST(Gpu, RawStepsMatchTheHostAndKeepEveryState) {
    REQUIRE_CUDA();
    brain::RawBrainNetwork gpu(30, {10.0, 10.0}, utils::system::MemoryMode::Cuda);
    brain::RawBrainNetwork ram(30, {10.0, 10.0}, utils::system::MemoryMode::Ram);
    const brain::Values inputs = {0.5, -0.25, 1.0};
    gpu.size(3, 2);
    gpu.rand(-1.0, 1.0);
    gpu.weightInit();
    gpu.biasInit();
    gpu.internalActivationFunction(brain::ActivationFunction::ReLU);
    ram.deserialize(gpu.serialize());
    gpu.inputs(inputs);
    ram.inputs(inputs);
    expectNear(gpu.apply(), ram.apply());
    expectNear(gpu.reduce()->apply(inputs), ram.reduce()->apply(inputs));
}

TEST(Gpu, AnExternalActivationFallsBackToTheHost) {
    REQUIRE_CUDA();
    brain::LayerNetwork gpu(brain::Layers{3, 2}, utils::system::MemoryMode::Cuda);
    brain::LayerNetwork ram(brain::Layers{3, 2}, utils::system::MemoryMode::Ram);
    const brain::Values inputs = {1.0, 2.0};
    gpu.size(2, 2);
    ram.size(2, 2);
    gpu.weightInit();
    ram.deserialize(gpu.serialize());
    gpu.activationFunction([](const brain::FType value) {return value * 2;});
    ram.activationFunction([](const brain::FType value) {return value * 2;});
    expectNear(gpu.apply(inputs), ram.apply(inputs));
}

TEST(Gpu, DirectCallRunsOnTheDevice) {
    REQUIRE_CUDA();
    brain::BrainNetwork network(4, 2, utils::system::MemoryMode::Cuda);
    std::vector<brain::Values> states;
    network.size(1, 1);
    EXPECT_TRUE(brain::tools::gpu::steps(network.gpu().get(), 0, 0, false, 4, 2, 1, 1, brain::ActivationFunction::None, 0.0, {1.0}, states, false) == false); // no memory given
    EXPECT_TRUE(states.empty());
    EXPECT_FALSE(brain::tools::gpu::available(nullptr));
}

TEST(Gpu, LayersBackpropagationMatchesTheHost) {
    REQUIRE_CUDA();
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::None, brain::ActivationFunction::Sigmoid, brain::ActivationFunction::ReLU}) {
        brain::LayerNetwork gpu(brain::Layers{5, 7, 3}, utils::system::MemoryMode::Cuda);
        brain::LayerNetwork ram(brain::Layers{5, 7, 3}, utils::system::MemoryMode::Ram);
        const brain::Values inputs = {0.3, -0.8, 1.5, 0.1, 2.0};
        const brain::Values errors = {0.2, -0.1, 0.05};

        gpu.size(5, 3);
        ram.size(5, 3);
        gpu.rand(-1.0, 1.0);
        gpu.weightInit();
        gpu.biasInit();
        gpu.internalActivationFunction(fn);
        ram.deserialize(gpu.serialize());
        for (brain::UIType round = 0; round < 3; ++round) {
            gpu.inputs(inputs);
            ram.inputs(inputs);
            (void)gpu.apply();
            (void)ram.apply();
            gpu.backpropagation(errors);
            ram.backpropagation(errors);
        }
        ASSERT_TRUE(gpu.onGpu());
        expectNear(gpu.apply(inputs), ram.apply(inputs));
        const brain::Bytes a = gpu.serialize();
        const brain::Bytes b = ram.serialize();
        ASSERT_EQ(a.size(), b.size());
        EXPECT_GT(a.size(), 0u);
    }
}

TEST(Gpu, RawBackpropagationMatchesTheHost) {
    REQUIRE_CUDA();
    for (const brain::ActivationFunction fn: {brain::ActivationFunction::None, brain::ActivationFunction::Sigmoid, brain::ActivationFunction::ReLU}) {
        brain::RawBrainNetwork gpu(30, {10.0, 10.0}, utils::system::MemoryMode::Cuda);
        brain::RawBrainNetwork ram(30, {10.0, 10.0}, utils::system::MemoryMode::Ram);
        const brain::Values inputs = {0.5, -0.25, 1.0};

        gpu.linksCap(0.1); // 3 links: without activation a dense network grows past the floats over the steps
        gpu.size(3, 2);
        gpu.rand(-1.0, 1.0);
        gpu.weightInit();
        gpu.biasInit();
        gpu.internalActivationFunction(fn);
        gpu.weightLimits(1e-9, 1e9); // the links stay: only the weights are compared
        ram.deserialize(gpu.serialize());
        for (brain::UIType round = 0; round < 3; ++round) {
            gpu.inputs(inputs);
            ram.inputs(inputs);
            (void)gpu.apply();
            (void)ram.apply();
            gpu.backpropagation(brain::Values{0.05, -0.02});
            ram.backpropagation(brain::Values{0.05, -0.02});
        }
        ASSERT_TRUE(gpu.onGpu());
        gpu.inputs(inputs);
        ram.inputs(inputs);
        expectNear(gpu.apply(), ram.apply());
        expectNear(gpu.reduce()->apply(inputs), ram.reduce()->apply(inputs)); // the execution copy follows
    }
}

TEST(Gpu, RawTrainingOptionsMatchTheHost) {
    REQUIRE_CUDA();
    brain::RawBrainNetwork gpu(30, {10.0, 10.0}, utils::system::MemoryMode::Cuda);
    brain::RawBrainNetwork ram(30, {10.0, 10.0}, utils::system::MemoryMode::Ram);
    const brain::Values inputs = {0.5, -0.25, 1.0};

    gpu.linksCap(0.2);
    gpu.size(3, 2);
    gpu.rand(-1.0, 1.0);
    gpu.weightInit();
    gpu.biasInit();
    gpu.weightLimits(1e-9, 1.5); // the clamp keeps the weights under 1.5, the links stay
    gpu.momentum(0.5);
    gpu.adam(0.05);
    gpu.stepLimit(0.2);
    gpu.clampWeights(true);
    gpu.batch(true);
    ram.deserialize(gpu.serialize());
    for (brain::UIType round = 0; round < 3; ++round) {
        for (brain::RawBrainNetwork* network: {&gpu, &ram}) {
            network->inputs(inputs);
            (void)network->apply();
            network->backpropagation(brain::Values{0.5, -0.2});
            network->backpropagation(brain::Values{0.1, 0.3});
            network->applyBatch();
        }
    }
    ASSERT_TRUE(gpu.onGpu());
    gpu.inputs(inputs);
    ram.inputs(inputs);
    expectNear(gpu.apply(), ram.apply());
    expectNear(gpu.reduce()->apply(inputs), ram.reduce()->apply(inputs)); // the execution copy follows
}
