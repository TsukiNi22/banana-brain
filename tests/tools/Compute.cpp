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
