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
##  @file NeuralNetwork.cpp

File Description:
##  Unit tests of the NeuralNetwork (configuration, training & usage)
\**************************************************************/

#define _Exception
#include "brain/NeuralNetwork.hpp"
#include "brain/tools/random.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <functional>
#include <fstream>
#include <utility>
#include <cstdint>
#include <cmath>

static bool throwsCode(const std::function<void()>& fn, const utils::exception::ExternalCode code)
{
    try {fn();} catch (const utils::exception::IException& e) {return e.getCode() == static_cast<utils::exception::InternalCode>(code);}
    return false;
}

static bool throwsInternal(const std::function<void()>& fn, const utils::exception::InternalCode code)
{
    try {fn();} catch (const utils::exception::IException& e) {return e.getCode() == code;}
    return false;
}

static std::filesystem::path tmpFile(const std::string& name)
{
    return std::filesystem::temp_directory_path() / ("brain_test_" + name);
}

/* ------------------------------- NeuralNetwork ------------------------------- */
TEST(NeuralNetwork, DefaultState) {
    brain::NeuralNetwork network;
    EXPECT_FALSE(network.isFrozen());
    EXPECT_EQ(network.learningRate(), 1.0);
    EXPECT_EQ(network.learningRateDecayed(), 1.0);
    EXPECT_EQ(network.learningSteps(), 0u);
    EXPECT_EQ(network.lossFunction().first, brain::LossFunction::None);
    EXPECT_EQ(network.lossFunction().second, nullptr);
}

TEST(NeuralNetwork, NoNetworkBeforeInit) {
    brain::NeuralNetwork network;
    const utils::exception::ExternalCode noNetwork = utils::exception::ExternalCode::NoNetwork;

    EXPECT_TRUE(throwsCode([&] {(void)network.exec({1.0});}, noNetwork));
    EXPECT_TRUE(throwsCode([&] {network.size(2, 1);}, noNetwork)); // the sizes, the random and the activation belong to the network
    EXPECT_TRUE(throwsCode([&] {(void)network.input();}, noNetwork));
    EXPECT_TRUE(throwsCode([&] {network.rand(0.0, 1.0);}, noNetwork));
    EXPECT_TRUE(throwsCode([&] {network.internalActivationFunction(brain::ActivationFunction::ReLU);}, noNetwork));
    EXPECT_TRUE(throwsCode([&] {(void)network.neurons();}, noNetwork));
    EXPECT_TRUE(throwsCode([&] {network.save(tmpFile("none.net"));}, noNetwork));
}

TEST(NeuralNetwork, InitLayersFromAListOrFromACount) {
    brain::NeuralNetwork network;

    network.init<brain::NetworkType::Layered>(brain::Layers{3, 2, 1});
    EXPECT_EQ(network.neurons(), (std::pair<brain::UIType, brain::UIType>{3, 6}));
    network.init<brain::NetworkType::Layered>(2, 4);
    EXPECT_EQ(network.neurons(), (std::pair<brain::UIType, brain::UIType>{2, 8}));
    network.init<brain::NetworkType::Brain>(5);
    EXPECT_EQ(network.neurons(), (std::pair<brain::UIType, brain::UIType>{0, 5}));
}

TEST(NeuralNetwork, InitKeepsTheNetworkOnBadParameters) {
    brain::NeuralNetwork network;

    network.init<brain::NetworkType::Layered>(2, 4);
    EXPECT_THROW(network.init<brain::NetworkType::Layered>(brain::Layers{3, 0}), utils::exception::IException);
    EXPECT_THROW(network.init<brain::NetworkType::Layered>(0, 4), utils::exception::IException);
    EXPECT_THROW(network.init<brain::NetworkType::Brain>(0), utils::exception::IException);
    EXPECT_EQ(network.neurons(), (std::pair<brain::UIType, brain::UIType>{2, 8}));
}

TEST(NeuralNetwork, SettersAreAppliedToTheNetwork) {
    brain::NeuralNetwork network;

    network.init<brain::NetworkType::Layered>(2, 3);
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{0, 0}));
    EXPECT_EQ(network.rand().first, (std::pair<brain::FType, brain::FType>{-1.0, 1.0}));
    EXPECT_EQ(network.activationFunction().first, brain::ActivationFunction::None);
    network.size(3, 2);
    network.rand(0.0, 0.5);
    network.internalActivationFunction(brain::ActivationFunction::ReLU);
    EXPECT_EQ(network.input(), 3u);
    EXPECT_EQ(network.output(), 2u);
    EXPECT_EQ(network.rand().first, (std::pair<brain::FType, brain::FType>{0.0, 0.5}));
    EXPECT_EQ(network.activationFunction().first, brain::ActivationFunction::ReLU);
    network.input(5);
    network.output(1);
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{5, 1}));
    network.init<brain::NetworkType::Layered>(2, 3); // a new network: the settings are the ones of the new one
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{0, 0}));
}

TEST(NeuralNetwork, ExternalFunctionsResetTheInternalOne) {
    brain::NeuralNetwork network;

    network.init<brain::NetworkType::Layered>(2, 3);
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    network.activationFunction([](const brain::FType value) {return value;});
    EXPECT_EQ(network.activationFunction().first, brain::ActivationFunction::None);
    EXPECT_NE(network.activationFunction().second, nullptr);
    network.internalLossFunction(brain::LossFunction::CrossEntropy);
    network.lossFunction([](const brain::FType output, const brain::FType expected) {return output - expected;});
    EXPECT_EQ(network.lossFunction().first, brain::LossFunction::None);
    EXPECT_NE(network.lossFunction().second, nullptr);
    network.rand([](const brain::IType seed) {return static_cast<brain::FType>(seed);});
    EXPECT_NE(network.rand().second, nullptr);
}

TEST(NeuralNetwork, ExecChecksTheInputs) {
    brain::NeuralNetwork network;

    network.init<brain::NetworkType::Layered>(2, 3);
    network.size(2, 1);
    EXPECT_TRUE(throwsCode([&] {(void)network.exec({1.0, 2.0, 3.0});}, utils::exception::ExternalCode::InvalidInputSize));
    EXPECT_TRUE(throwsCode([&] {(void)network.exec({});}, utils::exception::ExternalCode::InvalidInputSize));
    EXPECT_NO_THROW((void)network.exec({1.0, 2.0}));
}

TEST(NeuralNetwork, ExecRunsTheNetworks) {
    brain::NeuralNetwork layered;
    brain::NeuralNetwork brain;

    layered.init<brain::NetworkType::Layered>(brain::Layers{2, 1});
    layered.size(2, 1);
    EXPECT_EQ(layered.exec({1.0, 1.0}), (brain::Values{4.0})); // weights 1.0, bias 0.0: 2, 2 then 4
    layered.internalActivationFunction(brain::ActivationFunction::ReLU);
    EXPECT_EQ(layered.exec({-1.0, 0.5}), (brain::Values{0.0}));
    brain.init<brain::NetworkType::Brain>(20);
    brain.size(3, 2);
    EXPECT_EQ(brain.exec({0.1, 0.2, 0.3}).size(), 2u);
}

TEST(NeuralNetwork, TrainingUsesTheStatefulForward) {
    brain::NeuralNetwork network;
    const brain::Dataset inputs = {{1.0, 1.0}};
    const brain::Dataset expected = {{1.0}};

    network.init<brain::NetworkType::Layered>(brain::Layers{2, 1});
    network.size(2, 1);
    network.internalLossFunction(brain::LossFunction::MSE);
    network.inputs({1.0, 1.0});
    EXPECT_EQ(network.apply(), (brain::Values{4.0}));
    EXPECT_NO_THROW(network.train(brain::UIType(1), inputs, expected));
}

TEST(NeuralNetwork, FrozenOnlyExecutes) {
    brain::NeuralNetwork network;

    network.init<brain::NetworkType::Layered>(2, 3);
    network.size(2, 1);
    network.freeze();
    EXPECT_TRUE(network.isFrozen());
    EXPECT_NO_THROW((void)network.exec({1.0, 2.0}));
    EXPECT_TRUE(throwsInternal([&] {network.size(1, 1);}, utils::exception::InternalCode::Frozen));
    EXPECT_TRUE(throwsInternal([&] {network.rand(0.0, 1.0);}, utils::exception::InternalCode::Frozen));
    EXPECT_TRUE(throwsInternal([&] {network.learningRate(0.5);}, utils::exception::InternalCode::Frozen));
    EXPECT_TRUE(throwsInternal([&] {network.inputs({1.0, 2.0});}, utils::exception::InternalCode::Frozen));
    EXPECT_TRUE(throwsInternal([&] {network.clear();}, utils::exception::InternalCode::Frozen));
    EXPECT_TRUE(throwsInternal([&] {network.init<brain::NetworkType::Brain>(3);}, utils::exception::InternalCode::Frozen));
}

TEST(NeuralNetwork, LearningRateIsChecked) {
    brain::NeuralNetwork network;

    EXPECT_NO_THROW(network.learningRate(0.5, 10, {true, 0.9}));
    EXPECT_EQ(network.learningRate(), 0.5f);
    EXPECT_TRUE(throwsCode([&] {network.learningRate(0.0);}, utils::exception::ExternalCode::InvalidLearningRate));
    EXPECT_TRUE(throwsCode([&] {network.learningRate(-1.0);}, utils::exception::ExternalCode::InvalidLearningRate));
    EXPECT_TRUE(throwsCode([&] {network.learningRate(std::nanf(""));}, utils::exception::ExternalCode::InvalidLearningRate));
    EXPECT_TRUE(throwsCode([&] {network.learningRate(0.5, 10, {true, 0.0});}, utils::exception::ExternalCode::InvalidLearningRate));
}

TEST(NeuralNetwork, TrainChecksItsParameters) {
    brain::NeuralNetwork network;
    const brain::Dataset data = {{1.0, 2.0}};

    EXPECT_TRUE(throwsCode([&] {network.train(brain::UIType(1), data, data);}, utils::exception::ExternalCode::NoNetwork));
    network.init<brain::NetworkType::Layered>(2, 3);
    EXPECT_TRUE(throwsInternal([&] {network.train(brain::UIType(1), data, data);}, utils::exception::InternalCode::InvalidArgument)); // no loss function
    network.internalLossFunction(brain::LossFunction::MSE);
    EXPECT_TRUE(throwsInternal([&] {network.train(brain::UIType(1), {}, {});}, utils::exception::InternalCode::InvalidArgument)); // empty
    EXPECT_TRUE(throwsInternal([&] {network.train(brain::UIType(1), data, {{1.0}, {2.0}});}, utils::exception::InternalCode::InvalidArgument)); // not the same number of samples
    network.size(2, 1);
    EXPECT_TRUE(throwsCode([&] {network.train(brain::UIType(1), {{1.0, 2.0, 3.0}}, {{1.0}});}, utils::exception::ExternalCode::InvalidInputSize));
    EXPECT_TRUE(throwsCode([&] {network.train(brain::UIType(1), data, {{1.0, 2.0}});}, utils::exception::ExternalCode::InvalidOutputSize));
    EXPECT_TRUE(throwsInternal([&] {network.train(brain::FType(-1.0), {{1.0, 2.0}}, {{1.0}});}, utils::exception::InternalCode::InvalidArgument)); // negative loss trigger
}

TEST(NeuralNetwork, LossOfOneOutput) {
    brain::NeuralNetwork network;

    EXPECT_THROW((void)network.loss(0.5f, 1.0f), utils::exception::IException); // no loss function
    network.internalLossFunction(brain::LossFunction::MSE);
    EXPECT_FLOAT_EQ(network.loss(3.0f, 1.0f), 4.0f);
    network.lossFunction([](const brain::FType output, const brain::FType expected) {return output - expected;});
    EXPECT_FLOAT_EQ(network.loss(3.0f, 1.0f), 2.0f); // the external one wins
}

TEST(NeuralNetwork, LossOfSeveralOutputs) {
    brain::NeuralNetwork network;
    const brain::Values outputs = {1.0, 3.0};
    const brain::Values expected = {0.0, 0.0};

    network.internalLossFunction(brain::LossFunction::MSE);
    EXPECT_FLOAT_EQ(network.loss<true>(outputs, expected), 5.0f); // mean of 1 and 9
    EXPECT_EQ(network.loss<false>(outputs, expected), (brain::Values{1.0, 9.0}));
    EXPECT_EQ(network.loss<false>(outputs, expected, 2.0f), (brain::Values{2.0, 18.0})); // the coef applies to every value
    const std::pair<brain::Values, brain::FType> both = network.loss(outputs, expected, 0.5f);
    EXPECT_EQ(both.first, (brain::Values{0.5, 4.5}));
    EXPECT_FLOAT_EQ(both.second, 2.5f);
    EXPECT_THROW((void)network.loss<true>(outputs, {0.0}), utils::exception::IException); // not the same number of values
    EXPECT_FLOAT_EQ(network.loss<true>({}, {}), 0.0f);
}

/* ------------------------------- save / load ------------------------------- */
TEST(NeuralNetwork, SaveAndLoadALayeredNetwork) {
    brain::NeuralNetwork network;
    brain::NeuralNetwork loaded;
    const std::filesystem::path path = tmpFile("layer.ln");

    network.init<brain::NetworkType::Layered>(brain::Layers{3, 2});
    network.size(4, 2);
    network.rand(-0.5, 0.25);
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    ASSERT_NO_THROW(network.save(path));
    ASSERT_NO_THROW(loaded.load(path));
    EXPECT_EQ(loaded.neurons(), network.neurons());
    EXPECT_EQ(loaded.size(), network.size());
    EXPECT_EQ(loaded.rand().first, network.rand().first);
    EXPECT_EQ(loaded.activationFunction().first, brain::ActivationFunction::Sigmoid);
    std::filesystem::remove(path);
}

TEST(NeuralNetwork, SaveAndLoadEveryKindOfBrainNetwork) {
    brain::NeuralNetwork network;
    brain::NeuralNetwork loaded;
    const std::filesystem::path raw = tmpFile("raw.rbn");
    const std::filesystem::path any = tmpFile("any.net");

    network.init<brain::NetworkType::Brain>(20);
    ASSERT_NO_THROW(network.save(raw));
    ASSERT_NO_THROW(network.save(any)); // .net: any kind of network
    ASSERT_NO_THROW(loaded.load(raw));
    EXPECT_EQ(loaded.neurons(), (std::pair<brain::UIType, brain::UIType>{0, 20}));
    ASSERT_NO_THROW(loaded.load(any));
    EXPECT_EQ(loaded.neurons(), (std::pair<brain::UIType, brain::UIType>{0, 20}));
    std::filesystem::remove(raw);
    std::filesystem::remove(any);
}

TEST(NeuralNetwork, AnUnknownExtensionOnlyWarns) {
    brain::NeuralNetwork network;
    brain::NeuralNetwork loaded;
    const std::filesystem::path path = tmpFile("other.dat");

    network.init<brain::NetworkType::Layered>(2, 2);
    EXPECT_NO_THROW(network.save(path));
    EXPECT_NO_THROW(loaded.load(path));
    std::filesystem::remove(path);
}

TEST(NeuralNetwork, ACustomActivationIsNotRestored) {
    brain::NeuralNetwork network;
    brain::NeuralNetwork loaded;
    const std::filesystem::path path = tmpFile("custom.ln");

    network.init<brain::NetworkType::Layered>(2, 2);
    network.activationFunction([](const brain::FType value) {return value;});
    ASSERT_NO_THROW(network.save(path));
    ASSERT_NO_THROW(loaded.load(path)); // a warning, not an error
    EXPECT_EQ(loaded.activationFunction().first, brain::ActivationFunction::None);
    EXPECT_EQ(loaded.activationFunction().second, nullptr);
    std::filesystem::remove(path);
}

TEST(NeuralNetwork, LoadRefusesABadFile) {
    brain::NeuralNetwork network;
    const std::filesystem::path path = tmpFile("bad.net");

    network.init<brain::NetworkType::Layered>(2, 2);
    EXPECT_TRUE(throwsCode([&] {network.load(tmpFile("missing.net"));}, utils::exception::ExternalCode::InvalidNetworkFile));
    std::ofstream(path, std::ios::binary).put(0x22); // only the first magic byte
    EXPECT_TRUE(throwsCode([&] {network.load(path);}, utils::exception::ExternalCode::InvalidNetworkFile));
    std::ofstream(path, std::ios::binary) << "\x22z\x00"; // unknown kind of network
    EXPECT_TRUE(throwsCode([&] {network.load(path);}, utils::exception::ExternalCode::InvalidNetworkFile));
    std::ofstream(path, std::ios::binary) << "\x01l\x00";
    EXPECT_TRUE(throwsCode([&] {network.load(path);}, utils::exception::ExternalCode::InvalidNetworkFile));
    EXPECT_EQ(network.neurons(), (std::pair<brain::UIType, brain::UIType>{2, 4})); // the network is kept
    std::filesystem::remove(path);
}

TEST(NeuralNetwork, GetGivesTheNetworkWithoutItsOwnership) {
    brain::NeuralNetwork network;

    EXPECT_THROW((void)network.get(), utils::exception::IException); // no network yet
    EXPECT_THROW((void)std::as_const(network).get(), utils::exception::IException);
    network.init<brain::NetworkType::Layered>(brain::Layers{3, 1});
    brain::INetwork& net = network.get();
    net.size(2, 1); // a change through the reference is seen by the NeuralNetwork: the same object
    EXPECT_EQ(network.size(), (std::pair<brain::UIType, brain::UIType>{2, 1}));
    EXPECT_EQ(&network.get(), &net); // still owned by the NeuralNetwork
    EXPECT_EQ(&std::as_const(network).get(), &net);
    network.freeze();
    EXPECT_THROW((void)network.get(), utils::exception::IException); // could change a frozen network
    EXPECT_EQ(std::as_const(network).get().neurons().second, 4u); // reading stays allowed
}

// train sends the signed error: the output moves towards the target from above and from below
TEST(NeuralNetwork, TrainMovesTheOutputTowardsTheTarget) {
    for (const brain::FType target: {brain::FType(0.1), brain::FType(0.9)}) {
        brain::NeuralNetwork layered;
        brain::NeuralNetwork brainy;
        const brain::Dataset inputs = {{0.3, 0.7}};
        const brain::Dataset expected = {{target}};

        brain::tools::rand(5);
        layered.init<brain::NetworkType::Layered>(brain::Layers{2, 3, 1});
        layered.size(2, 1);
        layered.internalActivationFunction(brain::ActivationFunction::Sigmoid);
        layered.weightInit();
        layered.internalLossFunction(brain::LossFunction::MSE);
        layered.learningRate(1.0);
        brainy.init<brain::NetworkType::Brain>(12, {4.0, 4.0});
        brainy.size(2, 1);
        brainy.internalActivationFunction(brain::ActivationFunction::Sigmoid);
        brainy.weightInit();
        brainy.internalLossFunction(brain::LossFunction::MSE);
        brainy.learningRate(1.0);
        for (brain::NeuralNetwork* network: {&layered, &brainy}) {
            const brain::FType before = std::abs(network->exec(inputs[0])[0] - target);

            network->train(brain::UIType(50), inputs, expected);
            EXPECT_LT(std::abs(network->exec(inputs[0])[0] - target), before);
        }
    }
}

TEST(NeuralNetwork, TrainWithTheMeanSendsOneCoefficient) {
    brain::NeuralNetwork network;
    const brain::Dataset inputs = {{0.3, 0.7}};
    const brain::Dataset expected = {{0.9, 0.1}};

    brain::tools::rand(3);
    network.init<brain::NetworkType::Layered>(brain::Layers{2, 2});
    network.size(2, 2);
    network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
    network.weightInit();
    network.internalLossFunction(brain::LossFunction::MSE);
    const brain::Values before = network.exec(inputs[0]);
    network.train<true>(brain::UIType(1), inputs, expected);
    const brain::Values after = network.exec(inputs[0]);

    // one coefficient for both outputs: they move the same way
    EXPECT_EQ(after[0] > before[0], after[1] > before[1]);
}

TEST(NeuralNetwork, ScoreFunctionAndRateShapeWhatTrainSends) {
    brain::NeuralNetwork brainy;
    brain::NeuralNetwork layered;

    EXPECT_TRUE(throwsCode([&] {brainy.scoreRate(2.0);}, utils::exception::ExternalCode::NoNetwork));
    EXPECT_TRUE(throwsCode([&] {brainy.internalScoreFunction(brain::ScoreFunction::Sign);}, utils::exception::ExternalCode::NoNetwork));
    brainy.init<brain::NetworkType::Brain>(12, {4.0, 4.0});
    layered.init<brain::NetworkType::Layered>(brain::Layers{2, 3, 1});
    brainy.size(2, 1);
    layered.size(2, 1);

    // the internal ones, a brain network (a score: expected - output) and a layered one (an error: output - expected), then the score rate
    const brain::Values outputs = {0.25, 0.75};
    const brain::Values expected = {0.75, 0.25};
    for (const brain::NetworkType kind: {brain::NetworkType::Brain, brain::NetworkType::Layered}) {
        brain::NeuralNetwork& network = kind == brain::NetworkType::Brain ? brainy : layered;
        const brain::FType sign = kind == brain::NetworkType::Brain ? 1.0f : -1.0f; // the error of the layered network is the opposite of a score
        const brain::INetwork& inner = network.get();

        EXPECT_EQ(inner.scores(outputs, expected), (brain::Values{0.5f * sign, -0.5f * sign})); // Difference by default
        network.internalScoreFunction(brain::ScoreFunction::Sign);
        EXPECT_EQ(inner.scores(outputs, expected), (brain::Values{sign, -sign}));
        network.internalScoreFunction(brain::ScoreFunction::Loss);
        EXPECT_EQ(inner.scores(outputs, expected), (brain::Values{0.25f, 0.25f}));
        network.internalScoreFunction(brain::ScoreFunction::Difference);
        network.scoreRate(3.0);
        EXPECT_EQ(inner.scores(outputs, expected), (brain::Values{1.5f * sign, -1.5f * sign}));
        EXPECT_THROW((void)inner.scores(brain::Values{0.5}, expected), utils::exception::IException);
    }
    EXPECT_THROW(brainy.scoreRate(0.0), utils::exception::IException);
    EXPECT_THROW(brainy.scoreRate(-1.0), utils::exception::IException);
    EXPECT_THROW(brainy.scoreFunction(nullptr), utils::exception::IException);

    // a function wins over the internal one and is the whole score (the score rate still applies)
    brainy.scoreFunction([](const brain::FType output, const brain::FType target) {return output + 10 * target;});
    EXPECT_EQ(brainy.get().scores(outputs, expected), (brain::Values{3.0f * (0.25f + 7.5f), 3.0f * (0.75f + 2.5f)}));
}

TEST(NeuralNetwork, TrainSendsTheScoreTimesTheRates) {
    // a custom score of 1 for every output: the weights move with the score rate, further with a rate of 2
    brain::FType moved[2] = {0, 0};

    for (brain::UIType k = 0; k < 2; ++k) {
        brain::NeuralNetwork network;
        brain::tools::rand(5);
        network.init<brain::NetworkType::Layered>(brain::Layers{2, 3, 1});
        network.size(2, 1);
        network.internalActivationFunction(brain::ActivationFunction::Linear);
        network.weightInit();
        network.biasInit();
        network.internalLossFunction(brain::LossFunction::MSE);
        network.learningRate(0.5);
        network.scoreRate(k == 0 ? 1.0f : 2.0f);
        network.scoreFunction([&](const brain::FType, const brain::FType) {return 1.0f;});
        const brain::FType before = network.exec({0.3, 0.7})[0];

        network.train(brain::UIType(1), brain::Dataset{{0.3, 0.7}}, brain::Dataset{{0.0}});
        moved[k] = std::abs(network.exec({0.3, 0.7})[0] - before);
    }
    EXPECT_GT(moved[0], 0.0f);
    EXPECT_GT(moved[1], 1.5f * moved[0]); // the output is not linear in the weights: more than that, not exactly twice
}
