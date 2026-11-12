/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 12/11/2026 by @author Tsukini

File Name:
##  @file NetworkType.hpp

File Description:
##  Types of the networks: values, storage, functions, neurons and the options / state / result of a training
\**************************************************************/

#ifndef BRAIN_NETWORKTYPE_H
    #define BRAIN_NETWORKTYPE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>     // _noaddress
    #include "NetworkDefine.hpp"   // brain::ActivationFunction, brain::LossFunction, brain::ScoreFunction, brain::RandInit, brain::NormalizeMode, brain::StopReason, brain::flags, BRAIN_DEFAULT_*
    #include <type_traits>         // std::is_floating_point_v, std::is_integral_v, std::is_signed_v, std::is_unsigned_v, std::conditional_t
    #include <filesystem>          // std::filesystem::path
    #include <functional>          // std::function
    #include <optional>            // std::optional
    #include <variant>             // std::variant, std::monostate
    #include <utility>             // std::pair
    #include <cstdint>             // std::int64_t, std::uint8_t, std::uint16_t, std::uint32_t
    #include <cstddef>             // std::size_t
    #include <memory>              // std::shared_ptr
    #include <vector>              // std::vector
    #include <span>                // std::span

namespace brain { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

class ITrainHook; // hook/ITrainHook.hpp (brain::TrainOptions::hook)

//----------------------------------------------------------------//
/* TYPEDEF */

/* values */
using IType = std::int64_t; // signed integer type (seeds)
using UIType = std::size_t; // unsigned integer type (sizes, counts, ids)
using FType = float; // floating type of every neuron / coefficient value (float, double, ...)
using Value = brain::FType; // one input / output / expected value of a neuron or of the network (the coefs / losses / weights / gradients stay FType)
static_assert(std::is_integral_v<brain::IType> && std::is_signed_v<brain::IType>, "IType must be a signed integral type");
static_assert(std::is_integral_v<brain::UIType> && std::is_unsigned_v<brain::UIType>, "UIType must be an unsigned integral type");
static_assert(std::is_floating_point_v<brain::FType>, "FType must be a floating point type");

/* data */
using Layers = std::vector<brain::UIType>; // number of neurons of each layer
using Bytes = std::vector<std::uint8_t>; // raw data of a saved network
using Values = std::vector<brain::Value>; // inputs / outputs / coefficients of one sample
using Dataset = std::vector<brain::Values>; // several samples

/* storage (where a network keeps its weights / caches, in the ram) */
using Buffer = std::vector<brain::FType>;
using Slice = std::span<brain::FType>; // a part of a buffer (one layer / one neuron)
using CSlice = std::span<const brain::FType>; // a read only part of a buffer

/* activation (internal enum / fn on one value / fn on the whole vector), the derivative always from the output */
using ActivationFn = std::pair<std::function<brain::Value(const brain::Value value)>, std::function<brain::FType(const brain::Value output)>>; // <activation, derivative>
using ActivationVectorFn = std::pair<std::function<brain::Values(const brain::Values& values)>, std::function<brain::Values(const brain::Values& outputs)>>; // <activation, derivative> on the whole layer (softmax like)
using Activation = std::variant<brain::ActivationFunction, brain::ActivationFn, brain::ActivationVectorFn>;

/* loss (same variants) */
using LossFn = std::pair<std::function<brain::FType(const brain::Value output, const brain::Value expected)>, std::function<brain::FType(const brain::Value output, const brain::Value expected)>>; // <loss, derivative>
using LossVectorFn = std::pair<std::function<brain::FType(const brain::Values& outputs, const brain::Values& expected)>, std::function<brain::Values(const brain::Values& outputs, const brain::Values& expected)>>; // <mean loss, derivative of each output>
using Loss = std::variant<brain::LossFunction, brain::LossFn, brain::LossVectorFn>;

/* score (same variants) */
using ScoreFn = std::function<brain::FType(const brain::FType learning_rate, const brain::UIType step, const brain::FType original_learning_rate, const brain::FType loss)>; // the coef sent to the backpropagation (value / mean mode)
using ScoreVectorFn = std::function<brain::Values(const brain::FType learning_rate, const brain::UIType step, const brain::FType original_learning_rate, const brain::Values& losses)>; // one coef per output (vector mode)
using Score = std::variant<brain::ScoreFunction, brain::ScoreFn, brain::ScoreVectorFn>;

/* random */
using RandFn = std::function<brain::FType(const brain::UIType fan_in, const brain::UIType fan_out)>; // weight / bias generator (custom xavier / he like init)
using Rand = std::variant<brain::RandInit, brain::RandFn>;

/* brain caps */
using LinksCapFn = std::function<brain::UIType(const brain::UIType neurons)>; // links per neuron from the number of neurons
using StepsCapFn = std::function<brain::UIType(const brain::UIType neurons)>; // time steps from the number of neurons

//----------------------------------------------------------------//
/* STRUCT */

/* neurons */
// Neuron of a layered network, its weights are in the weights of the network
struct LayerNeuron {
    brain::UIType incoming = 0; // index of the first weight of the neuron (one weight per input of the neuron)
    brain::UIType count = 0; // number of weights
    brain::FType bias = 0.0;
};

// Input link of a reduced brain network neuron: the output of the neuron id times the weight (a free slot is all at 0: no effect)
struct BrainLink {
    brain::UIType id = 0; // id of the neuron sending its output
    brain::FType weight = 0.0;
};

// Neuron of a reduced brain network (execution only): the bias, its input links are in the links of the network
struct BrainNeuron {
    brain::FType bias = 0.0;
};

// Link of a raw brain network neuron, a link between two neurons is found 2 times (the receiver '<-' for the execution, the sender '->' a clone for the backpropagation)
struct RawBrainLink {
    brain::UIType id = 0; // id of the other neuron
    brain::FType weight = 0.0; // weight of the link
    bool inward = false; // true: input from the other neuron (execution) | false: output to the other neuron (only used for the backpropagation)
};

// Neuron of a raw brain network (execution and training), placed in a space of n dimensions
struct RawBrainNeuron {
    std::vector<brain::FType> pos; // one value per axis (pos.size() == BrainOptions::dimension, empty in 0D: only the creation order)
    brain::FType bias = 0.0;
};

/* data */
// Params of a normalization, one per column (brain::tools::normalize / denormalize)
struct Normalization {
    /* mode */
    brain::NormalizeMode mode = brain::NormalizeMode::MinMax;

    /* params (one per column) */
    brain::Values a; // min or mean of each column
    brain::Values b; // max or deviation of each column
};

// Raw image (brain::tools::image_to_input / input_to_image)
struct Image {
    /* shape */
    brain::UIType width = 0;
    brain::UIType height = 0;
    brain::UIType channels = 0; // 1 = gray, 3 = rgb, 4 = rgba

    /* data */
    brain::Bytes pixels; // row by row, the channels of a pixel side by side (pixels.size() == width * height * channels)
};

/* options */
// Shape and behavior of a brain network (init<Brain>)
struct BrainOptions {
    /* links */
    std::variant<brain::FType, brain::LinksCapFn> links = static_cast<brain::FType>(BRAIN_DEFAULT_LINKS); // links per neuron: share of the neurons (0 < p <= 1) or fn
    brain::UIType minLinks = BRAIN_DEFAULT_MIN_LINKS; // links a neuron keeps at least
    brain::UIType farLinks = BRAIN_DEFAULT_FAR_LINKS; // links to random far neurons at the init (just random ones in 0D)
    brain::UIType pathHops = BRAIN_DEFAULT_PATH_HOPS; // every input has a path of at most n links to every output at the init (0 = off)

    /* time */
    std::variant<brain::FType, brain::StepsCapFn> steps = static_cast<brain::FType>(BRAIN_DEFAULT_STEPS); // time steps: share of the neurons or fn

    /* limits */
    brain::FType weightMin = BRAIN_DEFAULT_WEIGHT_MIN; // a link is removed under it
    brain::FType weightMax = BRAIN_DEFAULT_WEIGHT_MAX; // a link is added over it
    brain::FType threshold = BRAIN_DEFAULT_THRESHOLD; // a signal under it is cancelled

    /* space */
    std::uint16_t dimension = BRAIN_DEFAULT_DIMENSION; // 0 = no dimension (everyone near everyone, only the creation order), otherwise n dimensions
    std::vector<brain::FType> space = {BRAIN_DEFAULT_SPACE}; // size of each axis: one value = the same size on every axis, else one per axis (space.size() == dimension), ignored in 0D

    /* behavior */
    std::uint32_t flags = brain::flags::DEFAULT; // brain::Flag
};

// Shape of a layered network (init<Layered>), the same as the setters by layer
struct LayerOptions {
    std::vector<brain::Activation> activations; // one per layer (missing = the activation of the network), ex: relu hidden + softmax output
    brain::Values dropout; // probability per layer to cut a neuron during the training only (missing / 0 = none)
};

/* training */
// Every setting of a training
struct TrainOptions {
    /* loop */
    brain::UIType epochs = BRAIN_DEFAULT_EPOCHS; // 0 = no limit (another rule has to stop it)
    brain::UIType batchSize = BRAIN_DEFAULT_BATCH_SIZE; // 1 = update after each sample, n = mini-batch, 0 = the whole set
    bool shuffle = false; // another order at each epoch
    brain::IType seed = 0; // 0 = random (weights init, shuffle, validation split, dropout)

    /* functions */
    brain::Loss loss = brain::LossFunction::None;
    brain::Score score = brain::ScoreFunction::Default;

    /* learning rate */
    brain::FType learningRate = BRAIN_DEFAULT_LEARNING_RATE; // the start one, never edited by the training (the decayed one is TrainState::learningRate)
    brain::UIType decay = 0; // learning rate decay every n steps (0 = none)
    std::pair<bool, brain::FType> decayCoef = {true, BRAIN_DEFAULT_DECAY_COEF}; // <smooth, coef>
    brain::FType stepLimit = 0.0; // biggest change of a weight per update (0 = none)

    /* validation */
    brain::FType validation = 0.0; // share of the samples kept apart to measure the training (0 = none)
    const brain::Dataset* validationIn = nullptr; // a validation set of its own, not owned (kept alive by the caller, validation share ignored when set)
    const brain::Dataset* validationOut = nullptr;

    /* stop */
    brain::UIType patience = 0; // stop after n epochs without a better validation loss (0 = never)
    brain::FType targetLoss = BRAIN_DEFAULT_TARGET_LOSS; // stop when the mean loss is at most this (< 0 = off)

    /* best network */
    bool keepBest = true; // snapshot (clone) of the best network, rollback on it at the stop if the last one is worse
    std::optional<std::filesystem::path> autosave; // save only when a new best network is found

    /* follow-up */
    std::shared_ptr<brain::ITrainHook> hook = nullptr; // shared (TrainOptions stays copyable), kept alive as long as an options / the caller holds it
};

// The current / last training
struct TrainState {
    /* progress */
    brain::UIType epoch = 0;
    brain::UIType batch = 0;
    brain::UIType updates = 0;
    double seconds = 0.0; // since the start

    /* loss */
    brain::FType loss = 0.0;
    brain::FType validationLoss = 0.0;
    brain::FType bestLoss = 0.0;
    brain::UIType sinceBest = 0; // epochs since the best loss

    /* update */
    brain::FType learningRate = 0.0; // the decayed one (from TrainOptions::learningRate, decay and updates)
};

// History of a training (TrainResult<true>)
struct TrainHistory {
    brain::Values losses; // loss of each epoch
    brain::Values validationLosses; // validation loss of each epoch
};

// What a training returns
template<bool history_enable = false>
struct TrainResult {
    /* summary */
    brain::UIType epochs = 0;
    brain::UIType updates = 0;
    brain::FType loss = 0.0; // of the best / last epoch
    brain::StopReason reason = brain::StopReason::Epochs;

    /* history (only with history_enable, can take some place) */
    _noaddress std::conditional_t<history_enable, brain::TrainHistory, std::monostate> history;
};

//----------------------------------------------------------------//
/* TYPEDEF */

/* neurons */
using RawBrainNeurons = std::vector<brain::RawBrainNeuron>; // every neuron of a raw brain network

} // namespace end
#endif /* BRAIN_NETWORKTYPE_H */
