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
##  @file NetworkType.hpp

File Description:
##  Value types, data / hook aliases and the neuron structs of the neural network
\**************************************************************/

#ifndef BRAIN_NETWORKTYPE_H
    #define BRAIN_NETWORKTYPE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _CustomType
    #define _System
    #include <utils/utils.hpp>   // _hot, _nodiscard, utils::type::OVector2
    #include "NetworkDefine.hpp" // brain::StopCondition, brain::StopReason
    #include <type_traits>       // std::is_floating_point_v, std::is_integral_v, std::is_signed_v, std::is_unsigned_v, std::is_trivially_copyable_v
    #include <functional>        // std::function
    #include <cstdint>           // std::int64_t
    #include <cstddef>           // std::size_t
    #include <utility>           // std::pair
    #include <vector>            // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* TYPEDEF */

/* values */
using IType = std::int64_t; // signed integer type (seeds)
using UIType = std::size_t; // unsigned integer type (sizes, counts, ids)
using FType = float; // floating type of every neuron / coefficient value (float, double, ...)
static_assert(std::is_integral_v<brain::IType> && std::is_signed_v<brain::IType>, "IType must be a signed integral type");
static_assert(std::is_integral_v<brain::UIType> && std::is_unsigned_v<brain::UIType>, "UIType must be an unsigned integral type");
static_assert(std::is_floating_point_v<brain::FType>, "FType must be a floating point type");

/* data */
using Layers = std::vector<brain::UIType>; // number of neurons of each layer
using Bytes = std::vector<std::uint8_t>; // raw data of a saved network
using Values = std::vector<brain::FType>; // inputs / outputs / coefficients of one sample
using Dataset = std::vector<brain::Values>; // several samples

/* hook */
using LinksCapFn = std::function<brain::UIType(const brain::UIType n)>; // external links limit computing function
using StepsCapFn = std::function<brain::UIType(const brain::UIType n)>; // external time steps limit computing function (brain network)
using ActivationFn = std::function<brain::FType(const brain::FType value)>; // external activation function
using LossFn = std::function<brain::FType(const brain::FType output, const brain::FType expected)>; // external loss function
using ScoreFn = std::function<brain::FType(const brain::FType output, const brain::FType expected)>; // external score function: the value sent to the backpropagation for one output (brain network: positive = good, layered network: positive = too high)
using RandFn = std::function<brain::FType(const brain::IType seed)>; // weight / bias generator from the seed

//----------------------------------------------------------------//
/* STRUCT */

// Neuron of a layered network (trivially copyable: stored in a utils::system::HybridMemory), its weights are in the weights memory of the network
struct LayerNeuron {
    brain::UIType incoming = 0; // index of the first weight in the weights memory (one weight per input of the neuron)
    brain::UIType count = 0; // number of weights
    brain::FType bias = 0.0;
};

// Input link of a reduced brain network neuron: the output of the neuron id times the weight (a free slot is all at 0: no effect)
struct BrainLink {
    brain::UIType id = 0; // id of the neuron sending its output
    brain::FType weight = 0.0;
};

// Neuron of a reduced brain network (execution only): the bias, its input links are in the links memory of the network
struct BrainNeuron {
    brain::FType bias = 0.0;
};

// Link of a raw brain network neuron, a link between two neurons is found 2 times (one slot on each neuron: the receiver '<-' for the execution, the sender '->' a clone with the same weight for the backpropagation)
struct RawBrainLink {
    brain::UIType id = 0; // id of the other neuron
    brain::FType weight = 0.0; // weight of the link
    bool inward = false; // true: input from the other neuron (execution) | false: output to the other neuron (only used for the backpropagation)
};

// Neuron of a raw brain network (execution and training), placed in a 2d space
struct RawBrainNeuron {
    utils::type::OVector2<brain::FType> pos; // position in the 2d space of the network
    brain::FType bias = 0.0;

    _hot _nodiscard inline brain::FType distance2(const RawBrainNeuron& other) const {return (this->pos - other.pos).lengthSquared();}; // squared distance between the positions (no sqrt: only the order matters, every distance stays squared and comparable)
    _hot _nodiscard inline brain::FType operator|(const RawBrainNeuron& other) const {return this->distance2(other);}; // squared distance (same as distance2)

    // Copy only (defaulted): without move operations, the non trivial move assignment of utils::type::OVector2 doesn't make the neuron non trivially copyable
    RawBrainNeuron() = default;
    RawBrainNeuron(const RawBrainNeuron& other) = default;
    RawBrainNeuron& operator=(const RawBrainNeuron& other) = default;
};

// What the stop function of a training sees after each epoch
struct TrainState {
    brain::UIType epoch = 0; // epochs done (this one included)
    brain::UIType updates = 0; // updates of the weights since the start of the training
    brain::FType loss = 0.0; // mean loss of the epoch on the training set
    brain::FType validationLoss = 0.0; // mean loss on the validation set (0 without one)
    brain::FType bestValidation = 0.0; // the best validation loss seen
    brain::UIType sinceBest = 0; // epochs since the best validation loss
    double seconds = 0.0; // time since the start of the training
};

struct TrainOptions;
using StopFn = std::function<bool(const brain::TrainOptions& options, const brain::TrainState& state)>; // external stop rule: true = go on, false = stop

// The settings of NeuralNetwork::train, every one has its default (the whole struct can be left out: one epoch, one sample at a time, as before)
struct TrainOptions {
    brain::UIType epochs = 1; // at most this many passes over the samples (0 = no limit: another rule has to stop it)
    brain::UIType batchSize = 1; // samples per update of the weights: 1 = after each one, n = mini-batch of n (their mean gradient), 0 = the whole set
    bool shuffle = false; // the samples in another order at each epoch
    brain::IType seed = 0; // seed of the shuffle (0 = random)
    bool mean = false; // one coefficient for the backpropagation (the mean of the outputs) instead of one per output
    brain::FType targetLoss = -1.0; // StopCondition::TargetLoss: stop when the mean loss is at most this
    brain::UIType patience = 0; // StopCondition::Patience: stop after this many epochs without a better validation loss
    brain::Dataset validationIn; // the validation set, measured after each epoch (without update)
    brain::Dataset validationOut; // its expected outputs
    brain::StopCondition stop = brain::StopCondition::Epochs; // the built-in rule
    brain::StopFn onEpoch = nullptr; // the rule of the user, after each epoch: it wins over `stop` (epochs stays the limit)
};

// What a training did
struct TrainResult {
    brain::UIType epochs = 0; // epochs done
    brain::UIType updates = 0; // updates of the weights
    brain::FType loss = 0.0; // mean loss of the last epoch
    brain::StopReason reason = brain::StopReason::Epochs; // why it stopped
};

// Everything stored in a utils::system::HybridMemory is copied as bytes
static_assert(std::is_trivially_copyable_v<brain::LayerNeuron> && std::is_trivially_copyable_v<brain::BrainNeuron> && std::is_trivially_copyable_v<brain::BrainLink>, "the neurons and the links are stored in a HybridMemory");
static_assert(std::is_trivially_copyable_v<brain::RawBrainNeuron> && std::is_trivially_copyable_v<brain::RawBrainLink>, "the raw neurons and links are stored in a HybridMemory");

} // namespace end
#endif /* BRAIN_NETWORKTYPE_H */
