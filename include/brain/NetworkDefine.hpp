/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 30/10/2026 by @author Tsukini

File Name:
##  @file NetworkDefine.hpp

File Description:
##  Enums of the neural network: kind of network, internal activation & loss functions
\**************************************************************/

#ifndef BRAIN_NETWORKDEFINE_H
    #define BRAIN_NETWORKDEFINE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include <cstdint>  // std::uint32_t
    #include <cmath>    // std::log, std::log2

    //----------------------------------------------------------------//
    /* DEFINE */

    /* file extensions (a different one is accepted with a warning) */
    #define BRAIN_EXTENSION_BRAIN ".bn" // reduced brain network (execution only): BRAIN_MAGIC_BRAIN
    #define BRAIN_EXTENSION_RAW ".rbn" // raw brain network (execution and training): BRAIN_MAGIC_RAW
    #define BRAIN_EXTENSION_LAYER ".ln" // layer network: BRAIN_MAGIC_LAYER
    #define BRAIN_EXTENSION_NETWORK ".net" // any network, the magic bytes tell which one

    /* magic bytes (two first bytes of the serialized data: the common one, then one per kind of network) */
    #define BRAIN_MAGIC 0x22 // common first byte
    #define BRAIN_MAGIC_BRAIN 'b' // 0x62, second byte of a reduced brain network
    #define BRAIN_MAGIC_RAW 'r' // 0x72, second byte of a raw brain network
    #define BRAIN_MAGIC_LAYER 'l' // 0x6C, second byte of a layer network

    /* activation byte (third byte of the serialized data: how the activation function is restored) */
    #define BRAIN_ACTIVATION_NONE 0x00 // no activation function (raw values, default)
    #define BRAIN_ACTIVATION_CUSTOM 0xFF // external function (std::function): can't be saved, to give again after the load
    // 0x01 -> 0xFE: internal mode, the value of brain::ActivationFunction

    /* limits */
    #define BRAIN_LINKS_CAP_PERCENTAGE 1.0 // (default value) percentage of the total number of neurons
    #define BRAIN_LINKS_CAP_COMPUTING(n) (3.0 * std::sqrt(static_cast<double>(n))) // (default value) computing of the default limits of links: 3 sqrt(n) (12 links at 16 neurons, 94 at 1000, 300 at 10000)

    #define BRAIN_STEPS_PATH_MARGIN 1.1 // the time steps are never under this times the farthest path (in links) from an input to an output, whatever the cap (the signal has to arrive)
    #define BRAIN_THREADS_MIN_WORK 131072 // a time step is shared between threads only when it has at least this many slots of links (neurons * cap) for each of them: under it the barrier of each step costs more than it saves
    #define BRAIN_THREADS_CHUNK 64 // the neurons of a thread are whole chunks of this many (the loops over the neurons are vectorized: a range that does not start and end on the same lanes as the whole loop would round the last neurons differently, the results must not depend on the number of threads)
    #define BRAIN_GPU_SINGLE_KERNEL_NEURONS 256 // on a gpu a network of at most this many neurons and a cap under 32 (one thread per neuron) computes all its time steps in one launch of one block, with a barrier between the steps; a bigger one, or one with a warp per neuron, launches a kernel per step on the whole device (measured: the one block is slower there)
    #define BRAIN_SIGNAL_THRESHOLD 0.0 // (default value, placeholder to give) a signal under it (absolute value) is cancelled in a brain network: 0 = nothing is cancelled
    #define BRAIN_WEIGHT_MIN 0.01 // (default value) a raw brain network removes a link (and its clone) when the absolute value of its weight falls to it (or under), unless it is the last link the neuron needs
    #define BRAIN_WEIGHT_MAX 2.0 // (default value) a raw brain network links the neuron to its nearest free neuron (within the cap) when the absolute value of the weight of one of its links rises to it (or over)
    #define BRAIN_STEPS_CAP_PERCENTAGE 1.0 // (default value) percentage of the total number of neurons
    #define BRAIN_STEPS_CAP_COMPUTING(n) (std::log2(static_cast<double>(n))) // (default value) computing of the default number of time steps of a brain network from the number of neurons n: a signal crosses the network in about log2(n) hops, twice for the loops

namespace brain { // namespace start
//----------------------------------------------------------------//
/* ENUM */

// Kind of network stored behind a brain::NeuralNetwork
enum class NetworkType {
    Layered, // classic layers of neurons (brain::LayerNetwork)
    Brain, // neurons placed in a 2d space (brain::BrainNetwork)
};

// How a text is converted to the inputs of a network (brain::tools::text_to_input / input_to_text)
enum class TextMode {
    Normal, // one value per character: its byte divided by 255 (small values, exact, the size of the text)
    Compact, // the bytes of the text copied as is in the floats (4 characters per value): compact, but huge values / NaN, only to carry the bytes
};

// Internal activation functions (more optimized than an external one)
enum class ActivationFunction {
    None = BRAIN_ACTIVATION_NONE, // raw values
    Sigmoid,
    //Softmax,
    Linear,
    ReLU,
};

// Internal loss functions (more optimized than an external one)
enum class LossFunction {
    None, // no loss computed
    MSE, // mean squared error
    CrossEntropy,
};

// Internal score functions: what is sent to the backpropagation for one output, before the rates (the error d = expected - output for a brain network, output - expected for a layered one)
enum class ScoreFunction {
    Difference, // d
    Sign, // -1, 0 or 1 (the direction of d only)
    Loss, // d squared
};

// The on / off rules of a raw brain network, one bit each (like the flags of the cli of libutils): addRules(), subRules(), setRules(), resetRules(), clearRules() and getRules() of brain::RawBrainNetwork
enum Rule: std::uint32_t {
    ORPHANS = 1u << 0, // the backpropagation can remove the last link of a neuron
    LEFT_TO_RIGHT = 1u << 1, // the links only go from the left to the right (no loop)
    CLAMP_WEIGHTS = 1u << 2, // |weight| and |bias| kept under weightMax
    LOW_NEW_LINKS = 1u << 3, // a new link starts at 5 % of the limits
    DIRECT_OUTPUT = 1u << 4, // the score reaches the sum of an output as it is (a saturated output still learns)
    KEEP_PATHS = 1u << 5, // a link is never removed when it takes the last path of an input to an output away
    BATCH = 1u << 6, // mini-batch: the gradients are accumulated, applyBatch() applies their mean
    RANDOM_LINKS = 1u << 7, // a random number of links per neuron at the init (between minLinks and the cap)
    INVERTED_LINKS = 1u << 8, // at the limits: a link is removed at weightMax and a new one grows at weightMin
    INPUT_PATHS = 1u << 9, // every input keeps a path to at least one output (built at the init, never cut by the training)
    MOVE_LAST_LINK = 1u << 10, // the last link of a neuron that is too weak is moved to another neuron instead of being kept
};

// Built-in stop rules of a training (brain::TrainOptions::stop), checked after each epoch
enum class StopCondition {
    Epochs, // only the number of epochs stops it
    TargetLoss, // the mean loss of the epoch is at most TrainOptions::targetLoss
    Patience, // the loss on the validation set did not improve for TrainOptions::patience epochs
};

// Why a training stopped (brain::TrainResult::reason)
enum class StopReason {
    Epochs, // the epochs asked are done
    TargetLoss, // the loss asked is reached
    Patience, // the validation loss stopped improving
    Callback, // the function of the options said stop
};

namespace rules { // namespace brain::rules: the sets of rules
constexpr std::uint32_t ALL = ORPHANS | LEFT_TO_RIGHT | CLAMP_WEIGHTS | LOW_NEW_LINKS | DIRECT_OUTPUT | KEEP_PATHS | BATCH | RANDOM_LINKS | INVERTED_LINKS | INPUT_PATHS | MOVE_LAST_LINK;
constexpr std::uint32_t DEFAULT = RANDOM_LINKS; // what a new network has, what resetRules() gives back
constexpr std::uint32_t NONE = 0; // everything off, what clearRules() gives
constexpr std::uint32_t LAB = ORPHANS | LOW_NEW_LINKS | KEEP_PATHS | DIRECT_OUTPUT | BATCH; // the rules of the lab (as many links as the cap at the init, a saturated output still learns, mini-batch, the last paths kept)
} // namespace rules

} // namespace end
#endif /* BRAIN_NETWORKDEFINE_H */
