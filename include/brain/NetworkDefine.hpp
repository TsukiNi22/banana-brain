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
##  @file NetworkDefine.hpp

File Description:
##  Defines and enums of the networks: file extensions, magic bytes, defaults, functions, flags and events
\**************************************************************/

#ifndef BRAIN_NETWORKDEFINE_H
    #define BRAIN_NETWORKDEFINE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include <cstdint>   // std::uint32_t

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

    /* default values (every numeric default of the api, can be changed at the compilation: defined before the include) */
    // serialization
    #ifndef BRAIN_DEFAULT_ENDIAN
        #define BRAIN_DEFAULT_ENDIAN brain::Endian::Little // byte order of the convertor (the save files stay the same between machines)
    #endif

    // init
    #ifndef BRAIN_DEFAULT_RAND_MIN
        #define BRAIN_DEFAULT_RAND_MIN -1.0 // min of the random weights / bias (RandInit::Uniform)
    #endif
    #ifndef BRAIN_DEFAULT_RAND_MAX
        #define BRAIN_DEFAULT_RAND_MAX 1.0 // max of the random weights / bias (RandInit::Uniform)
    #endif
    #ifndef BRAIN_DEFAULT_FALLBACK
        #define BRAIN_DEFAULT_FALLBACK 0.01 // value used by weightInit / biasInit on a result that falls on 0
    #endif

    // training
    #ifndef BRAIN_DEFAULT_LEARNING_RATE
        #define BRAIN_DEFAULT_LEARNING_RATE 1.0
    #endif
    #ifndef BRAIN_DEFAULT_EPOCHS
        #define BRAIN_DEFAULT_EPOCHS 1 // 0 = no limit (another rule has to stop it)
    #endif
    #ifndef BRAIN_DEFAULT_BATCH_SIZE
        #define BRAIN_DEFAULT_BATCH_SIZE 1 // 1 = update after each sample, n = mini-batch, 0 = the whole set
    #endif
    #ifndef BRAIN_DEFAULT_DECAY_COEF
        #define BRAIN_DEFAULT_DECAY_COEF 1.0 // coef of the learning rate decay
    #endif
    #ifndef BRAIN_DEFAULT_TARGET_LOSS
        #define BRAIN_DEFAULT_TARGET_LOSS -1.0 // < 0 = off
    #endif

    // functions
    #ifndef BRAIN_DEFAULT_LEAKY_ALPHA
        #define BRAIN_DEFAULT_LEAKY_ALPHA 0.01 // slope of leaky_relu under 0
    #endif
    #ifndef BRAIN_DEFAULT_HUBER_DELTA
        #define BRAIN_DEFAULT_HUBER_DELTA 1.0 // switch point of huber between the squared and the linear part
    #endif
    #ifndef BRAIN_DEFAULT_BITS_THRESHOLD
        #define BRAIN_DEFAULT_BITS_THRESHOLD 0.5 // a value over it is a 1 (bits_to_int / bits_to_bytes)
    #endif

    // optimizer
    #ifndef BRAIN_DEFAULT_MOMENTUM
        #define BRAIN_DEFAULT_MOMENTUM 0.9
    #endif
    #ifndef BRAIN_DEFAULT_BETA1
        #define BRAIN_DEFAULT_BETA1 0.9 // Adam
    #endif
    #ifndef BRAIN_DEFAULT_BETA2
        #define BRAIN_DEFAULT_BETA2 0.999 // Adam
    #endif
    #ifndef BRAIN_DEFAULT_EPSILON
        #define BRAIN_DEFAULT_EPSILON 1e-8 // Adam
    #endif

    // brain
    #ifndef BRAIN_DEFAULT_LINKS
        #define BRAIN_DEFAULT_LINKS 1.0 // links per neuron, share of the neurons (0 < p <= 1)
    #endif
    #ifndef BRAIN_DEFAULT_MIN_LINKS
        #define BRAIN_DEFAULT_MIN_LINKS 1 // links a neuron keeps at least
    #endif
    #ifndef BRAIN_DEFAULT_FAR_LINKS
        #define BRAIN_DEFAULT_FAR_LINKS 0 // links to random far neurons at the init
    #endif
    #ifndef BRAIN_DEFAULT_PATH_HOPS
        #define BRAIN_DEFAULT_PATH_HOPS 3 // every input has a path of at most n links to every output at the init (0 = off)
    #endif
    #ifndef BRAIN_DEFAULT_STEPS
        #define BRAIN_DEFAULT_STEPS 1.0 // time steps, share of the neurons
    #endif
    #ifndef BRAIN_DEFAULT_WEIGHT_MIN
        #define BRAIN_DEFAULT_WEIGHT_MIN 0.01 // a link is removed under it
    #endif
    #ifndef BRAIN_DEFAULT_WEIGHT_MAX
        #define BRAIN_DEFAULT_WEIGHT_MAX 5.0 // a link is added over it
    #endif
    #ifndef BRAIN_DEFAULT_NEW_LINK
        #define BRAIN_DEFAULT_NEW_LINK 0.05 // share of the limits given to a new link (LOW_NEW_LINKS)
    #endif
    #ifndef BRAIN_DEFAULT_THRESHOLD
        #define BRAIN_DEFAULT_THRESHOLD 0.0 // a signal under it is cancelled
    #endif
    #ifndef BRAIN_DEFAULT_DIMENSION
        #define BRAIN_DEFAULT_DIMENSION 2 // 0 = no dimension (only the creation order)
    #endif
    #ifndef BRAIN_DEFAULT_SPACE
        #define BRAIN_DEFAULT_SPACE 1.0 // size of every axis of the space
    #endif

    // prune
    #ifndef BRAIN_DEFAULT_PRUNE
        #define BRAIN_DEFAULT_PRUNE 0.01 // influence under 1% -> removed by prune()
    #endif

    // backend
    #ifndef BRAIN_DEFAULT_THREADS
        #define BRAIN_DEFAULT_THREADS 0 // workers of a ThreadBackend, 0 = hardware_concurrency()
    #endif
    #ifndef BRAIN_DEFAULT_GRAIN
        #define BRAIN_DEFAULT_GRAIN 64 // items per worker under which parallelFor stays on the calling thread (the threads would cost more than they save)
    #endif

namespace brain { // namespace start
//----------------------------------------------------------------//
/* ENUM */

/* network */
// Kind of network stored behind a brain::NeuralNetwork
enum class NetworkType {
    Layered, // classic layers of neurons (brain::LayerNetwork)
    Brain, // neurons placed in a space of n dimensions (brain::RawBrainNetwork, reduced in a brain::BrainNetwork)
};

// How a text is converted to the inputs of a network (brain::tools::text_to_input / input_to_text)
enum class TextMode {
    Normal, // one value per character: its byte divided by 255 (small values, exact, the size of the text)
    Compact, // the bytes of the text copied as is in the floats (4 characters per value): compact, but huge values / NaN, only to carry the bytes
};

// Byte order of the serialized data (brain::tools::put / get)
enum class Endian {
    Little,
    Big,
};

/* functions */
// Internal activation functions (more optimized than an external one)
enum class ActivationFunction {
    None = BRAIN_ACTIVATION_NONE, // raw values
    Sigmoid,
    Linear,
    ReLU,
    Tanh,
    LeakyReLU,
    Softmax, // only on the whole vector (output layer), the gradient is output - expected with CrossEntropy
};

// Internal loss functions (more optimized than an external one)
enum class LossFunction {
    None, // no loss computed
    MSE, // mean squared error
    CrossEntropy,
    MAE, // mean absolute error
    Huber, // squared under BRAIN_DEFAULT_HUBER_DELTA, linear over it
};

// Internal score functions: the coef sent to the backpropagation
enum class ScoreFunction {
    Default, // derivative of the loss * learning rate (the real gradient)
    Difference, // (expected - output) * learning rate
    Sign, // sign(expected - output) * learning rate
    Loss, // value of the loss * learning rate
};

// Internal generation of the random weights / bias
enum class RandInit {
    Uniform, // in the rand(min, max) range
    Normal, // normal distribution in the range
    Xavier, // uniform in [-sqrt(6 / (in + out)), sqrt(6 / (in + out))]
    He, // normal with a deviation of sqrt(2 / in) (relu)
};

/* data */
// Scaling of a dataset (brain::tools::normalize)
enum class NormalizeMode {
    MinMax, // in [0, 1] from the min / max of each column
    ZScore, // mean 0, deviation 1 of each column
};

/* training */
// Why a training stopped
enum class StopReason {
    Epochs, // every epoch done
    Patience, // no better validation loss for n epochs
    TargetLoss, // the mean loss reached the target
    Hook, // a hook asked for it (ITrainHook::onEpoch returned false)
};

/* prune */
// What prune() can remove
enum class PruneTarget {
    Neurons,
    Links,
    Both,
};

/* backend */
// Where the compute of a network runs
enum class BackendType {
    Cpu, // the calling thread (brain::CpuBackend)
    Threads, // several workers (brain::ThreadBackend)
};

/* brain flags */
// Behavior of a brain network, named and handled like the flags of the cli of libutils (addFlags / subFlags / setFlags / resetFlags / getFlags)
enum Flag: std::uint32_t {
    /* links */
    ORPHANS = 1u << 0, // the backpropagation can remove the last link of a neuron
    LEFT_TO_RIGHT = 1u << 1, // the links only go from the left to the right on the first axis (no loop), creation order in 0D
    LOW_NEW_LINKS = 1u << 2, // a new link starts at BRAIN_DEFAULT_NEW_LINK of the limits
    RANDOM_LINKS = 1u << 3, // a random number of links per neuron at the init (between minLinks and the cap)
    INVERTED_LINKS = 1u << 4, // at the limits: a link is removed at weightMax and a new one grows at weightMin
    MOVE_LAST_LINK = 1u << 5, // the last link of a neuron too weak is moved to another neuron instead of being kept

    /* paths */
    KEEP_PATHS = 1u << 6, // a link is never removed when it takes the last path of an input to an output away
    INPUT_PATHS = 1u << 7, // every input keeps a path to at least one output (init + never cut by the training)
    INPUT_OUT_ONLY = 1u << 8, // only out links for the inputs
    OUTPUT_IN_ONLY = 1u << 9, // only in links for the outputs

    /* learning */
    CLAMP_WEIGHTS = 1u << 10, // |weight| and |bias| kept under weightMax
    DIRECT_OUTPUT = 1u << 11, // the score reaches the sum of an output as it is (a saturated output still learns)

    /* neurons */
    REVIVE = 1u << 12, // the neurons stuck at 0 or 1 get another chance after each batch
    GROW = 1u << 13, // grow() / prune() of neurons allowed during the training
};

/* hook events */
// Events of a training a hook asks for (ITrainHook::events), the train skips the call of the others (no virtual call)
enum TrainEvent: std::uint32_t {
    ON_START = 1u << 0,
    ON_STOP = 1u << 1,
    ON_EPOCH = 1u << 2,
    ON_BATCH = 1u << 3, // can cost a lot (once per batch)
    ON_SAMPLE = 1u << 4, // can cost a lot (once per sample)
    ON_BEST = 1u << 5,
    ON_ROLLBACK = 1u << 6,
};

//----------------------------------------------------------------//
/* GLOBAL */

// Presets of brain::Flag (brain::BrainOptions::flags)
namespace flags {
    inline constexpr std::uint32_t NONE = 0;
    inline constexpr std::uint32_t DEFAULT = brain::Flag::RANDOM_LINKS; // flags of a new brain network
    inline constexpr std::uint32_t ALL = (brain::Flag::GROW << 1) - 1; // every flag
}

// Presets of brain::TrainEvent (brain::ITrainHook::events)
namespace events {
    inline constexpr std::uint32_t NONE = 0;
    inline constexpr std::uint32_t DEFAULT = brain::TrainEvent::ON_START | brain::TrainEvent::ON_STOP | brain::TrainEvent::ON_EPOCH | brain::TrainEvent::ON_BEST | brain::TrainEvent::ON_ROLLBACK; // every event except the costly ones (batch / sample)
    inline constexpr std::uint32_t ALL = (brain::TrainEvent::ON_ROLLBACK << 1) - 1; // every event
}

} // namespace end
#endif /* BRAIN_NETWORKDEFINE_H */
