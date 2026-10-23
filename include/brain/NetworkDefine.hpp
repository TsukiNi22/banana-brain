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
##  @file NetworkDefine.hpp

File Description:
##  Enums of the neural network: kind of network, internal activation & loss functions
\**************************************************************/

#ifndef NETWORKDEFINE_H
    #define NETWORKDEFINE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
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
    #define BRAIN_LINKS_CAP_PERCENTAGE 0.1 // (default value) percentage of the total number of neurons
    #define BRAIN_LINKS_CAP_COMPUTING(n) (2.0 * std::log2(static_cast<double>(n)) * std::log(static_cast<double>(n)) / std::log(1.1)) // (default value) computing of the default limits of links: log1.1(x^(2log2(x))) written without the pow (no overflow on a big network)

    #define BRAIN_SIGNAL_THRESHOLD 0.0 // (default value, placeholder to give) a signal under it (absolute value) is cancelled in a brain network: 0 = nothing is cancelled
    #define BRAIN_WEIGHT_MIN 0.01 // (default value) a raw brain network removes a link (and its clone) when the absolute value of its weight falls to it (or under), unless it is the last link the neuron needs
    #define BRAIN_WEIGHT_MAX 2.0 // (default value) a raw brain network links the neuron to its nearest free neuron (within the cap) when the absolute value of the weight of one of its links rises to it (or over)
    #define BRAIN_STEPS_CAP_PERCENTAGE 0.25 // (default value) percentage of the total number of neurons
    #define BRAIN_STEPS_CAP_COMPUTING(n) (4.0 * std::log2(static_cast<double>(n))) // (default value) computing of the default number of time steps of a brain network from the number of neurons n: a signal crosses the network in about log2(n) hops, twice for the loops

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

} // namespace end
#endif /* NETWORKDEFINE_H */
