/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 10/11/2026 by @author Tsukini

File Name:
##  @file NetworkDefine.hpp

File Description:
##  Enums of the neural network: kind of network, internal activation & loss functions
\**************************************************************/

#ifndef BRAIN_NETWORKDEFINE_H
    #define BRAIN_NETWORKDEFINE_H

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

namespace brain { // namespace start
//----------------------------------------------------------------//
/* ENUM */

// Kind of network stored behind a brain::NeuralNetwork
enum class NetworkType {
    Layered, // classic layers of neurons (brain::LayerNetwork)
    Brain, // neurons placed in a 2d space (brain::RawBrainNetwork, reduced in a brain::BrainNetwork)
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
#endif /* BRAIN_NETWORKDEFINE_H */
