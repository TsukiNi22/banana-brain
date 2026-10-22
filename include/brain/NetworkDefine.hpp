/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 21/10/2026 by @author Tsukini

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
    #include <cmath>    // std::log, std::pow

    //----------------------------------------------------------------//
    /* DEFINE */

    /* file extensions (a different one is accepted with a warning) */
    #define BRAIN_EXTENSION_BRAIN ".bn" // brain network: BRAIN_MAGIC_BRAIN
    #define BRAIN_EXTENSION_LAYER ".ln" // layer network: BRAIN_MAGIC_LAYER
    #define BRAIN_EXTENSION_NETWORK ".net" // any network, the magic bytes tell which one

    /* magic bytes (two first bytes of the serialized data: the common one, then one per kind of network) */
    #define BRAIN_MAGIC 0x22 // common first byte
    #define BRAIN_MAGIC_BRAIN 'b' // 0x62, second byte of a brain network
    #define BRAIN_MAGIC_LAYER 'l' // 0x6C, second byte of a layer network

    /* activation byte (third byte of the serialized data: how the activation function is restored) */
    #define BRAIN_ACTIVATION_NONE 0x00 // no activation function (raw values, default)
    #define BRAIN_ACTIVATION_CUSTOM 0xFF // external function (std::function): can't be saved, to give again after the load
    // 0x01 -> 0xFE: internal mode, the value of brain::ActivationFunction

    /* limits */
    #define BRAIN_LINKS_CAP_PERCENTAGE 0.1 // (default value) percentage of the total number of neurons
    #define BRAIN_LINKS_CAP_COMPUTING(n) (std::log(std::pow(n, 2 * std::log2(n))) / std::log(1.1)) // (default value) computing of the default limits of links: log1.1(x^(2log2(x)))

namespace brain { // namespace start
//----------------------------------------------------------------//
/* ENUM */

// Kind of network stored behind a brain::NeuralNetwork
enum class NetworkType {
    Layered, // classic layers of neurons (brain::LayerNetwork)
    Brain, // neurons placed in a 2d space (brain::BrainNetwork)
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
