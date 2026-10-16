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
##  @file NetworkDefine.hpp

File Description:
##  Enums of the neural network: kind of network, internal activation & loss functions
\**************************************************************/

#ifndef NETWORKDEFINE_H
    #define NETWORKDEFINE_H

namespace brain { // namespace start
//----------------------------------------------------------------//
/* ENUM */

// Kind of network stored behind a brain::NeuralNetwork
enum class NetworkType {
    Layered, // classic layers of neurons (brain::LayerNetwork)
    Brain,   // neurons placed in a 2d space (brain::BrainNetwork)
    Banana,  // pre made layer / neuron numbers shaped like a banana (don't ask why...)
};

// Internal activation functions (more optimized than an external one)
enum class ActivationFunction {
    None, // raw values
    Sigmoid,
    Softmax,
    Linear,
    ReLU,
};

// Internal loss functions (more optimized than an external one)
enum class LossFunction {
    None, // no loss computed
    MSE,  // mean squared error
    CrossEntropy,
};

} // namespace end
#endif /* NETWORKDEFINE_H */
