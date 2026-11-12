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
##  @file activation.hpp

File Description:
##  Activation functions of the neurons and their derivatives (from the output), on one value or on the whole vector
\**************************************************************/

#ifndef ACTIVATION_H
    #define ACTIVATION_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _hot, _nodiscard
    #include "../NetworkDefine.hpp"    // BRAIN_DEFAULT_LEAKY_ALPHA
    #include "../NetworkType.hpp"      // brain::FType, brain::Value, brain::Values
    #include <cmath>                   // std::exp

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* value (brain::ActivationFunction) */
_hot _nodiscard inline brain::Value sigmoid(const brain::Value value) {return static_cast<brain::Value>(1.0) / (static_cast<brain::Value>(1.0) + std::exp(-value));}; // 1 / (1 + e^-value)
_hot _nodiscard inline brain::Value linear(const brain::Value value)  {return value;}; // value unchanged
_hot _nodiscard inline brain::Value relu(const brain::Value value)    {return value > 0 ? value : static_cast<brain::Value>(0.0);}; // max(0, value)
_hot _nodiscard brain::Value tanh(const brain::Value value); // (e^value - e^-value) / (e^value + e^-value)
_hot _nodiscard brain::Value leaky_relu(const brain::Value value, const brain::FType alpha = BRAIN_DEFAULT_LEAKY_ALPHA); // value, alpha * value under 0

/* derivative (from the output of the activation: no exp computed again) */
_hot _nodiscard brain::FType sigmoid_d(const brain::Value output); // output * (1 - output)
_hot _nodiscard brain::FType linear_d(const brain::Value output); // 1
_hot _nodiscard brain::FType relu_d(const brain::Value output); // 1 over 0, else 0
_hot _nodiscard brain::FType tanh_d(const brain::Value output); // 1 - output^2
_hot _nodiscard brain::FType leaky_relu_d(const brain::Value output, const brain::FType alpha = BRAIN_DEFAULT_LEAKY_ALPHA); // 1 over 0, else alpha

/* vector */
_hot _nodiscard brain::Values softmax(const brain::Values& values); // e^value / sum of every e^value (the max is removed first to avoid an overflow)
_hot _nodiscard brain::Values softmax_d(const brain::Values& outputs); // diagonal of the jacobian: output * (1 - output) (with CrossEntropy the train uses output - expected directly)

} // namespace end
#endif /* ACTIVATION_H */
