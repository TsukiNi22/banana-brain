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
##  @file activation.hpp

File Description:
##  Tools of the internal activation functions: sigmoid, softmax, linear and ReLU
\**************************************************************/

#ifndef ACTIVATION_H
    #define ACTIVATION_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _hot, _nodiscard
    #include "../NetworkType.hpp"   // brain::FType
    #include <cmath>                // std::exp

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* activation (brain::ActivationFunction) */
_hot _nodiscard inline brain::FType sigmoid(const brain::FType value) {return static_cast<brain::FType>(1.0) / (static_cast<brain::FType>(1.0) + std::exp(-value));}; // 1 / (1 + e^-value)
//_hot _nodiscard brain::Values softmax(const brain::Values& values); // e^value / sum of every e^value (the max is removed first to avoid an overflow)
_hot _nodiscard inline brain::FType linear(const brain::FType value) {return value;}; // value unchanged
_hot _nodiscard inline brain::FType relu(const brain::FType value)   {return value > 0 ? value : static_cast<brain::FType>(0.0);}; // max(0, value)

} // namespace end
#endif /* ACTIVATION_H */
