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
##  @file loss.hpp

File Description:
##  Loss functions of the outputs and their derivatives, on one value or on the whole vector
\**************************************************************/

#ifndef LOSS_H
    #define LOSS_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _hot, _nodiscard
    #include "../NetworkDefine.hpp"    // BRAIN_DEFAULT_HUBER_DELTA
    #include "../NetworkType.hpp"      // brain::FType, brain::Value, brain::Values, brain::LossFn

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* value (brain::LossFunction) */
_hot _nodiscard brain::FType mse(const brain::Value output, const brain::Value expected); // mean squared error of one output: (output - expected)^2
_hot _nodiscard brain::FType cross_entropy(const brain::Value output, const brain::Value expected); // binary cross entropy of one output (output clamped away from 0 and 1 to avoid log(0))
_hot _nodiscard brain::FType mae(const brain::Value output, const brain::Value expected); // mean absolute error of one output: |output - expected|
_hot _nodiscard brain::FType huber(const brain::Value output, const brain::Value expected, const brain::FType delta = BRAIN_DEFAULT_HUBER_DELTA); // squared under delta, linear over it

/* derivative (by the output) */
_hot _nodiscard brain::FType mse_d(const brain::Value output, const brain::Value expected); // 2 * (output - expected)
_hot _nodiscard brain::FType cross_entropy_d(const brain::Value output, const brain::Value expected); // (output - expected) / (output * (1 - output)), output clamped
_hot _nodiscard brain::FType mae_d(const brain::Value output, const brain::Value expected); // sign(output - expected)
_hot _nodiscard brain::FType huber_d(const brain::Value output, const brain::Value expected, const brain::FType delta = BRAIN_DEFAULT_HUBER_DELTA); // output - expected under delta, delta * sign over it

/* vector */
_hot _nodiscard brain::FType mean(const brain::LossFn& fn, const brain::Values& outputs, const brain::Values& expected); // mean loss of the outputs
_hot _nodiscard brain::Values each(const brain::LossFn& fn, const brain::Values& outputs, const brain::Values& expected); // loss of each output (vector mode)

} // namespace end
#endif /* LOSS_H */
