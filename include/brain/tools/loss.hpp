/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 19/10/2026 by @author Tsukini

File Name:
##  @file loss.hpp

File Description:
##  Tools of the internal loss functions: mean squared error and cross entropy
\**************************************************************/

#ifndef LOSS_H
    #define LOSS_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _hot, _nodiscard
    #include "../NetworkType.hpp"   // brain::FType

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* loss (brain::LossFunction) */
_hot _nodiscard brain::FType mse(const brain::FType output, const brain::FType expected); // mean squared error of one output: (output - expected)^2
_hot _nodiscard brain::FType cross_entropy(const brain::FType output, const brain::FType expected); // binary cross entropy of one output (output clamped away from 0 and 1 to avoid log(0))

} // namespace end
#endif /* LOSS_H */
