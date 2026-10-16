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
##  @file dataset.hpp

File Description:
##  Dataset tools: text conversion, shuffle and random training / testing sets
\**************************************************************/

#ifndef DATASET_H
    #define DATASET_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _hot, _cold, _nodiscard
    #include "../NetworkType.hpp"   // brain::IType, brain::FType, brain::Values, brain::Dataset
    #include <utility>              // std::pair
    #include <string>               // std::string

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* conversion */
_hot _nodiscard brain::Values text_to_input(const std::string& text); // can be not what you want
_hot _nodiscard std::string input_to_text(const brain::Values& input); // doesn't ensure the validity of the output

/* dataset */
_cold void shuffle(brain::Dataset& inputs, const brain::IType seed = 0); // shuffle the samples order (seed == 0 -> random seed)
_cold _nodiscard std::pair<brain::Dataset, brain::Dataset> rand_sets(const brain::Dataset& inputs, const brain::FType percentage, const brain::IType seed = 0); // <training, testing> sets picked randomly (percentage = part of the whole data kept for the training, seed == 0 -> random seed)

} // namespace end
#endif /* DATASET_H */
