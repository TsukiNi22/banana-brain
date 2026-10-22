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
    #include <cstring>              // std::memcpy
    #include <string>               // std::string

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* conversion */
_hot _nodiscard inline brain::Values text_to_input(const std::string& s) // can be not what you want
{
    brain::Values input((s.size() + sizeof(brain::FType) - 1) / sizeof(brain::FType));
    std::memcpy(input.data(), s.data(), s.size());
    return input;
}

template<bool round = false>
_hot _nodiscard inline std::string input_to_text(const brain::Values& input) // doesn't ensure the validity of the output
{
    std::string s(input.size() * sizeof(brain::FType), '\0');
    std::memcpy(s.data(), input.data(), input.size() * sizeof(brain::FType));
    if constexpr (round) {
        while (!s.empty() && s.back() == '\0') s.pop_back();
    }
    return s;
}

/* dataset */
_cold void shuffle(brain::Dataset& inputs, const brain::IType seed = 0); // shuffle the samples order (seed == 0 -> random seed)
_cold _nodiscard std::pair<brain::Dataset, brain::Dataset> rand_sets(const brain::Dataset& inputs, const brain::FType percentage, const brain::IType seed = 0); // <training, testing> sets picked randomly (percentage = part of the whole data kept for the training, seed == 0 -> random seed)

} // namespace end
#endif /* DATASET_H */
