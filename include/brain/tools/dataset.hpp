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
    #include <utils/utils.hpp>         // _hot, _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::TextMode
    #include "../NetworkType.hpp"      // brain::IType, brain::FType, brain::Values, brain::Dataset
    #include <algorithm>               // std::clamp
    #include <utility>                 // std::pair
    #include <cstring>                 // std::memcpy
    #include <string>                  // std::string
    #include <cmath>                   // std::isfinite, std::lround

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* conversion */
template<brain::TextMode mode = brain::TextMode::Normal>
_hot _nodiscard inline brain::Values text_to_input(const std::string& text) // Normal: one value per character (byte / 255), Compact: the bytes copied as is in the values (4 per value, can be not what you want)
{
    if constexpr (mode == brain::TextMode::Normal) {
        brain::Values input(text.size());

        for (std::size_t i = 0; i < text.size(); ++i)
            input[i] = static_cast<brain::FType>(static_cast<unsigned char>(text[i])) / static_cast<brain::FType>(255.0);
        return input;
    } else {
        brain::Values input((text.size() + sizeof(brain::FType) - 1) / sizeof(brain::FType));

        std::memcpy(input.data(), text.data(), text.size());
        return input;
    }
}

template<brain::TextMode mode = brain::TextMode::Normal, bool round = false>
_hot _nodiscard inline std::string input_to_text(const brain::Values& input) // doesn't ensure the validity of the output, round: Compact only, removes the '\0' of the end (the padding of the last value)
{
    if constexpr (mode == brain::TextMode::Normal) {
        std::string text(input.size(), '\0');

        for (std::size_t i = 0; i < input.size(); ++i)
            text[i] = std::isfinite(input[i]) ? static_cast<char>(static_cast<unsigned char>(std::clamp<long>(std::lround(input[i] * static_cast<brain::FType>(255.0)), 0, 255))) : '\0';
        return text;
    } else {
        std::string text(input.size() * sizeof(brain::FType), '\0');

        std::memcpy(text.data(), input.data(), input.size() * sizeof(brain::FType));
        if constexpr (round) {
            while (!text.empty() && text.back() == '\0') text.pop_back();
        }
        return text;
    }
}

/* dataset */
_cold void shuffle(brain::Dataset& inputs, const brain::IType seed = 0); // shuffle the samples order (seed == 0 -> random seed)
_cold _nodiscard std::pair<brain::Dataset, brain::Dataset> rand_sets(const brain::Dataset& inputs, const brain::FType percentage, const brain::IType seed = 0); // <training, testing> sets picked randomly (percentage = part of the whole data kept for the training, seed == 0 -> random seed)

} // namespace end
#endif /* DATASET_H */
