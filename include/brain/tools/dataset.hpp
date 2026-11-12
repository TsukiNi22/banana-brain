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
##  @file dataset.hpp

File Description:
##  Tools of the datasets: order, split, batches, scale, classes and csv files
\**************************************************************/

#ifndef DATASET_H
    #define DATASET_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _hot, _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::NormalizeMode
    #include "../NetworkType.hpp"      // brain::IType, brain::UIType, brain::FType, brain::Values, brain::Dataset, brain::Normalization
    #include <filesystem>              // std::filesystem::path
    #include <utility>                 // std::pair
    #include <vector>                  // std::vector

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* order */
_cold void shuffle(brain::Dataset& inputs, const brain::IType seed = 0); // shuffle the samples order (seed == 0 -> random seed)
_cold void shuffle(brain::Dataset& inputs, brain::Dataset& expected, const brain::IType seed = 0); // same order on both (the pairs are kept)
_cold _nodiscard std::pair<brain::Dataset, brain::Dataset> rand_sets(const brain::Dataset& inputs, const brain::FType percentage, const brain::IType seed = 0); // <training, testing> sets picked randomly (percentage = part of the whole data kept for the training, seed == 0 -> random seed)
_cold _nodiscard std::pair<std::pair<brain::Dataset, brain::Dataset>, std::pair<brain::Dataset, brain::Dataset>> rand_sets(const brain::Dataset& inputs, const brain::Dataset& expected, const brain::FType percentage, const brain::IType seed = 0); // <<training inputs, training expected>, <testing inputs, testing expected>> (the pairs are kept)
_cold _nodiscard std::vector<brain::Dataset> batches(const brain::Dataset& data, const brain::UIType size); // copies (each batch owns its samples), the last one can be smaller

/* scale */
template<brain::NormalizeMode mode = brain::NormalizeMode::MinMax>
_cold _nodiscard inline brain::Normalization normalize(brain::Dataset& data) // by column, returns the params (the same params for another set / denormalize)
{
    /* Nothing */
    return {};
}

_cold void normalize(brain::Dataset& data, const brain::Normalization& params); // same params on another set (testing / exec)
_hot _nodiscard brain::Values denormalize(const brain::Values& values, const brain::Normalization& params); // outputs back to their real scale

/* classes */
_hot _nodiscard brain::Values one_hot(const brain::UIType index, const brain::UIType count); // count values at 0, the index one at 1
_cold _nodiscard brain::Dataset one_hot(const std::vector<brain::UIType>& classes, const brain::UIType count); // one sample per class
_hot _nodiscard brain::UIType argmax(const brain::Values& values); // index of the highest value (the first one on a tie)

/* files */
_cold _nodiscard std::pair<brain::Dataset, brain::Dataset> from_csv(const std::filesystem::path& path, const std::vector<brain::UIType>& inputs_columns, const std::vector<brain::UIType>& outputs_columns, const char separator = ',', const bool header = true); // <inputs, expected> from the columns picked
_cold void to_csv(const std::filesystem::path& path, const brain::Dataset& inputs, const brain::Dataset& outputs, const char separator = ','); // one line per sample: the inputs then the outputs

} // namespace end
#endif /* DATASET_H */
