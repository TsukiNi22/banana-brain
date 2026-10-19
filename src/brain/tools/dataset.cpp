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
##  @file dataset.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/tools/dataset.hpp"
#include <algorithm>
#include <utility>
#include <random>
#include <string>

_hot brain::Values brain::tools::text_to_input(const std::string& text)
{
    /* Nothing */
    return {};
}

_hot std::string brain::tools::input_to_text(const brain::Values& input)
{
    /* Nothing */
    return {};
}

_cold void brain::tools::shuffle(brain::Dataset& inputs, const brain::IType seed)
{
    std::random_device device;
    std::mt19937_64 generator(seed == 0 ? static_cast<std::uint64_t>(device()) : static_cast<std::uint64_t>(seed));

    std::shuffle(inputs.begin(), inputs.end(), generator);
}

_cold std::pair<brain::Dataset, brain::Dataset> brain::tools::rand_sets(const brain::Dataset& inputs, const brain::FType percentage, const brain::IType seed)
{
    brain::Dataset shuffled = inputs;
    std::size_t split;

    // check inputs
    if (percentage < 0 || percentage > 1) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the percentage of the training set should be between 0 and 1");
    }

    // pick randomly, then split
    brain::tools::shuffle(shuffled, seed);
    split = static_cast<std::size_t>(static_cast<brain::FType>(shuffled.size()) * percentage);
    return {brain::Dataset(shuffled.begin(), shuffled.begin() + static_cast<std::ptrdiff_t>(split)), brain::Dataset(shuffled.begin() + static_cast<std::ptrdiff_t>(split), shuffled.end())};
}
