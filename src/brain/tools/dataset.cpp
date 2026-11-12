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
##  @file dataset.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/tools/dataset.hpp"
#include <filesystem>
#include <algorithm>
#include <utility>
#include <cstdint>
#include <random>
#include <vector>

_cold void brain::tools::shuffle(brain::Dataset& inputs, const brain::IType seed)
{
    std::random_device device;
    std::mt19937_64 generator(seed == 0 ? static_cast<std::uint64_t>(device()) : static_cast<std::uint64_t>(seed));

    std::shuffle(inputs.begin(), inputs.end(), generator);
}

_cold void brain::tools::shuffle(brain::Dataset& inputs, brain::Dataset& expected, const brain::IType seed)
{
    /* Nothing */
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

_cold std::pair<std::pair<brain::Dataset, brain::Dataset>, std::pair<brain::Dataset, brain::Dataset>> brain::tools::rand_sets(const brain::Dataset& inputs, const brain::Dataset& expected, const brain::FType percentage, const brain::IType seed)
{
    /* Nothing */
    return {};
}

_cold std::vector<brain::Dataset> brain::tools::batches(const brain::Dataset& data, const brain::UIType size)
{
    /* Nothing */
    return {};
}

_cold void brain::tools::normalize(brain::Dataset& data, const brain::Normalization& params)
{
    /* Nothing */
}

_hot brain::Values brain::tools::denormalize(const brain::Values& values, const brain::Normalization& params)
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::tools::one_hot(const brain::UIType index, const brain::UIType count)
{
    /* Nothing */
    return {};
}

_cold brain::Dataset brain::tools::one_hot(const std::vector<brain::UIType>& classes, const brain::UIType count)
{
    /* Nothing */
    return {};
}

_hot brain::UIType brain::tools::argmax(const brain::Values& values)
{
    /* Nothing */
    return {};
}

_cold std::pair<brain::Dataset, brain::Dataset> brain::tools::from_csv(const std::filesystem::path& path, const std::vector<brain::UIType>& inputs_columns, const std::vector<brain::UIType>& outputs_columns, const char separator, const bool header)
{
    /* Nothing */
    return {};
}

_cold void brain::tools::to_csv(const std::filesystem::path& path, const brain::Dataset& inputs, const brain::Dataset& outputs, const char separator)
{
    /* Nothing */
}
