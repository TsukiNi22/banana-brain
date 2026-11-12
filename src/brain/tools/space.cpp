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
##  @file space.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/tools/space.hpp"
#include <algorithm>
#include <utility>
#include <cstddef>
#include <string>

_cold std::vector<brain::FType> brain::tools::rand_pos(const std::vector<brain::FType>& space, const std::uint16_t dimension)
{
    /* Nothing */
    return {};
}

_hot bool brain::tools::inside(const std::vector<brain::FType>& pos, const std::vector<brain::FType>& space)
{
    /* Nothing */
    return false;
}

_hot brain::FType brain::tools::distance(const std::vector<brain::FType>& a, const std::vector<brain::FType>& b)
{
    /* Nothing */
    return {};
}

_cold std::vector<brain::UIType> brain::tools::closest(std::span<const brain::RawBrainNeuron> neurons, const brain::UIType from, const brain::UIType count)
{
    const brain::UIType size = neurons.size();
    std::vector<std::pair<brain::FType, brain::UIType>> distances; // <squared distance, id>
    std::vector<brain::UIType> nearest;
    brain::UIType kept;

    // check inputs
    if (from >= size) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfBounds, "the neuron " + std::to_string(from) + " is outside of the " + std::to_string(size) + " neurons");
    }

    // Squared distance to every other neuron on the common axes (squared: only the order matters)
    distances.reserve(size - 1);
    for (brain::UIType other = 0; other < size; ++other) {
        brain::FType squared = 0.0;

        if (other == from) continue;
        for (std::size_t axis = 0; axis < neurons[from].pos.size() && axis < neurons[other].pos.size(); ++axis)
            squared += (neurons[from].pos[axis] - neurons[other].pos[axis]) * (neurons[from].pos[axis] - neurons[other].pos[axis]);
        distances.push_back({squared, other});
    }

    // Closest first (the id breaks the ties: always the same order)
    kept = std::min(count, size - 1);
    std::partial_sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(kept), distances.end());
    nearest.reserve(kept);
    for (brain::UIType i = 0; i < kept; ++i)
        nearest.push_back(distances[i].second);
    return nearest;
}

_cold void brain::tools::sort(brain::RawBrainNeurons& neurons)
{
    /* Nothing */
}
