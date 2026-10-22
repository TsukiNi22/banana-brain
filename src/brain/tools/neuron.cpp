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
##  @file neuron.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/tools/neuron.hpp"
#include <algorithm>
#include <utility>
#include <cstddef>
#include <string>

_cold std::vector<brain::UIType> brain::tools::sort(const brain::UIType index, std::span<const brain::RawBrainNeuron> neurons, const brain::UIType cap)
{
    const brain::UIType count = neurons.size();
    std::vector<std::pair<brain::FType, brain::UIType>> distances; // <squared distance, id>
    std::vector<brain::UIType> nearest;
    brain::UIType kept;

    // check inputs
    if (index >= count) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfBounds, "the neuron " + std::to_string(index) + " is outside of the " + std::to_string(count) + " neurons");
    }

    // Squared distance to every other neuron (squared: only the order matters)
    distances.reserve(count - 1);
    for (brain::UIType other = 0; other < count; ++other) {
        if (other != index) distances.push_back({neurons[index] | neurons[other], other});
    }

    // Closest first (the id breaks the ties: always the same order)
    kept = std::min(cap, count - 1);
    std::partial_sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(kept), distances.end());
    nearest.reserve(kept);
    for (brain::UIType i = 0; i < kept; ++i)
        nearest.push_back(distances[i].second);
    return nearest;
}
