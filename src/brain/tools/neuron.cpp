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
#include <cstddef>
#include <string>

_cold void brain::tools::sort(const brain::UIType index, std::span<brain::BrainNeuron> neurons, std::span<brain::BrainDistance> ordered)
{
    const brain::UIType count = neurons.size();

    // check inputs
    if (index >= count) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfBounds, "the neuron " + std::to_string(index) + " is outside of the " + std::to_string(count) + " neurons");
    } else if (ordered.size() < count * (count - 1)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the distances memory should hold every neuron to every other one (" + std::to_string(count * (count - 1)) + ")");
    }

    // Place the list of the neuron
    brain::BrainNeuron& neuron = neurons[index];
    brain::UIType slot = 0;
    neuron.ordered = index * (count - 1);
    neuron.orderedCount = count - 1;

    // Every other neuron: squared distance, stored on both sides (the other list is left unsorted)
    for (brain::UIType other = 0; other < count; ++other) {
        if (other == index) continue;
        const brain::FType distance = neuron | neurons[other];
        neurons[other].ordered = other * (count - 1);
        neurons[other].orderedCount = count - 1;
        ordered[neuron.ordered + slot] = {neurons[other].id, distance};
        ordered[neurons[other].ordered + (index < other ? index : index - 1)] = {neuron.id, distance};
        ++slot;
    }

    // Closest first
    std::sort(ordered.begin() + static_cast<std::ptrdiff_t>(neuron.ordered), ordered.begin() + static_cast<std::ptrdiff_t>(neuron.ordered + neuron.orderedCount), [](const brain::BrainDistance& a, const brain::BrainDistance& b) {return a.distance < b.distance;});
}
