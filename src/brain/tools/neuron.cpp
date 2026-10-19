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
#include <utils/utils.hpp>
#include "brain/tools/neuron.hpp"
#include <unordered_set>
#include <algorithm>
#include <vector>

_cold void brain::tools::sort(brain::BrainNeuron& neuron, std::vector<brain::BrainNeuron>& neurons)
{
    std::unordered_set<brain::UIType> known;

    // Ids already linked to the neuron
    for (const std::pair<brain::UIType, brain::FType>& other: neuron.ordered)
        known.insert(other.first);

    // Every other neuron (neuron itself skipped: operator== compares the ids)
    for (brain::BrainNeuron& other: neurons) {
        if (other == neuron || known.contains(other.id)) continue;

        // Squared distance, stored on both sides (the other list is left unsorted)
        const brain::FType distance = neuron | other;
        neuron.ordered.push_back({other.id, distance});
        other.ordered.push_back({neuron.id, distance});
    }

    // Closest first
    std::sort(neuron.ordered.begin(), neuron.ordered.end(), [](const std::pair<brain::UIType, brain::FType>& a, const std::pair<brain::UIType, brain::FType>& b) {return a.second < b.second;});
}
