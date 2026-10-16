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
#include <vector>

/* brain neuron */
_cold void brain::tools::sort(brain::BrainNeuron& neuron, std::vector<brain::BrainNeuron>& neurons)
{
    /* Nothing */
    // 1. For every other neuron of neurons (skip neuron itself: operator== compares the ids)
    //    a. If the squared distance between the two is not already computed (no <other.id, distance> in neuron.ordered yet)
    //       - compute it from the positions: neuron | other (squared, no sqrt: only the order matters)
    //       - store <other.id, distance> in neuron.ordered
    //       - store <neuron.id, distance> in other.ordered too (symmetric, the other list is left unsorted)
    // 2. Sort neuron.ordered by distance ascending (closest first)
}
