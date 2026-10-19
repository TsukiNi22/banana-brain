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
##  @file neuron.hpp

File Description:
##  Tools of the brain neurons: distance ordering of the neurons of a network
\**************************************************************/

#ifndef NEURON_H
    #define NEURON_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _cold
    #include "../NetworkType.hpp"   // brain::UIType, brain::BrainNeuron, brain::BrainDistance
    #include <span>                 // std::span

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* brain neuron */
// The spans are host memory (fetch / store the HybridMemory of the network first on a gpu), the distances of the neuron i are ordered[i * (n - 1), (i + 1) * (n - 1)) with n the number of neurons
_cold void sort(const brain::UIType index, std::span<brain::BrainNeuron> neurons, std::span<brain::BrainDistance> ordered); // fill the distances of the neuron index with every other neuron sorted closest first (the distances are also stored, unsorted, in the other neurons), sets BrainNeuron::ordered and BrainNeuron::orderedCount

} // namespace end
#endif /* NEURON_H */
