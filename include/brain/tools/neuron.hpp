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
    #include "../NetworkType.hpp"   // brain::BrainNeuron
    #include <vector>               // std::vector

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* brain neuron */
_cold void sort(brain::BrainNeuron& neuron, std::vector<brain::BrainNeuron>& neurons); // fill neuron.ordered with every other neuron <id, squared distance> sorted closest first (the distances are also stored, unsorted, in the other neurons)

} // namespace end
#endif /* NEURON_H */
