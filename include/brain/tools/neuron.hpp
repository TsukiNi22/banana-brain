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
    #include "../NetworkType.hpp"   // brain::UIType, brain::RawBrainNeuron
    #include <vector>               // std::vector
    #include <span>                 // std::span

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* brain neuron */
// The distances are computed again at each call (done once, at the init of the network): nothing stored
_cold _nodiscard std::vector<brain::UIType> sort(const brain::UIType index, std::span<const brain::RawBrainNeuron> neurons, const brain::UIType cap); // the ids of the cap nearest neurons of the neuron index (never itself), closest first (cap above the other neurons: all of them)

} // namespace end
#endif /* NEURON_H */
