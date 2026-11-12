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
##  @file space.hpp

File Description:
##  Positions of the brain neurons in a space of n dimensions (same code for every dimension, nothing in 0D)
\**************************************************************/

#ifndef SPACE_H
    #define SPACE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _hot, _cold, _nodiscard
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::RawBrainNeuron, brain::RawBrainNeurons
    #include <cstdint>              // std::uint16_t
    #include <vector>               // std::vector
    #include <span>                 // std::span

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* position (space: one value = the same size on every axis, else one per axis) */
_cold _nodiscard std::vector<brain::FType> rand_pos(const std::vector<brain::FType>& space, const std::uint16_t dimension); // random position in the space (empty in 0D)
_hot _nodiscard bool inside(const std::vector<brain::FType>& pos, const std::vector<brain::FType>& space); // every axis of the position in [0, size of the axis]

/* distance */
_hot _nodiscard brain::FType distance(const std::vector<brain::FType>& a, const std::vector<brain::FType>& b); // euclidean on n axes (0 in 0D)
// The distances are computed again at each call (init / relink / grow): nothing stored
_cold _nodiscard std::vector<brain::UIType> closest(std::span<const brain::RawBrainNeuron> neurons, const brain::UIType from, const brain::UIType count); // the ids of the count closest neurons of the neuron from (never itself), closest first (count above the other neurons: all of them)

/* order */
_cold void sort(brain::RawBrainNeurons& neurons); // on the first axis (LEFT_TO_RIGHT), nothing in 0D (creation order)

} // namespace end
#endif /* SPACE_H */
