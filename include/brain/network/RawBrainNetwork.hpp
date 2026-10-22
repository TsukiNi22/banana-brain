/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 22/10/2026 by @author Tsukini

File Name:
##  @file RawBrainNetwork.hpp

File Description:
##  Raw custom network (more like a brain): neurons placed in a 2d space linked in both directions, executes and trains (brain::BrainNetwork is its reduced version)
\**************************************************************/

#ifndef RAWBRAINNETWORK_H
    #define RAWBRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _System
    #include <utils/utils.hpp>      // _cold, _nodiscard, utils::system::HybridMemory
    #include "../NetworkDefine.hpp" // BRAIN_LINKS_CAP_PERCENTAGE, BRAIN_LINKS_CAP_COMPUTING
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::RawBrainNeuron, brain::RawBrainLink, brain::LinksCapFn
    #include "BrainNetwork.hpp"     // brain::BrainNetwork
    #include <utility>              // std::pair
    #include <memory>               // std::unique_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// The reduced network (the memories of brain::BrainNetwork) is the execution copy of this one: kept up to date by sync_() (deserialize, reduce)
class RawBrainNetwork: public brain::BrainNetwork {
    private:
        utils::system::HybridMemory<brain::RawBrainNeuron> _rawNeurons; // every neuron of the 2d space (gpu when available)
        utils::system::HybridMemory<brain::RawBrainLink> _rawLinks; // [neuron * _cap + slot] the links of every neuron, inward and outward (a free slot is all at 0)
        std::pair<brain::FType, brain::FType> _size = {1.0, 1.0}; // <width, height> of the 2d space (could be applied in 3d in the future)
        std::pair<brain::FType, brain::LinksCapFn> _limit; // <percentage, hook> limits of linking: the smallest of the two is the number of links of a neuron
        bool _customLimit = false; // true: the hook was replaced (can't be saved)

        // ---------- Pre-Function -------- //
        void init_(void) final; // compute the number of links of each neuron (_cap)
        brain::UIType linksCap_(void) const; // number of links allowed for a neuron: min(percentage * neurons, hook(neurons)), between 1 and neurons - 1 (0 with a single neuron)
        void sync_(void); // rebuild the reduced memories (biases and inward links) from the raw ones

        // ------------ Function ---------- //
        _cold _nodiscard static inline brain::UIType defaultCap_(const brain::UIType n) {return static_cast<brain::UIType>(BRAIN_LINKS_CAP_COMPUTING(n));}; // the default hook

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const final;
        void deserialize(const brain::Bytes& data) final;

        /* execution */
        using brain::BrainNetwork::apply;
        brain::Values apply(void) final;
        void backpropagation(const brain::FType coef) final;
        void backpropagation(const brain::Values& coefs) final;

        /* setup */
        void weightInit(const brain::FType fallback = 0.01) final;
        void biasInit(const brain::FType fallback = 0.01) final;

        /* setter (links) */
        void linksCap(const brain::FType percentage); // set the maximum allowed links using the percentage of the neurons (0 < percentage <= 1)
        void linksCap(const brain::LinksCapFn& fn); // replace the function to compute the number of allowed links
        void resetLinksCap(void); // reset the links cap used

        /* reduction */
        std::unique_ptr<brain::BrainNetwork> reduce(void); // the reduced network (execution only), from the inward links and the biases

        // ------------ Operator ---------- //
        RawBrainNetwork& operator=(const RawBrainNetwork& other) = delete;
        RawBrainNetwork& operator=(RawBrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        RawBrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size = {1.0, 1.0}, const utils::system::MemoryMode mode = utils::system::MemoryMode::Auto); // size = 2d space of the neurons, calls init_ at the end
        RawBrainNetwork(const RawBrainNetwork& other) = delete;
        RawBrainNetwork(RawBrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~RawBrainNetwork() = default;
};

} // namespace end
#endif /* RAWBRAINNETWORK_H */
