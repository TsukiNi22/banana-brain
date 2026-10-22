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
##  @file BrainNetwork.hpp

File Description:
##  Custom network (more like a brain): neurons placed in a 2d space, linked by ids
\**************************************************************/

#ifndef BRAINNETWORK_H
    #define BRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _System
    #include <utils/utils.hpp>      // utils::system::HybridMemory
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::BrainNeuron, brain::BrainLink, brain::LinksCapFn
    #include "../NetworkDefine.hpp" // BRAIN_LINKS_CAP_PERCENTAGE, BRAIN_LINKS_CAP_COMPUTING
    #include "ANetwork.hpp"         // brain::ANetwork
    #include <utility>              // std::pair

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class BrainNetwork: public brain::ANetwork {
    private:
        utils::system::HybridMemory<brain::BrainNeuron> _neurons; // every neuron of the 2d space (gpu when available)
        utils::system::HybridMemory<brain::BrainLink> _links; // the links of every neuron, from BrainNeuron::incoming (BrainNeuron::incomingCount links)
        std::pair<brain::FType, brain::FType> _size = {1.0, 1.0}; // <width, height> of the 2d space (could be applied in 3d in the future)
        brain::UIType _neuron = 0; // number of neurons
        sd::pair<brain::FType, brain::LinksCapFn> _limit = {BRAIN_LINKS_CAP_PERCENTAGE, [](const brain::UIType n){return BRAIN_LINKS_CAP_COMPUTING(n);}}; // <percentage, hook> limits of linking
        brain::UIType _links = 0; // limit of the number of link allowed for each neuron

        // ---------- Pre-Function -------- //
        void init_(void) final; // place _neuron neurons in the _size space

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const final;
        void deserialize(const brain::Bytes& data) final;

        /* execution */
        brain::Values apply(void) const final;
        void backpropagation(const brain::FType coef) final;
        void backpropagation(const brain::Values& coefs) final;

        /* setup */
        void weightInit(const brain::FType fallback = 0.01) final;
        void biasInit(const brain::FType fallback = 0.01) final;

        // ------------ Function ---------- //
        /* setter (links) */
        _cold inline void linksCap(const brain::FType percentage) {this->_limit.first = percentage;}; // set the maximum allowed links using percentage
        _cold inline void linksCap(const brain::LinksCapFn& fn)   {this->_limit.second = fn;}; // replace the function to compute the number of allowed links
        _cold inline void resetLinksCap(void)                     {this->_limit = {BRAIN_LINKS_CAP_PERCENTAGE, [](const brain::UIType n){return BRAIN_LINKS_CAP_COMPUTING(n);}};}; // reset the links cap used

        /* getter */
        _cold _nodiscard std::pair<brain::UIType, brain::UIType> neurons(void) const final {return {0, this->_neuron};}; // <0, total neurons> (no layer)

        // ------------ Operator ---------- //
        BrainNetwork& operator=(const BrainNetwork& other) = delete;
        BrainNetwork& operator=(BrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        BrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size = {1.0, 1.0}); // size = 2d space of the neurons, calls init_ at the end
        BrainNetwork(const BrainNetwork& other) = delete;
        BrainNetwork(BrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~BrainNetwork() = default;
};

} // namespace end
#endif /* BRAINNETWORK_H */
