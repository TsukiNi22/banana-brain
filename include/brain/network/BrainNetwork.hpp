/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 10/11/2026 by @author Tsukini

File Name:
##  @file BrainNetwork.hpp

File Description:
##  Reduced custom network (more like a brain): execution only, made by brain::RawBrainNetwork::reduce()
\**************************************************************/

#ifndef BRAINNETWORK_H
    #define BRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::BrainNeuron, brain::BrainLink
    #include "ANetwork.hpp"         // brain::ANetwork
    #include <utility>              // std::pair
    #include <vector>               // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class BrainNetwork: public brain::ANetwork {
    protected:
        std::vector<brain::BrainNeuron> _neurons; // every neuron (its bias)
        std::vector<brain::BrainLink> _links; // the input links of every neuron, _cap slots each
        brain::UIType _neuron = 0; // number of neurons
        brain::UIType _cap = 0; // slots of links of each neuron

        // ---------- Pre-Function -------- //
        void init_(void) override; // create _neuron neurons with _cap free slots of links each

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const override;
        void deserialize(const brain::Bytes& data) override;

        /* execution */
        using brain::ANetwork::apply;
        brain::Values apply(void) override; // execution only: ReadOnlyNetwork
        void backpropagation(const brain::FType coef) override; // execution only: ReadOnlyNetwork
        void backpropagation(const brain::Values& coefs) override; // execution only: ReadOnlyNetwork

        /* setup */
        void weightInit(const brain::FType fallback = 0.01) override; // execution only: ReadOnlyNetwork
        void biasInit(const brain::FType fallback = 0.01) override; // execution only: ReadOnlyNetwork

        /* getter */
        std::pair<brain::UIType, brain::UIType> neurons(void) const final; // <0, total neurons> (no layer)

        // ------------ Operator ---------- //
        BrainNetwork& operator=(const BrainNetwork& other) = delete;
        BrainNetwork& operator=(BrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        BrainNetwork(const brain::UIType neuron, const brain::UIType cap = 0); // cap = slots of links of each neuron, calls init_ at the end
        BrainNetwork(const BrainNetwork& other) = delete;
        BrainNetwork(BrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~BrainNetwork() = default;
};

} // namespace end
#endif /* BRAINNETWORK_H */
