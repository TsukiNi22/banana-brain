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
##  @file RawBrainNetwork.hpp

File Description:
##  Raw custom network (more like a brain): neurons placed in a 2d space, executes and trains (reduced in a brain::BrainNetwork)
\**************************************************************/

#ifndef RAWBRAINNETWORK_H
    #define RAWBRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::RawBrainNeuron, brain::RawBrainLink
    #include "BrainNetwork.hpp"     // brain::BrainNetwork
    #include <utility>              // std::pair
    #include <memory>               // std::unique_ptr
    #include <vector>               // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class RawBrainNetwork: public brain::BrainNetwork {
    private:
        std::vector<brain::RawBrainNeuron> _rawNeurons; // every neuron of the 2d space
        std::vector<brain::RawBrainLink> _rawLinks; // the links of every neuron (inward and their clones), _cap slots each
        std::pair<brain::FType, brain::FType> _space = {1.0, 1.0}; // <width, height> of the 2d space

        // ---------- Pre-Function -------- //
        void init_(void) final; // place _neuron neurons in the _space and link them

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const final;
        void deserialize(const brain::Bytes& data) final;

        /* execution */
        using brain::BrainNetwork::apply;
        brain::Values apply(void) final;
        void backpropagation(const brain::FType coef) final; // a score (positive = good, the links that made the output are strengthened, negative = weakened)
        void backpropagation(const brain::Values& coefs) final;

        /* setup */
        void weightInit(const brain::FType fallback = 0.01) final;
        void biasInit(const brain::FType fallback = 0.01) final;

        /* reduce */
        std::unique_ptr<brain::BrainNetwork> reduce(void) const; // the reduced network (execution only), from the inward links and the biases

        // ------------ Operator ---------- //
        RawBrainNetwork& operator=(const RawBrainNetwork& other) = delete;
        RawBrainNetwork& operator=(RawBrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        RawBrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size = {1.0, 1.0}); // size = 2d space of the neurons, calls init_ at the end
        RawBrainNetwork(const RawBrainNetwork& other) = delete;
        RawBrainNetwork(RawBrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~RawBrainNetwork() = default;
};

} // namespace end
#endif /* RAWBRAINNETWORK_H */
