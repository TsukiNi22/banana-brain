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
##  @file RawBrainNetwork.hpp

File Description:
##  Raw custom network (more like a brain): neurons placed in a space of n dimensions, executes and trains (reduced in a brain::BrainNetwork)
\**************************************************************/

#ifndef RAWBRAINNETWORK_H
    #define RAWBRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkDefine.hpp"   // brain::PruneTarget, BRAIN_DEFAULT_FALLBACK, BRAIN_DEFAULT_PRUNE
    #include "../NetworkType.hpp"     // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::Buffer, brain::RawBrainNeurons, brain::RawBrainLink, brain::BrainOptions
    #include "BrainNetwork.hpp"       // brain::BrainNetwork
    #include "INetwork.hpp"           // brain::INetwork
    #include <cstdint>                // std::uint16_t
    #include <memory>                 // std::unique_ptr
    #include <vector>                 // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// The time steps go through _backend->propagate (links), the gradients through _backend->axpy
class RawBrainNetwork: public brain::BrainNetwork {
    private:
        /* structure */
        brain::RawBrainNeurons _rawNeurons; // every neuron of the space, pos.size() == _options.dimension (no space of its own: _options.space of BrainNetwork)
        std::vector<brain::RawBrainLink> _rawLinks; // the links of every neuron (inward and their clones), _cap slots each

        /* training */
        std::vector<brain::Buffer> _gradients; // mini-batch, one per worker of the backend (summed by applyGradients)
        std::vector<brain::UIType> _stuck; // neurons stuck at 0 or 1 (REVIVE)

        // ---------- Pre-Function -------- //
        void init_(void) final; // place _neuron neurons in the space and link them

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const final;
        void deserialize(const brain::Bytes& data) final;

        /* execution */
        using brain::BrainNetwork::apply;
        brain::Values apply(void) final;

        /* training */
        void backpropagation(const brain::FType coef) final; // a score (positive = good, the links that made the output are strengthened, negative = weakened)
        void backpropagation(const brain::Values& coefs) final;
        void applyGradients(void); // mean of the batch through the optimizer, called by the train at the end of each batch

        /* init */
        void weightInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) final;
        void biasInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) final;

        /* options */
        using brain::BrainNetwork::dimension;
        using brain::BrainNetwork::space;
        void dimension(const std::uint16_t dimension) final; // the neurons placed again in the new space (links kept)
        void space(const std::vector<brain::FType>& space) final; // the positions scaled to the new space (links kept)

        /* conversion */
        std::unique_ptr<brain::BrainNetwork> reduce(void) const; // the reduced network (execution only), from the inward links and the biases

        /* shape */
        void relink(void); // links under weightMin moved to another close neuron (a random one in 0D), new link at BRAIN_DEFAULT_NEW_LINK of the limits with LOW_NEW_LINKS
        void revive(void); // the stuck neurons get another chance (REVIVE)
        void grow(const brain::UIType count); // add neurons close to the most used / strongest ones (GROW), 0D: a random position in the creation order, next to the strongest ones
        void prune(const brain::FType influence = BRAIN_DEFAULT_PRUNE, const brain::PruneTarget target = brain::PruneTarget::Both) final; // neurons (GROW) / links

        /* state */
        std::unique_ptr<brain::INetwork> clone(void) const final;
        void reset(void) final;

        // ------------ Operator ---------- //
        RawBrainNetwork& operator=(const RawBrainNetwork& other) = delete;
        RawBrainNetwork& operator=(RawBrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        RawBrainNetwork(const brain::UIType neuron, const brain::BrainOptions& options = {}); // the shape of the space in the options (dimension / space), calls init_ at the end
        RawBrainNetwork(const RawBrainNetwork& other) = delete;
        RawBrainNetwork(RawBrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~RawBrainNetwork() = default;
};

} // namespace end
#endif /* RAWBRAINNETWORK_H */
