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
    #include <vector>               // std::vector

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
        std::pair<brain::FType, brain::StepsCapFn> _stepsLimit; // <percentage, hook> limits of the time steps: the smallest of the two is the number of steps of the execution
        bool _customSteps = false; // true: the steps hook was replaced (can't be saved)
        std::pair<brain::UIType, brain::UIType> _placed = {0, 0}; // number of inputs and outputs the neurons were placed for
        std::vector<brain::Values> _states; // training: the state of every neuron at each time step of the last apply() (the first one is the start)

        // ---------- Pre-Function -------- //
        void init_(void) final; // set the memories up: the input neurons on the left edge and the output ones on the right edge (evenly spread), the others placed randomly, each one linked to its nearest neurons (weights at 1.0, bias at 0.0)
        void resized_(void) final; // the first neurons are the inputs and the last ones the outputs: set the neurons up again when their number changes
        brain::UIType linksCap_(void) const; // number of links allowed for a neuron: min(percentage * neurons, hook(neurons)), between 1 and neurons - 1 (0 with a single neuron)
        brain::UIType stepsCap_(void) const; // time steps of the execution: min(percentage * neurons, hook(neurons)), at least 1
        void sync_(void); // rebuild the reduced memories (biases and inward links) from the raw ones

        // ------------ Function ---------- //
        _cold _nodiscard static inline brain::UIType defaultCap_(const brain::UIType n) {return static_cast<brain::UIType>(BRAIN_LINKS_CAP_COMPUTING(n));}; // the default hook
        _cold _nodiscard static inline brain::UIType defaultSteps_(const brain::UIType n) {return static_cast<brain::UIType>(BRAIN_STEPS_CAP_COMPUTING(n));}; // the default hook of the time steps

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
        void linksCap(const brain::FType percentage); // set the maximum allowed links using the percentage of the neurons (0 < percentage <= 1), sets the neurons and the links up again
        void linksCap(const brain::LinksCapFn& fn); // replace the function to compute the number of allowed links, sets the neurons and the links up again
        void resetLinksCap(void); // reset the links cap used
        void stepsCap(const brain::FType percentage); // set the maximum time steps using the percentage of the neurons (0 < percentage <= 1)
        void stepsCap(const brain::StepsCapFn& fn); // replace the function to compute the time steps
        void resetStepsCap(void); // reset the steps cap used

        /* reduction */
        std::unique_ptr<brain::BrainNetwork> reduce(void); // the reduced network (execution only), from the inward links and the biases

        // ------------ Function ---------- //
        _cold _nodiscard std::size_t memory(void) const final {return this->brain::BrainNetwork::memory() + this->_rawNeurons.bytes() + this->_rawLinks.bytes();}; // the execution copy and the raw memories

        /* getter (links, steps) */
        _cold _nodiscard inline const std::pair<brain::FType, brain::LinksCapFn>& linksCap(void) const {return this->_limit;}; // <percentage, hook> of the links limit
        _cold _nodiscard inline const std::pair<brain::FType, brain::StepsCapFn>& stepsCap(void) const {return this->_stepsLimit;}; // <percentage, hook> of the time steps limit

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
