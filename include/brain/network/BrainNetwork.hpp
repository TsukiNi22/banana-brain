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
##  Reduced custom network (more like a brain), execution only: the bias of the neurons and their input links, the training is done by brain::RawBrainNetwork
\**************************************************************/

#ifndef BRAINNETWORK_H
    #define BRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _System
    #define _Exception
    #include <utils/utils.hpp>      // _cold, _nodiscard, utils::system::HybridMemory
    #include "../NetworkDefine.hpp" // BRAIN_MAGIC_BRAIN
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::BrainNeuron, brain::BrainLink
    #include "ANetwork.hpp"         // brain::ANetwork
    #include <utility>              // std::pair

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// The smallest network to execute a trained brain network: no position, no link for the backpropagation (every training call throws ReadOnlyNetwork)
class BrainNetwork: public brain::ANetwork {
    protected:
        utils::system::HybridMemory<brain::BrainNeuron> _neurons; // [neuron] (gpu when available)
        utils::system::HybridMemory<brain::BrainLink> _links; // [neuron * _cap + slot] the input links of every neuron (a free slot is all at 0)
        brain::UIType _neuron = 0; // number of neurons
        brain::UIType _cap = 0; // slots of links of each neuron (the limit of the links of a neuron)

        // ---------- Pre-Function -------- //
        void init_(void) override; // nothing to place, the neurons and the links come from deserialize
        brain::Values compute_(const brain::Values& inputs) const override;
        void serializeBody_(brain::Bytes& out) const; // the neurons and the links
        void deserializeBody_(const brain::Bytes& data, brain::UIType offset); // read the neurons and the links, the end of the data included (nothing is edited on an invalid data)

        // ---------- Constructor --------- //
        BrainNetwork(const brain::UIType neuron, const brain::UIType cap, const utils::system::MemoryMode mode, const bool setup); // setup: calls init_ at the end (false: the child does it)

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const override;

        /* execution */
        using brain::ANetwork::apply;

        // ------------ Function ---------- //
        /* serialization */
        _cold void deserialize(const brain::Bytes& data) override {this->deserializeBody_(data, this->deserializeHeader_(data, static_cast<std::uint8_t>(BRAIN_MAGIC_BRAIN)));};

        /* execution (the training is refused) */
        _hot brain::Values apply(void) override                            {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork
        _hot void backpropagation(const brain::FType coef) override        {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork
        _hot void backpropagation(const brain::Values& coefs) override     {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork

        /* setup (the training is refused) */
        _cold void weightInit(const brain::FType fallback = 0.01) override {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork
        _cold void biasInit(const brain::FType fallback = 0.01) override   {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork

        /* getter */
        _cold _nodiscard inline brain::UIType links(void) const                            {return this->_cap;}; // slots of links of each neuron
        _cold _nodiscard std::pair<brain::UIType, brain::UIType> neurons(void) const final {return {0, this->_neuron};}; // <0, total neurons> (no layer)

        // ------------ Operator ---------- //
        BrainNetwork& operator=(const BrainNetwork& other) = delete;
        BrainNetwork& operator=(BrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        BrainNetwork(const brain::UIType neuron, const brain::UIType cap = 0, const utils::system::MemoryMode mode = utils::system::MemoryMode::Auto); // cap = slots of links of each neuron, calls init_ at the end
        BrainNetwork(const BrainNetwork& other) = delete;
        BrainNetwork(BrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~BrainNetwork() = default;
};

} // namespace end
#endif /* BRAINNETWORK_H */
