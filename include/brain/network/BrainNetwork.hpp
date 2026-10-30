/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 27/10/2026 by @author Tsukini

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
        brain::UIType _steps = 1; // time steps of the execution (the outputs are read after the last one)
        brain::UIType _threads = 0; // threads of the time steps on the host: 0 = by the size of the network (see BRAIN_THREADS_MIN_WORK), 1 = one thread, n = at most n (not saved in the files)
        brain::FType _threshold = BRAIN_SIGNAL_THRESHOLD; // a signal whose absolute value is under it is cancelled (set to 0) at each time step

        // ---------- Pre-Function -------- //
        void init_(void) override; // nothing to place, the neurons and the links come from deserialize
        template<typename Activation>
        void steps_(const Activation& activation, const brain::BrainNeuron* neurons, const brain::BrainLink* links, const brain::Values& inputs, brain::Values& state, brain::Values& next) const; // the time steps on the host with this activation function (compute_ chooses it once)
        template<typename Activation>
        void stepsAll_(const Activation& activation, const brain::BrainNeuron* neurons, const brain::BrainLink* links, const brain::UIType count, std::vector<brain::FType>& state, std::vector<brain::FType>& next) const; // the time steps of applyAll() on the host: the state of every neuron for the <count> samples side by side (neuron * count + sample)
        brain::UIType threadsFor_(void) const; // the threads this network uses for a time step: 1 for an external activation function, else by the number of slots of links of a step
        brain::Values compute_(const brain::Values& inputs) const override; // time steps: the first `input` neurons hold the inputs, the others compute from the previous step, the last `output` neurons are the outputs
        void resized_(void) override; // the inputs and the outputs have to fit in the neurons, an input receives no link and an output sends none
        void fits_(void) const; // the inputs and the outputs fit in the neurons (InvalidArgument)
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
        brain::Dataset applyAll(const brain::Dataset& inputs) const override; // the samples together on the host: the states of every sample side by side, the links read once per step for all of them (the same outputs as apply() for each one; a gpu network executes them one by one)
        void threshold(const brain::FType threshold); // a signal under it (absolute value) is cancelled at each time step, to stop the signals that are too weak to matter (>= 0, 0 = none)

        // ------------ Function ---------- //
        /* serialization */
        _cold void deserialize(const brain::Bytes& data) override {this->deserializeBody_(data, this->deserializeHeader_(data, static_cast<std::uint8_t>(BRAIN_MAGIC_BRAIN)));};

        /* execution (the training is refused) */
        _hot brain::Values apply(void) override                  {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork
        _hot void backpropagation(const brain::FType) override   {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork
        _hot void backpropagation(const brain::Values&) override {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork
        _hot void batch(const bool enable) override              {if (enable) throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork (off: nothing to do)
        _hot _nodiscard bool batch(void) const override          {return false;}; // never in mini-batch
        _hot void applyBatch(void) override                      { /* nothing pending: nothing done */ }; // ReadOnlyNetwork

        /* setup (the training is refused) */
        _cold void weightInit(const brain::FType = 0.01) override {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork
        _cold void biasInit(const brain::FType = 0.01) override   {throw utils::exception::ErrorException(utils::exception::ExternalCode::ReadOnlyNetwork);}; // ReadOnlyNetwork

        /* setter */
        _cold inline void threads(const brain::UIType count) {this->_threads = count;}; // threads of the time steps on the host: 0 = automatic (more neurons and links: more threads, a small network stays on one), 1 = one thread, n = at most n

        /* getter */
        _cold _nodiscard inline brain::UIType threads(void) const                          {return this->_threads;}; // 0 = automatic
        _cold _nodiscard inline brain::FType threshold(void) const                         {return this->_threshold;}; // the signal threshold
        _cold _nodiscard inline brain::UIType steps(void) const                            {return this->_steps;}; // time steps of the execution
        _cold _nodiscard inline brain::UIType links(void) const                            {return this->_cap;}; // slots of links of each neuron
        _cold _nodiscard std::size_t memory(void) const override                           {return this->_neurons.bytes() + this->_links.bytes();}; // neurons and links
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
