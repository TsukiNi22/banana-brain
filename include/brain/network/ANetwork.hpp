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
##  @file ANetwork.hpp

File Description:
##  Abstract network: the common configuration (sizes, random, activation), caches, update and serialization
\**************************************************************/

#ifndef ANETWORK_H
    #define ANETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>                // _cold, _nodiscard
    #include "../optimizer/IOptimizer.hpp"    // brain::IOptimizer
    #include "../backend/CpuBackend.hpp"      // brain::CpuBackend
    #include "../backend/IBackend.hpp"        // brain::IBackend
    #include "../NetworkDefine.hpp"           // brain::RandInit, brain::ActivationFunction, BRAIN_DEFAULT_RAND_MIN, BRAIN_DEFAULT_RAND_MAX
    #include "../NetworkType.hpp"             // brain::UIType, brain::FType, brain::Value, brain::Values, brain::Dataset, brain::Bytes, brain::Buffer, brain::Rand, brain::Activation
    #include "INetwork.hpp"                   // brain::INetwork
    #include <utility>                        // std::pair
    #include <cstdint>                        // std::uint8_t
    #include <memory>                         // std::shared_ptr
    #include <vector>                         // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ANetwork: public brain::INetwork {
    protected:
        /* configuration */
        brain::Rand _rand = brain::RandInit::Uniform; // RandInit mode or generator fn(fan_in, fan_out)
        std::pair<brain::FType, brain::FType> _range = {BRAIN_DEFAULT_RAND_MIN, BRAIN_DEFAULT_RAND_MAX}; // <min, max> of the random weights / bias (RandInit::Uniform / Normal)
        brain::Activation _activation = brain::ActivationFunction::None; // raw values when none
        std::pair<brain::UIType, brain::UIType> _ioSize = {0, 0}; // <input, output> (named from 0 to n - 1), 0 = not set: the size is not checked

        /* cache (training, one set per worker of the backend: [0] = the manual mode) */
        std::vector<brain::Buffer> _inputs; // last inputs given to the network
        std::vector<brain::Buffer> _outputs; // output of each neuron at the last apply (the derivatives start from it)
        std::vector<brain::Buffer> _deltas; // gradient of each neuron

        /* update */
        brain::IOptimizer* _optimizer = nullptr; // not owned (the NeuralNetwork keeps it), SGD when nullptr
        brain::FType _stepLimit = 0.0; // copy of TrainOptions::stepLimit for update_() (set by the NeuralNetwork, not a setting of its own)

        /* compute */
        std::shared_ptr<brain::IBackend> _backend = brain::CpuBackend::shared(); // never null, where apply / backpropagation / applyAll run, shared with the NeuralNetwork and the clones

        // ---------- Pre-Function -------- //
        /* serialization */
        void serializeHeader_(brain::Bytes& out, const std::uint8_t kind) const; // magic bytes, kind, activation mode, sizes and random range (the common part of every network)
        brain::UIType deserializeHeader_(const brain::Bytes& data, const std::uint8_t kind); // check the magic bytes and the kind, read the common part, returns the offset after it

        /* compute */
        brain::Value activate_(const brain::Value value) const; // the activation function of the network on a value
        brain::Values activate_(const brain::Values& values) const; // same on the whole vector (vector mode: softmax like)
        brain::FType derivative_(const brain::Value output) const; // the derivative of the activation from its output
        brain::Values derivative_(const brain::Values& outputs) const; // same on the whole vector (vector mode)
        brain::FType random_(const brain::FType fallback, const brain::UIType fan_in, const brain::UIType fan_out) const; // a random value for a weight / bias (RandInit / RandFn), the fallback on a result that falls on 0

        /* training */
        void update_(brain::FType& weight, const brain::UIType index, const brain::FType gradient); // through the optimizer (SGD when none) + the step limit

        /* init */
        void init_(void) override; // default: nothing (fallthrough), a child inits its neurons

    public:
        // ---------- Pre-Function -------- //
        /* execution */
        using brain::INetwork::apply;
        void inputs(const brain::Values& inputs) override;
        brain::Values apply(const brain::Values& inputs) const override;
        brain::Dataset applyAll(const brain::Dataset& inputs) const override; // _backend->applyAll(*this, inputs)

        /* setter (size) */
        void size(const brain::UIType input, const brain::UIType output) override;
        void input(const brain::UIType input) override;
        void output(const brain::UIType output) override;

        /* setter (random) */
        void rand(const brain::FType min, const brain::FType max) override;
        void rand(const brain::Rand& rand) override;

        /* setter (function) */
        void activation(const brain::Activation& fn) override;

        /* setter (backend) */
        void backend(std::shared_ptr<brain::IBackend> backend) override; // the old one synced first (pending jobs), then supports() checked, then one set of caches per worker of the new one

        // ------------ Function ---------- //
        /* setter (training) */
        _cold void optimizer(brain::IOptimizer* optimizer) override {this->_optimizer = optimizer;};
        _cold void stepLimit(const brain::FType limit) override     {this->_stepLimit = limit;};

        /* getter */
        _cold _nodiscard brain::UIType input(void) const override                                 {return this->_ioSize.first;};
        _cold _nodiscard brain::UIType output(void) const override                                {return this->_ioSize.second;};
        _cold _nodiscard const std::pair<brain::UIType, brain::UIType>& size(void) const override {return this->_ioSize;};
        _cold _nodiscard const brain::Rand& rand(void) const override                             {return this->_rand;};
        _cold _nodiscard const std::pair<brain::FType, brain::FType>& range(void) const override  {return this->_range;};
        _cold _nodiscard const brain::Activation& activation(void) const override                 {return this->_activation;};
        _cold _nodiscard const std::shared_ptr<brain::IBackend>& backend(void) const override     {return this->_backend;};

        // ------------ Operator ---------- //
        ANetwork& operator=(const ANetwork& other) = delete;
        ANetwork& operator=(ANetwork&& other) = delete;

        // ---------- Constructor --------- //
        ANetwork() = default;
        ANetwork(const ANetwork& other) = delete;
        ANetwork(ANetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~ANetwork() = default;
};

} // namespace end
#endif /* ANETWORK_H */
