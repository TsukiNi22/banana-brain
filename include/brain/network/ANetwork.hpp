/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 21/10/2026 by @author Tsukini

File Name:
##  @file ANetwork.hpp

File Description:
##  Abstract network: inputs storage and the default (empty) neurons init
\**************************************************************/

#ifndef ANETWORK_H
    #define ANETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _System
    #include <utils/utils.hpp>         // _hot, _cold, _nodiscard, utils::system::HybridMemory, utils::system::IGpu, utils::system::MemoryMode
    #include "../tools/activation.hpp" // brain::tools::SigmoidActivation...
    #include "../NetworkDefine.hpp"    // brain::ActivationFunction
    #include "../NetworkType.hpp"      // brain::FType, brain::Values, brain::Bytes, brain::RandFn, brain::ActivationFn
    #include "INetwork.hpp"            // brain::INetwork
    #include <utility>                 // std::pair
    #include <cstdint>                 // std::uint8_t
    #include <cstddef>                 // std::size_t
    #include <memory>                  // std::shared_ptr
    #include <string>                  // std::string

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ANetwork: public brain::INetwork {
    protected:
        std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn> _rand = {{-1.0, 1.0}, nullptr}; // <<min, max>, generator from the seed> used for the random weights / bias at start (the function wins when set)
        std::pair<brain::ActivationFunction, brain::ActivationFn> _activationFunction = {brain::ActivationFunction::None, nullptr}; // internal first, the function when set, raw values when none
        std::pair<brain::ScoreFunction, brain::ScoreFn> _scoreFunction = {brain::ScoreFunction::Difference, nullptr}; // internal first, the function when set (not saved in the files, like the learning rate)
        brain::FType _scoreRate = 1.0; // the scores are multiplied by it
        std::pair<brain::UIType, brain::UIType> _ioSize = {0, 0}; // <input, output> (named from 0 to n - 1), 0 = not set: the size is not checked
        std::shared_ptr<utils::system::IGpu> _gpu = nullptr; // the gpu handler of the mode (shared by the networks of the program), given to every memory of the network (nullptr = ram)
        utils::system::MemoryMode _mode = utils::system::MemoryMode::Auto; // mode asked at the construction
        brain::Values _inputs; // last inputs given to the network (training)

        // ---------- Pre-Function -------- //
        std::uint8_t serializeActivation_(void) const; // BRAIN_ACTIVATION_NONE | BRAIN_ACTIVATION_CUSTOM | the internal mode
        void deserializeActivation_(const std::uint8_t mode); // setup the internal mode, a custom one is only warned (to give again)
        void serializeHeader_(brain::Bytes& out, const std::uint8_t kind) const; // magic bytes, kind, activation mode, sizes and random range (the common part of every network)
        brain::UIType deserializeHeader_(const brain::Bytes& data, const std::uint8_t kind); // check the magic bytes and the kind, read the common part, returns the offset after it
        virtual brain::Values compute_(const brain::Values& inputs) const = 0; // execution of the network (the inputs are already checked)
        brain::FType activate_(const brain::FType value) const; // the activation function of the network on a value (the external function when set, else the internal one, None: the value)
        brain::FType derivative_(const brain::FType sum, const brain::FType output) const; // derivative of the activation function at the sum (output = activation of the sum): the exact one of an internal function, estimated for an external one
        brain::FType random_(const brain::FType fallback) const; // a random value for a weight / bias: the random function when set, else in the random range, the fallback on a result that falls on 0 (non finite: error)

        // ------------ Function ---------- //
        _cold void init_(void) override        { /* Nothing */ }; // default: nothing (fallthrough), a child inits its neurons
        _cold virtual bool scored_(void) const {return false;}; // true: the coefficient is a score (positive = good), else an error (positive = too high): the sign of the internal score functions
        _cold virtual void resized_(void)      { /* Nothing */ }; // the sizes of the network changed: a child that depends on them sets its memory up again
        template<typename T>
        _cold _nodiscard inline utils::system::HybridMemory<T> makeMemory_(const brain::UIType size = 0) const // a memory on the gpu of the network (ram without one)
        {return this->_gpu ? utils::system::HybridMemory<T>(size, this->_gpu) : utils::system::HybridMemory<T>(size, utils::system::MemoryMode::Ram);};

    public:
        // ---------- Pre-Function -------- //
        /* execution */
        void inputs(const brain::Values& inputs) override;
        brain::Values apply(const brain::Values& inputs) const final;
        brain::Dataset applyAll(const brain::Dataset& inputs) const override; // one apply() per sample (a child that computes them together overrides it)

        /* setter (size) */
        void size(const brain::UIType input, const brain::UIType output) final; // the sizes are restored when resized_() refuses them

        /* setter (random) */
        void rand(const brain::FType min, const brain::FType max) final;
        void rand(const brain::RandFn& fn) final;

        /* setter (function) */
        void activationFunction(const brain::ActivationFn& fn) final;
        void internalActivationFunction(const brain::ActivationFunction fn) final;
        void scoreFunction(const brain::ScoreFn& fn) final;
        void internalScoreFunction(const brain::ScoreFunction fn) final;
        void scoreRate(const brain::FType rate) final;

        /* training */
        brain::Values scores(const brain::Values& outputs, const brain::Values& expected) const final;

        // ------------ Function ---------- //
        /* setter (size) */
        _cold inline void input(const brain::UIType input) final   {this->size(input, this->_ioSize.second);};
        _cold inline void output(const brain::UIType output) final {this->size(this->_ioSize.first, output);};

        /* getter */
        _cold _nodiscard inline brain::UIType input(void) const final  {return this->_ioSize.first;};
        _cold _nodiscard inline brain::UIType output(void) const final {return this->_ioSize.second;};
        _cold _nodiscard inline const std::pair<brain::UIType, brain::UIType>& size(void) const final            {return this->_ioSize;};
        _cold _nodiscard inline const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& rand(void) const final         {return this->_rand;};
        _cold _nodiscard inline const std::pair<brain::ActivationFunction, brain::ActivationFn>& activationFunction(void) const final {return this->_activationFunction;};
        _cold _nodiscard inline const std::pair<brain::ScoreFunction, brain::ScoreFn>& scoreFunction(void) const {return this->_scoreFunction;}; // <internal, function> of the score
        _cold _nodiscard inline brain::FType scoreRate(void) const     {return this->_scoreRate;};

        /* getter (memory) */
        _cold _nodiscard inline bool onGpu(void) const            {return this->_gpu != nullptr && this->_gpu->isAvailable();}; // the memories are stored by a gpu (else the ram)
        _cold _nodiscard inline utils::system::MemoryMode memoryMode(void) const            {return this->onGpu() ? this->_gpu->getMode() : utils::system::MemoryMode::Ram;}; // backend really used
        _cold _nodiscard inline utils::system::MemoryMode requestedMode(void) const         {return this->_mode;}; // mode asked at the construction
        _cold _nodiscard inline const std::shared_ptr<utils::system::IGpu>& gpu(void) const {return this->_gpu;}; // the shared handler (nullptr = ram)
        _cold _nodiscard inline std::string gpuName(void) const   {return this->onGpu() ? this->_gpu->getName() : "[None]";}; // name of the device
        _cold _nodiscard inline std::size_t gpuMemory(void) const {return this->onGpu() ? this->_gpu->getMemory() : 0;}; // memory of the device in bytes (0 = unknown or ram)

        // ------------ Operator ---------- //
        ANetwork& operator=(const ANetwork& other) = delete;
        ANetwork& operator=(ANetwork&& other) = delete;

        // ---------- Constructor --------- //
        ANetwork(const utils::system::MemoryMode mode = utils::system::MemoryMode::Auto); // the gpu handler of the network (Auto: the first gpu found, the ram without one)
        ANetwork(const ANetwork& other) = delete;
        ANetwork(ANetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~ANetwork() = default;
};

} // namespace end
#endif /* ANETWORK_H */
