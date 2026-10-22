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
    #include <utils/utils.hpp>      // _hot, _cold, _nodiscard, utils::system::HybridMemory, utils::system::IGpu, utils::system::MemoryMode
    #include "../NetworkDefine.hpp" // brain::ActivationFunction
    #include "../NetworkType.hpp"   // brain::FType, brain::Values, brain::Bytes, brain::RandFn, brain::ActivationFn
    #include "INetwork.hpp"         // brain::INetwork
    #include <utility>              // std::pair
    #include <cstdint>              // std::uint8_t
    #include <memory>               // std::shared_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ANetwork: public brain::INetwork {
    protected:
        std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn> _rand = {{-1.0, 1.0}, nullptr}; // <<min, max>, generator from the seed> used for the random weights / bias at start (the function wins when set)
        std::pair<brain::ActivationFunction, brain::ActivationFn> _activationFunction = {brain::ActivationFunction::None, nullptr}; // internal first, the function when set, raw values when none
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

        // ------------ Function ---------- //
        _cold void init_(void) override { /* Nothing */ }; // default: nothing (fallthrough), a child inits its neurons
        template<typename T>
        _cold _nodiscard inline utils::system::HybridMemory<T> makeMemory_(const brain::UIType size = 0) const // a memory on the gpu of the network (ram without one)
        {return this->_gpu ? utils::system::HybridMemory<T>(size, this->_gpu) : utils::system::HybridMemory<T>(size, utils::system::MemoryMode::Ram);};

    public:
        // ---------- Pre-Function -------- //
        /* execution */
        void inputs(const brain::Values& inputs) override;
        brain::Values apply(const brain::Values& inputs) const final;

        /* setter (random) */
        void rand(const brain::FType min, const brain::FType max) final;
        void rand(const brain::RandFn& fn) final;

        /* setter (function) */
        void activationFunction(const brain::ActivationFn& fn) final;
        void internalActivationFunction(const brain::ActivationFunction fn) final;

        // ------------ Function ---------- //
        /* setter (size) */
        _cold inline void size(const brain::UIType input, const brain::UIType output) final {this->_ioSize = {input, output};};
        _cold inline void input(const brain::UIType input) final                            {this->_ioSize.first = input;};
        _cold inline void output(const brain::UIType output) final                          {this->_ioSize.second = output;};

        /* getter */
        _cold _nodiscard inline brain::UIType input(void) const final                                  {return this->_ioSize.first;};
        _cold _nodiscard inline brain::UIType output(void) const final                                 {return this->_ioSize.second;};
        _cold _nodiscard inline const std::pair<brain::UIType, brain::UIType>& size(void) const final {return this->_ioSize;};
        _cold _nodiscard inline const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& rand(void) const final {return this->_rand;};
        _cold _nodiscard inline const std::pair<brain::ActivationFunction, brain::ActivationFn>& activationFunction(void) const final {return this->_activationFunction;};

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
