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
##  @file ANetwork.hpp

File Description:
##  Abstract network: the common configuration (sizes, random, activation) and its serialization
\**************************************************************/

#ifndef ANETWORK_H
    #define ANETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>        // _cold, _nodiscard
    #include "../NetworkDefine.hpp"   // brain::ActivationFunction
    #include "../NetworkType.hpp"     // brain::UIType, brain::FType, brain::Values, brain::Dataset, brain::Bytes, brain::RandFn, brain::ActivationFn
    #include "INetwork.hpp"           // brain::INetwork
    #include <utility>                // std::pair
    #include <cstdint>                // std::uint8_t

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ANetwork: public brain::INetwork {
    protected:
        std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn> _rand = {{-1.0, 1.0}, nullptr}; // <<min, max>, generator from the seed> used for the random weights / bias (the function wins when set)
        std::pair<brain::ActivationFunction, brain::ActivationFn> _activationFunction = {brain::ActivationFunction::None, nullptr}; // internal first, the function when set, raw values when none
        std::pair<brain::UIType, brain::UIType> _ioSize = {0, 0}; // <input, output> (named from 0 to n - 1), 0 = not set: the size is not checked
        brain::Values _inputs; // last inputs given to the network (training)

        // ---------- Pre-Function -------- //
        void serializeHeader_(brain::Bytes& out, const std::uint8_t kind) const; // magic bytes, kind, activation mode, sizes and random range (the common part of every network)
        brain::UIType deserializeHeader_(const brain::Bytes& data, const std::uint8_t kind); // check the magic bytes and the kind, read the common part, returns the offset after it
        brain::FType activate_(const brain::FType value) const; // the activation function of the network on a value
        brain::FType random_(const brain::FType fallback) const; // a random value for a weight / bias, the fallback on a result that falls on 0
        void init_(void) override; // default: nothing (fallthrough), a child inits its neurons

    public:
        // ---------- Pre-Function -------- //
        /* execution */
        using brain::INetwork::apply;
        void inputs(const brain::Values& inputs) override;
        brain::Values apply(const brain::Values& inputs) const override;
        brain::Dataset applyAll(const brain::Dataset& inputs) const override; // one apply() per sample

        /* setter (size) */
        void size(const brain::UIType input, const brain::UIType output) override;
        void input(const brain::UIType input) override;
        void output(const brain::UIType output) override;

        /* setter (random) */
        void rand(const brain::FType min, const brain::FType max) override;
        void rand(const brain::RandFn& fn) override;

        /* setter (function) */
        void activationFunction(const brain::ActivationFn& fn) override;
        void internalActivationFunction(const brain::ActivationFunction fn) override;

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard inline brain::UIType input(void) const override  {return this->_ioSize.first;};
        _cold _nodiscard inline brain::UIType output(void) const override {return this->_ioSize.second;};
        _cold _nodiscard inline const std::pair<brain::UIType, brain::UIType>& size(void) const override                                 {return this->_ioSize;};
        _cold _nodiscard inline const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& rand(void) const override         {return this->_rand;};
        _cold _nodiscard inline const std::pair<brain::ActivationFunction, brain::ActivationFn>& activationFunction(void) const override {return this->_activationFunction;};

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
