/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 19/10/2026 by @author Tsukini

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
    #include <utils/utils.hpp>      // _hot, _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::ActivationFunction
    #include "../NetworkType.hpp"   // brain::FType, brain::Values, brain::RandFn, brain::ActivationFn
    #include "INetwork.hpp"         // brain::INetwork
    #include <utility>              // std::pair
    #include <cstdint>              // std::uint8_t

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ANetwork: public brain::INetwork {
    protected:
        std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn> _rand = {{-1.0, 1.0}, nullptr}; // <<min, max>, generator from the seed> used for the random weights / bias at start (the function wins when set)
        std::pair<brain::ActivationFunction, brain::ActivationFn> _activationFunction = {brain::ActivationFunction::None, nullptr}; // internal first, the function when set, raw values when none
        mutable brain::Values _inputs; // last inputs given to the network

        // ---------- Pre-Function -------- //
        void init_(void) override; // default: nothing (fallthrough), a child inits its neurons
        std::uint8_t serializeActivation_(void) const; // BRAIN_ACTIVATION_NONE | BRAIN_ACTIVATION_CUSTOM | the internal mode
        void deserializeActivation_(const std::uint8_t mode); // setup the internal mode, a custom one is only warned (to give again)

    public:
        // ---------- Pre-Function -------- //
        /* execution */
        void inputs(const brain::Values& inputs) const final;

        /* setter (random) */
        void rand(const brain::FType min, const brain::FType max) final;
        void rand(const brain::RandFn& fn) final;

        /* setter (function) */
        void activationFunction(const brain::ActivationFn& fn) final;
        void internalActivationFunction(const brain::ActivationFunction fn) final;

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard inline const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& rand(void) const final {return this->_rand;};
        _cold _nodiscard inline const std::pair<brain::ActivationFunction, brain::ActivationFn>& activationFunction(void) const final {return this->_activationFunction;};

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
