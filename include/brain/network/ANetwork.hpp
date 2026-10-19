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
    #include <utils/utils.hpp>      // _hot
    #include "../NetworkType.hpp"   // brain::Values
    #include "INetwork.hpp"         // brain::INetwork

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ANetwork: public brain::INetwork {
    protected:
        mutable brain::Values _inputs; // last inputs given to the network

        // ---------- Pre-Function -------- //
        void init_(void) override; // default: nothing (fallthrough), a child inits its neurons

    public:
        // ------------ Function ---------- //
        /* execution */
        _hot void inputs(const brain::Values& inputs) const final {this->_inputs = inputs;};

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
