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
##  @file INetwork.hpp

File Description:
##  Interface of every network stored behind a brain::NeuralNetwork
\**************************************************************/

#ifndef INETWORK_H
    #define INETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes
    #include <utility>              // std::pair

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class INetwork {
    protected:
        // ---------- Pre-Function -------- //
        virtual void init_(void) = 0; // init the neurons, called at the end of the constructors

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        virtual brain::Bytes serialize(void) const = 0; // every info needed to rebuild the network (neurons, weights, bias...)
        virtual void deserialize(const brain::Bytes& data) = 0; // rebuild the network from the data given by serialize()

        /* execution */
        virtual void inputs(const brain::Values& inputs) const = 0; // load the inputs into the network
        virtual brain::Values apply(void) const = 0; // execute the network with the given inputs
        virtual void backpropagation(const brain::FType coef) = 0; // coef between -1 <-> 1 (100% == perfect, -100% == totally false), can be above or under for a massive change
        virtual void backpropagation(const brain::Values& coefs) = 0; // same as above with a special coef for each output

        /* setup */
        virtual void weightInit(const brain::FType fallback = 0.01) = 0; // weights at 1.0 by default (not init -> dead network), fallback used on a result that falls on 0
        virtual void biasInit(const brain::FType fallback = 0.01) = 0; // bias at 0.0 by default, fallback used on a result that falls on 0

        /* getter */
        virtual std::pair<brain::UIType, brain::UIType> neurons(void) const = 0; // <layers, neurons>: layers = 0 on a brain network, neurons = total number of neurons

        // ------------ Operator ---------- //
        INetwork& operator=(const INetwork& other) = delete;
        INetwork& operator=(INetwork&& other) = delete;

        // ---------- Constructor --------- //
        INetwork() = default;
        INetwork(const INetwork& other) = delete;
        INetwork(INetwork&& other) = delete;

        // ----------- Destructor --------- //
        virtual ~INetwork() = default;
};

} // namespace end
#endif /* INETWORK_H */
