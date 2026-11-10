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
##  @file LayerNetwork.hpp

File Description:
##  Classic network made of layers of neurons
\**************************************************************/

#ifndef LAYERNETWORK_H
    #define LAYERNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _cold, _nodiscard
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::Layers, brain::LayerNeuron
    #include "ANetwork.hpp"         // brain::ANetwork
    #include <utility>              // std::pair
    #include <vector>               // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class LayerNetwork: public brain::ANetwork {
    private:
        brain::Layers _layers; // number of neurons of each layer (the inputs are not a layer)
        std::vector<brain::LayerNeuron> _neurons; // every neuron, layer after layer
        std::vector<brain::FType> _weights; // the weights of every neuron, from LayerNeuron::incoming (LayerNeuron::count weights)

        // ---------- Pre-Function -------- //
        void init_(void) final; // create the neurons of every layer, the weights at 1.0 and the bias at 0.0

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const final;
        void deserialize(const brain::Bytes& data) final;

        /* execution */
        using brain::ANetwork::apply;
        brain::Values apply(void) final;
        void backpropagation(const brain::FType coef) final;
        void backpropagation(const brain::Values& coefs) final;

        /* setup */
        void weightInit(const brain::FType fallback = 0.01) final;
        void biasInit(const brain::FType fallback = 0.01) final;

        /* getter */
        std::pair<brain::UIType, brain::UIType> neurons(void) const final; // <layers, total neurons>

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard inline const brain::Layers& layers(void) const {return this->_layers;}; // number of neurons of each layer

        // ------------ Operator ---------- //
        LayerNetwork& operator=(const LayerNetwork& other) = delete;
        LayerNetwork& operator=(LayerNetwork&& other) = delete;

        // ---------- Constructor --------- //
        LayerNetwork(const brain::Layers& layers); // the number of neurons of each layer, calls init_ at the end
        LayerNetwork(const brain::UIType layer, const brain::UIType neuron_per_layer): LayerNetwork(brain::Layers(layer, neuron_per_layer)) { /* Nothing */ }; // the same number of neurons on every layer
        LayerNetwork(const LayerNetwork& other) = delete;
        LayerNetwork(LayerNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~LayerNetwork() = default;
};

} // namespace end
#endif /* LAYERNETWORK_H */
