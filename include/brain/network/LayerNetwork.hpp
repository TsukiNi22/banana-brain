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
##  @file LayerNetwork.hpp

File Description:
##  Classic network made of layers of neurons
\**************************************************************/

#ifndef LAYERNETWORK_H
    #define LAYERNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::LayerNeuron
    #include "ANetwork.hpp"         // brain::ANetwork
    #include <utility>              // std::pair
    #include <vector>               // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class LayerNetwork: public brain::ANetwork {
    private:
        std::vector<std::vector<brain::LayerNeuron>> _neurons; // [layer][neuron]
        brain::UIType _neuronPerLayer = 0; // neurons of each layer
        brain::UIType _layer = 0; // number of layers

        // ---------- Pre-Function -------- //
        void init_(void) final; // create layer * neuron_per_layer neurons

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const final;
        void deserialize(const brain::Bytes& data) final;

        /* execution */
        brain::Values apply(void) const final;
        void backpropagation(const brain::FType coef) final;
        void backpropagation(const brain::Values& coefs) final;

        /* setup */
        void weightInit(const brain::FType fallback = 0.01) final;
        void biasInit(const brain::FType fallback = 0.01) final;

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard std::pair<brain::UIType, brain::UIType> neurons(void) const final {return {this->_layer, this->_neuronPerLayer};}; // <layers, neuron_per_layer>

        // ------------ Operator ---------- //
        LayerNetwork& operator=(const LayerNetwork& other) = delete;
        LayerNetwork& operator=(LayerNetwork&& other) = delete;

        // ---------- Constructor --------- //
        LayerNetwork(const brain::UIType layer, const brain::UIType neuron_per_layer); // calls init_ at the end
        LayerNetwork(const LayerNetwork& other) = delete;
        LayerNetwork(LayerNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~LayerNetwork() = default;
};

} // namespace end
#endif /* LAYERNETWORK_H */
