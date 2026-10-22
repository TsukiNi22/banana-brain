/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 22/10/2026 by @author Tsukini

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
    #define _System
    #include <utils/utils.hpp>      // _cold, _nodiscard, utils::system::HybridMemory
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::Layers, brain::LayerNeuron
    #include "ANetwork.hpp"         // brain::ANetwork
    #include <numeric>              // std::accumulate
    #include <utility>              // std::pair
    #include <vector>               // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class LayerNetwork: public brain::ANetwork {
    private:
        brain::Layers _layers; // number of neurons of each layer (the inputs are not a layer)
        utils::system::HybridMemory<brain::LayerNeuron> _neurons; // every neuron, layer after layer (gpu when available)
        utils::system::HybridMemory<brain::FType> _weights; // the weights of every neuron, from LayerNeuron::incoming (LayerNeuron::count weights)
        brain::UIType _layoutInput = 0; // number of inputs the memories were set up for
        std::vector<brain::Values> _sums; // training: the sum (before the activation) of each neuron, layer by layer, of the last apply()
        std::vector<brain::Values> _outputs; // training: the output of each layer (the inputs are not in it), of the last apply()

        // ---------- Pre-Function -------- //
        brain::Values forward_(const brain::Values& inputs, std::vector<brain::Values>* sums, std::vector<brain::Values>* outputs) const; // layer after layer, what each layer computed is kept when asked (training)
        void layout_(void); // allocate the neurons and the weights, every one at its default (the weights are lost)

        // ------------ Function ---------- //
        _hot brain::Values compute_(const brain::Values& inputs) const final {return this->forward_(inputs, nullptr, nullptr);}; // execution: nothing kept
        _cold void init_(void) final    {this->layout_();}; // set the memories up: the neurons of every layer, the weights at 1.0 and the bias at 0.0
        _cold void resized_(void) final {if (this->_ioSize.first != this->_layoutInput) this->layout_();}; // the number of inputs gives the weights of the first layer: set the memories up again when it changes

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

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard inline const brain::Layers& layers(void) const                    {return this->_layers;}; // number of neurons of each layer
        _cold _nodiscard std::size_t memory(void) const final {return this->_neurons.bytes() + this->_weights.bytes();}; // neurons and weights
        _cold _nodiscard std::pair<brain::UIType, brain::UIType> neurons(void) const final {return {this->_layers.size(), std::accumulate(this->_layers.begin(), this->_layers.end(), brain::UIType{0})};}; // <layers, total neurons>

        // ------------ Operator ---------- //
        LayerNetwork& operator=(const LayerNetwork& other) = delete;
        LayerNetwork& operator=(LayerNetwork&& other) = delete;

        // ---------- Constructor --------- //
        LayerNetwork(const brain::Layers& layers, const utils::system::MemoryMode mode = utils::system::MemoryMode::Auto); // the number of neurons of each layer, calls init_ at the end
        LayerNetwork(const brain::UIType layer, const brain::UIType neuron_per_layer, const utils::system::MemoryMode mode = utils::system::MemoryMode::Auto): LayerNetwork(brain::Layers(layer, neuron_per_layer), mode) { /* Nothing */ }; // the same number of neurons on every layer, no layer: the empty list is refused
        LayerNetwork(const LayerNetwork& other) = delete;
        LayerNetwork(LayerNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~LayerNetwork() = default;
};

} // namespace end
#endif /* LAYERNETWORK_H */
