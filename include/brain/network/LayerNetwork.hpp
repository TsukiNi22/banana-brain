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
    #include <utils/utils.hpp>         // _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::NetworkType, brain::PruneTarget, BRAIN_DEFAULT_FALLBACK, BRAIN_DEFAULT_PRUNE
    #include "../NetworkType.hpp"      // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::Buffer, brain::Layers, brain::LayerNeuron, brain::LayerOptions, brain::Activation
    #include "INetwork.hpp"            // brain::INetwork
    #include "ANetwork.hpp"            // brain::ANetwork
    #include <utility>                 // std::pair
    #include <memory>                  // std::unique_ptr
    #include <vector>                  // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// apply / backpropagation / applyGradients only describe the compute: _backend->matVec / outer / axpy (no loop / thread of their own)
class LayerNetwork: public brain::ANetwork {
    private:
        /* structure */
        brain::Layers _layers; // number of neurons of each layer (the inputs are not a layer)
        std::vector<brain::LayerNeuron> _neurons; // every neuron, layer after layer
        brain::Buffer _weights; // weights + bias of every layer side by side (rows x (cols + 1): the last column of a row is its bias), a layer = one Slice
        std::vector<brain::Activation> _activations; // one per layer (missing = the activation of the network), ex: relu hidden + softmax output

        /* mini-batch */
        std::vector<brain::Buffer> _gradients; // accumulated on the mini-batch (weights + bias), one per worker of the backend (summed by applyGradients)
        brain::UIType _batchCount = 0; // samples accumulated since the last applyGradients

        /* dropout */
        brain::Values _dropout; // probability per layer to cut a neuron during the training only (0 = none, default)
        std::vector<std::vector<bool>> _masks; // the neurons cut for the current sample, one per worker of the backend

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

        /* training */
        void backpropagation(const brain::FType coef) final; // accumulated in the gradients of the worker
        void backpropagation(const brain::Values& coefs) final;
        void applyGradients(void); // mean of the batch through the optimizer, called by the train at the end of each batch

        /* init */
        void weightInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) final;
        void biasInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) final;

        /* setter */
        using brain::ANetwork::activation;
        void activation(const brain::UIType layer, const brain::Activation& fn); // the activation of one layer (wins over the one of the network)
        void dropout(const brain::UIType layer, const brain::FType probability); // 0 = none
        void dropout(const brain::FType probability); // every hidden layer, 0 = none

        /* shape */
        void addLayer(const brain::UIType index, const brain::UIType neurons); // keeps the existing weights, the new ones at init
        void removeLayer(const brain::UIType index);
        void resize(const brain::UIType layer, const brain::UIType neurons); // keeps the weights of the neurons left
        void prune(const brain::FType influence = BRAIN_DEFAULT_PRUNE, const brain::PruneTarget target = brain::PruneTarget::Both) final; // neurons / weights with an influence under x%

        /* state */
        std::unique_ptr<brain::INetwork> clone(void) const final;
        void reset(void) final;

        /* getter */
        std::pair<brain::UIType, brain::UIType> neurons(void) const final; // <layers, total neurons>

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard brain::NetworkType type(void) const final                            {return brain::NetworkType::Layered;};
        _cold _nodiscard inline const brain::Layers& layers(void) const                       {return this->_layers;}; // number of neurons of each layer
        _cold _nodiscard inline const std::vector<brain::Activation>& activations(void) const {return this->_activations;}; // one per layer
        _cold _nodiscard inline const brain::Values& dropout(void) const                      {return this->_dropout;}; // probability per layer

        // ------------ Operator ---------- //
        LayerNetwork& operator=(const LayerNetwork& other) = delete;
        LayerNetwork& operator=(LayerNetwork&& other) = delete;

        // ---------- Constructor --------- //
        LayerNetwork(const brain::Layers& layers, const brain::LayerOptions& options = {}); // the number of neurons of each layer, calls init_ at the end
        LayerNetwork(const brain::UIType layer, const brain::UIType neuron_per_layer, const brain::LayerOptions& options = {}): LayerNetwork(brain::Layers(layer, neuron_per_layer), options) { /* Nothing */ }; // the same number of neurons on every layer
        LayerNetwork(const LayerNetwork& other) = delete;
        LayerNetwork(LayerNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~LayerNetwork() = default;
};

} // namespace end
#endif /* LAYERNETWORK_H */
