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
##  @file NeuralNetwork.hpp

File Description:
##  Neural network handling every kind of network (layered, brain): configuration, training and usage
\**************************************************************/

#ifndef NEURALNETWORK_H
    #define NEURALNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _CustomType
    #include <utils/utils.hpp>        // _hot, _cold, _nodiscard, utils::type::Freezable
    #include "network/INetwork.hpp"   // brain::INetwork
    #include "NetworkDefine.hpp"      // brain::NetworkType, brain::ActivationFunction, brain::LossFunction
    #include "NetworkType.hpp"        // brain::UIType, brain::FType, brain::Values, brain::Dataset, brain::Layers, brain::*Fn
    #include <filesystem>             // std::filesystem::path
    #include <utility>                // std::pair
    #include <memory>                 // std::unique_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// Handle every kind of network and its training, frozen (utils::type::Freezable) only exec() is allowed
class NeuralNetwork: public utils::type::Freezable {
    private:
        /* configuration */
        std::pair<brain::LossFunction, brain::LossFn> _lossFunction = {brain::LossFunction::None, nullptr}; // internal first, the function when set, no loss when none
        brain::FType _learningRate = 1.0;

        /* network / neurons storage */
        std::unique_ptr<brain::INetwork> _network; // nullptr = no network (init<...> first)

    public:
        // ---------- Pre-Function -------- //
        /* init (clear any existing network, specialized in the .cpp where the network classes are known) */
        template<brain::NetworkType type> requires (type == brain::NetworkType::Layered)
        void init(const brain::UIType layer, const brain::UIType neuron_per_layer); // normal usage with layers, the same number of neurons on every layer
        template<brain::NetworkType type> requires (type == brain::NetworkType::Layered)
        void init(const brain::Layers& layers); // normal usage with layers, the number of neurons of each layer
        template<brain::NetworkType type> requires (type == brain::NetworkType::Brain)
        void init(const brain::UIType neurons, const std::pair<brain::FType, brain::FType>& size = {1.0, 1.0}); // custom network (more like a brain), size = 2d space of the neurons
        void weightInit(const brain::FType fallback = 0.01); // weights at 1.0 by default (not init -> dead network), fallback used on a result that falls on 0
        void biasInit(const brain::FType fallback = 0.01); // bias at 0.0 by default, fallback used on a result that falls on 0

        /* setter (applied to the network) */
        void size(const brain::UIType input, const brain::UIType output); // number of inputs and outputs (named from 0 to n - 1), 0 = not set: the size is not checked
        void input(const brain::UIType input);
        void output(const brain::UIType output);
        void rand(const brain::FType min, const brain::FType max); // range of the random weights / bias generation (min <= max)
        void rand(const brain::RandFn& fn); // function used for the weights / bias generation (wins over the range)
        void activationFunction(const brain::ActivationFn& fn); // external activation function (none by default)
        void internalActivationFunction(const brain::ActivationFunction fn); // internal, more optimized than an external one
        void lossFunction(const brain::LossFn& fn); // external loss function (none by default)
        void internalLossFunction(const brain::LossFunction fn); // internal, more optimized than an external one
        void learningRate(const brain::FType learning_rate);

        /* getter */
        _nodiscard std::pair<brain::UIType, brain::UIType> neurons(void) const; // <layers, neurons> of the network (NoNetwork without one)

        /* training (auto) */
        void train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs); // n rounds over the samples, one update per sample

        /* training (manual) */
        void inputs(const brain::Values& inputs); // load the inputs into the network
        _nodiscard brain::Values apply(void); // execute the network with the loaded inputs (keeps what the backpropagation needs)
        void backpropagation(const brain::FType coef); // one coefficient for every output
        void backpropagation(const brain::Values& coefs); // same as above with a special coef for each output
        _nodiscard brain::FType loss(const brain::FType output, const brain::FType expected) const; // compute the loss of one output
        _nodiscard brain::FType loss(const brain::Values& outputs, const brain::Values& expected) const; // compute the mean loss

        /* usage */
        _nodiscard brain::Values exec(const brain::Values& inputs) const; // doesn't touch the content at all (the only call allowed when frozen)

        /* miscellaneous */
        void load(const std::filesystem::path& path); // load a network from a file
        void save(const std::filesystem::path& path) const; // save the actual network in a file
        void clear(void); // remove the network and reset the training state

        // ------------ Function ---------- //
        /* getter */
        _cold _nodiscard inline brain::FType learningRate(void) const                                         {return this->_learningRate;};
        _cold _nodiscard inline const std::pair<brain::LossFunction, brain::LossFn>& lossFunction(void) const {return this->_lossFunction;};

        /* getter (network) */
        _cold _nodiscard inline brain::INetwork* get(void)             {return this->_network.get();}; // the internal network (nullptr without one), the NeuralNetwork keeps it (no move)
        _cold _nodiscard inline const brain::INetwork* get(void) const {return this->_network.get();}; // read only access to the internal network (nullptr without one)

        // ------------ Operator ---------- //
        NeuralNetwork& operator=(const NeuralNetwork& other) = delete;
        NeuralNetwork& operator=(NeuralNetwork&& other) = delete;

        // ---------- Constructor --------- //
        NeuralNetwork() = default;
        NeuralNetwork(const NeuralNetwork& other) = delete;
        NeuralNetwork(NeuralNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~NeuralNetwork() = default;
};

} // namespace end
#endif /* NEURALNETWORK_H */
