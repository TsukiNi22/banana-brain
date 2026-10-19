/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 20/10/2026 by @author Tsukini

File Name:
##  @file NeuralNetwork.hpp

File Description:
##  Neural network handling every kind of network (layered, brain, banana): configuration, training and usage
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
    #include "NetworkType.hpp"        // brain::IType, brain::UIType, brain::FType, brain::Values, brain::Dataset, brain::*Fn
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
        std::pair<brain::UIType, brain::UIType> _size = {0, 0}; // <input, output> (named from 0 to n - 1), 0 = not set: the size is not checked
        std::pair<bool, brain::FType> _decayCoef = {true, 1.0}; // <smooth decay | by step, coef> applied to the learning rate every _decay steps
        brain::FType _learningRate = 1.0; // not edited during the training
        brain::UIType _decay = 0; // 0 = no decay, otherwise the learning rate is multiplied by the coef every n steps
        brain::IType _seed = 0; // 0 = random seed setup at start (won't be the same each time)

        /* training */
        brain::FType _learningRateDecayed = 1.0; // edited during the training
        brain::UIType _steps = 0;

        /* network / neurons storage */
        std::unique_ptr<brain::INetwork> _network; // nullptr = no network (init<...> first)

        // ---------- Pre-Function -------- //
        void decay_(void); // decay the learning rate
        void checkDataset_(const brain::Dataset& inputs, const brain::Dataset& expected_outputs) const; // same number of samples, not empty, each sample of the size set (_size, 0 = not set: not checked)

    public:
        // ---------- Pre-Function -------- //
        /* init (clear any existing network, specialized in the .cpp where the network classes are known) */
        template<brain::NetworkType type> requires (type == brain::NetworkType::Layered)
        void init(const brain::UIType layer, const brain::UIType neuron_per_layer); // normal usage with layers
        template<brain::NetworkType type> requires (type == brain::NetworkType::Brain)
        void init(const brain::UIType neurons, const std::pair<brain::FType, brain::FType>& size = {1.0, 1.0}); // custom network (more like a brain), size = 2d space of the neurons

        /* setter */
        void rand(const brain::FType min, const brain::FType max); // range of the random weights / bias generation (min <= max), applied to the network (init<...> first)
        void rand(const brain::RandFn& fn); // function used for the weights / bias generation (wins over the range), applied to the network
        void activationFunction(const brain::ActivationFn& fn); // external activation function (none by default), applied to the network
        void internalActivationFunction(const brain::ActivationFunction fn); // internal, more optimized than an external one, applied to the network
        _nodiscard const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& rand(void) const; // <<min, max>, function> of the network
        _nodiscard const std::pair<brain::ActivationFunction, brain::ActivationFn>& activationFunction(void) const; // <internal, function> of the network
        void learningRate(const brain::FType learning_rate, const brain::UIType learning_rate_decay = 0, const std::pair<bool, brain::FType>& coef = {true, 1.0}); // decay 0 = none, otherwise the learning rate is multiplied by the coef every n steps, <smooth, coef>: smooth decay or by step
        void resetLearningRate(void);

        /* training (auto): the backpropagation hook receives the outputs and the expected ones (can be empty: ignore them) */
        template<bool mean = false>
        void train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs); // n rounds, one coef per sample
        template<bool mean = false>
        void train(const brain::FType loss, const brain::Dataset& inputs, const brain::Dataset& expected_outputs); // until the mean loss is under this rate (checked after each round), one coef per sample

        /* training (manual) */
        brain::FType loss(const brain::FType output, const brain::FType expected) const; // compute the loss of one output
        template<bool mean> requires (mean)
        brain::FType loss(const brain::Values& outputs, const brain::Values& expected, const brain::FType coef = 1.0) const; // compute the mean loss, coef applied to every value (specialized in the .cpp)
        template<bool mean> requires (!mean)
        brain::Values loss(const brain::Values& outputs, const brain::Values& expected, const brain::FType coef = 1.0) const; // compute the loss of each output, coef applied to every value (specialized in the .cpp)
        std::pair<brain::Values, brain::FType> loss(const brain::Values& outputs, const brain::Values& expected, const brain::FType coef = 1.0) const; // compute both: <loss of each output, mean loss>, coef applied to every value (call without template)

        /* training (manual) */
        void backpropagation(const brain::FType coef); // coef between -1 <-> 1 (100% == perfect, -100% == totally false), can be above or under for a massive change
        void backpropagation(const brain::Values& coefs); // same as above with a special coef for each output

        /* usage */
        brain::Values exec(const brain::Values& inputs) const; // doesn't touch the content at all (the only call allowed when frozen)

        /* init (the weights and the bias of the network) */
        void weightInit(const brain::FType fallback = 0.01); // weights at 1.0 by default (not init -> dead network), fallback used on a result that falls on 0
        void biasInit(const brain::FType fallback = 0.01); // bias at 0.0 by default, fallback used on a result that falls on 0

        /* training (manual) */
        void inputs(const brain::Values& inputs) const; // load the inputs into the network
        _nodiscard brain::Values apply(void) const; // execute the network with the given inputs

        /* getter */
        _nodiscard std::pair<brain::UIType, brain::UIType> neurons(void) const; // <layers, neurons> of the network (NoNetwork without one)

        /* miscellaneous */
        void load(const std::filesystem::path& path); // load a network from a file
        void save(const std::filesystem::path& path) const; // save the actual network in a file
        void clear(void); // remove the network and reset the training state

        // ------------ Function ---------- //
        /* init (clear any existing network, specialized in the .cpp where the network classes are known) */

        /* training (manual) */

        /* setter (random) */
        _cold inline void seed(const brain::IType seed)                        {this->requireUnfrozen(); this->_seed = seed;}; // seed of the random generator (0 = random)

        /* setter (size) */
        _cold inline void size(const brain::UIType input, const brain::UIType output) {this->requireUnfrozen(); this->_size = {input, output};}; // number of inputs and outputs (named from 0 to n - 1)
        _cold inline void input(const brain::UIType input)                            {this->requireUnfrozen(); this->_size.first = input;};
        _cold inline void output(const brain::UIType output)                          {this->requireUnfrozen(); this->_size.second = output;};

        /* setter (function) */
        _cold inline void lossFunction(const brain::LossFn& fn)                          {this->requireUnfrozen(); this->_lossFunction = {brain::LossFunction::None, fn};}; // external loss function (none by default)
        _cold inline void internalLossFunction(const brain::LossFunction fn)             {this->requireUnfrozen(); this->_lossFunction = {fn, nullptr};}; // internal, more optimized than an external one

        /* getter */
        _cold _nodiscard inline brain::IType seed(void) const                {return this->_seed;};
        _cold _nodiscard inline brain::UIType input(void) const              {return this->_size.first;};
        _cold _nodiscard inline brain::UIType output(void) const             {return this->_size.second;};
        _cold _nodiscard inline brain::FType learningRate(void) const        {return this->_learningRate;};
        _cold _nodiscard inline brain::FType learningRateDecayed(void) const {return this->_learningRateDecayed;};
        _cold _nodiscard inline brain::UIType learningSteps(void) const      {return this->_steps;};

        /* getter (pair) */
        _cold _nodiscard inline const std::pair<brain::UIType, brain::UIType>& size(void) const {return this->_size;}; // <input, output>
        _cold _nodiscard inline const std::pair<brain::LossFunction, brain::LossFn>& lossFunction(void) const {return this->_lossFunction;};

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
