/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 17/10/2026 by @author Tsukini

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
    #include "network/Network.hpp"    // brain::INetwork
    #include "NetworkDefine.hpp"      // brain::NetworkType, brain::ActivationFunction, brain::LossFunction
    #include "NetworkType.hpp"        // brain::IType, brain::UIType, brain::FType, brain::Values, brain::Dataset, brain::*Fn
    #include <filesystem>             // std::filesystem::path
    #include <concepts>               // std::same_as
    #include <utility>                // std::pair
    #include <memory>                 // std::unique_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// Handle every kind of network and its training, frozen (utils::type::Freezable) only exec() is allowed
class NeuralNetwork: public utils::type::Freezable {
    private:
        /* configuration */
        std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn> _rand = {{-1.0, 1.0}, nullptr};                                   // <<min, max>, generator from the seed> used for the random weights / bias at start (the function wins when set)
        std::pair<brain::ActivationFunction, brain::ActivationFn> _activationFunction = {brain::ActivationFunction::None, nullptr}; // internal first, the function when set, raw values when none
        std::pair<brain::LossFunction, brain::LossFn> _lossFunction = {brain::LossFunction::None, nullptr};                         // internal first, the function when set, no loss when none
        std::pair<brain::UIType, brain::UIType> _size = {0, 0};                                                                     // <input, output> (named from 0 to n - 1)
        std::pair<bool, brain::FType> _decayCoef = {true, 1.0};                                                                     // <smooth decay | by step, coef> applied to the learning rate every _decay steps
        brain::FType _learningRate = 1.0;                                                                                           // not edited during the training
        brain::UIType _decay = 0;                                                                                                   // 0 = no decay, otherwise the learning rate is multiplied by the coef every n steps
        brain::IType _seed = 0;                                                                                                     // 0 = random seed setup at start (won't be the same each time)

        /* training */
        brain::FType _learningRateDecayed = 1.0; // edited during the training
        brain::UIType _steps = 0;

        /* network / neurons storage */
        std::unique_ptr<brain::INetwork> _network; // nullptr = no network (init<...> first)

    public:
        // ---------- Pre-Function -------- //
        /* init */
        void weightInit(const brain::FType fallback = 0.01); // weights at 1.0 by default (not init -> dead network), fallback used on a result that falls on 0
        void biasInit(const brain::FType fallback = 0.01);   // bias at 0.0 by default, fallback used on a result that falls on 0

        /* setter */
        void learningRate(const brain::FType learning_rate, const brain::UIType learning_rate_decay = 0, const std::pair<bool, brain::FType>& coef = {true, 1.0}); // decay 0 = none, otherwise the learning rate is multiplied by the coef every n steps, <smooth, coef>: smooth decay or by step

        /* getter */
        std::pair<brain::UIType, brain::UIType> neurons(void) const; // <layers, neurons> of the network (NoNetwork without one)

        /* training (auto): the backpropagation hook receives the outputs and the expected ones (can be empty: ignore them) */
        void train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationFn& backpropagation);  // n rounds, one coef per sample
        void train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationsFn& backpropagation); // n rounds, one coef per output
        void train(const brain::FType loss, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationFn& backpropagation);    // until the mean loss is under this rate (checked after each round), one coef per sample
        void train(const brain::FType loss, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::BackpropagationsFn& backpropagation);   // until the mean loss is under this rate (checked after each round), one coef per output

        /* training (manual) */
        void load(void);                                                                    // load the network into memory (gpu) if not already done (failsafe)
        void unload(void);                                                                  // unload the network memory if not already done (failsafe)
        void inputs(const brain::Values& inputs);                                           // load the inputs into the network
        brain::Values apply(void);                                                          // execute the network with the given inputs
        void backpropagation(const brain::FType coef);                                      // coef between -1 <-> 1 (100% == perfect, -100% == totally false), can be above or under for a massive change
        void backpropagation(const brain::Values& coefs);                                   // same as above with a special coef for each output
        brain::FType loss(const brain::FType output, const brain::FType expected) const;    // compute the loss of one output
        brain::FType loss(const brain::Values& outputs, const brain::Values& expected) const; // compute the mean loss

        /* usage */
        brain::Values exec(const brain::Values& inputs) const; // doesn't touch the content at all (the only call allowed when frozen)

        /* miscellaneous */
        void load(const std::filesystem::path& path);       // load a network from a file
        void save(const std::filesystem::path& path) const; // save the actual network in a file
        void clear(void);                                   // remove the network and reset the training state

        // ------------ Function ---------- //
        /* init (clear any existing network) */
        template<brain::NetworkType type> requires (type == brain::NetworkType::Layered)
        _cold void init(const brain::UIType layer, const brain::UIType neuron_per_layer)                                        {/* Nothing */}; // normal usage with layers
        template<brain::NetworkType type> requires (type == brain::NetworkType::Brain) // xstyle: ignore CPP-CLASS-BLOCK (false positive: the default argument {1.0, 1.0} of the next line)
        _cold void init(const brain::UIType neurons, const std::pair<brain::FType, brain::FType>& size = {1.0, 1.0}) {/* Nothing */}; // custom network (more like a brain), size = 2d space of the neurons
        template<brain::NetworkType type> requires (type == brain::NetworkType::Banana)
        _cold void init(void)                                                                                          {/* Nothing */}; // pre made layer / neuron numbers shaped like a banana (don't ask why...)

        /* training (auto): the round / loss must be exactly a brain::UIType / brain::FType (no implicit conversion) */
        template<typename T, typename Fn> requires (!std::same_as<T, brain::UIType> && !std::same_as<T, brain::FType>)
        void train(const T round_or_loss, const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const Fn& backpropagation) = delete;

        /* setter */
        _cold inline void rand(const brain::FType min, const brain::FType max)                 {this->_rand.first = {min, max};};                                       // range of the random weights / bias generation
        _cold inline void rand(const brain::RandFn& fn)                                        {this->_rand.second = fn;};                                              // function used for the weights / bias generation (wins over the range)
        _cold inline void seed(const brain::IType seed)                                        {this->_seed = seed;};                                                   // seed of the random generator (0 = random)
        _cold inline void size(const brain::UIType input, const brain::UIType output)          {this->_size = {input, output};};                                        // number of inputs and outputs (named from 0 to n - 1)
        _cold inline void input(const brain::UIType input)                                     {this->_size.first = input;};
        _cold inline void output(const brain::UIType output)                                   {this->_size.second = output;};
        _cold inline void activationFunction(const brain::ActivationFn& fn)                    {this->_activationFunction = {brain::ActivationFunction::None, fn};};    // external activation function (none by default)
        _cold inline void internalActivationFunction(const brain::ActivationFunction fn)       {this->_activationFunction = {fn, nullptr};};                            // internal, more optimized than an external one
        _cold inline void lossFunction(const brain::LossFn& fn)                                {this->_lossFunction = {brain::LossFunction::None, fn};};                // external loss function (none by default)
        _cold inline void internalLossFunction(const brain::LossFunction fn)                   {this->_lossFunction = {fn, nullptr};};                                  // internal, more optimized than an external one

        /* getter */
        _cold _nodiscard inline brain::IType seed(void) const                                                                  {return this->_seed;};
        _cold _nodiscard inline const std::pair<brain::UIType, brain::UIType>& size(void) const                                {return this->_size;};               // <input, output>
        _cold _nodiscard inline brain::UIType input(void) const                                                                {return this->_size.first;};
        _cold _nodiscard inline brain::UIType output(void) const                                                               {return this->_size.second;};
        _cold _nodiscard inline brain::FType learningRate(void) const                                                          {return this->_learningRate;};
        _cold _nodiscard inline const std::pair<brain::ActivationFunction, brain::ActivationFn>& activationFunction(void) const {return this->_activationFunction;};
        _cold _nodiscard inline const std::pair<brain::LossFunction, brain::LossFn>& lossFunction(void) const                   {return this->_lossFunction;};
        _cold _nodiscard inline const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& rand(void) const        {return this->_rand;};               // <<min, max>, function>

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
