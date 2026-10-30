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
##  @file INetwork.hpp

File Description:
##  Interface of every network stored behind a brain::NeuralNetwork
\**************************************************************/

#ifndef INETWORK_H
    #define INETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkDefine.hpp"   // brain::ActivationFunction
    #include "../NetworkType.hpp"     // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::RandFn, brain::ActivationFn
    #include <utility>                // std::pair
    #include <cstddef>                // std::size_t

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
        virtual void deserialize(const brain::Bytes& data) = 0; // rebuild the network from the data given by serialize() (the common part is read first: on an invalid data use a new network, brain::NeuralNetwork::load does)

        /* execution */
        virtual void inputs(const brain::Values& inputs) = 0; // training: load the inputs into the network
        virtual brain::Values apply(void) = 0; // training: execute the network with the loaded inputs (keeps what the backpropagation needs)
        virtual brain::Values apply(const brain::Values& inputs) const = 0; // execution: nothing stored in the network, can be called from several threads at once
        virtual brain::Dataset applyAll(const brain::Dataset& inputs) const = 0; // execution of several samples: the outputs of each one, as apply() gives them (a brain network computes them together, which is faster than one call per sample)
        virtual void backpropagation(const brain::FType coef) = 0; // layered network: the error of the output (positive = too high, the output goes down), brain network: a score (positive = good, the links that made the output are strengthened, negative = weakened), the learning rate is already in it
        virtual void backpropagation(const brain::Values& coefs) = 0; // same as above with a special coef for each output
        virtual void batch(const bool enable) = 0; // true: backpropagation() only adds its gradient to the pending one, applyBatch() applies their mean (the pending gradients are dropped when it changes)
        virtual bool batch(void) const = 0; // the mini-batch mode
        virtual void applyBatch(void) = 0; // mini-batch: apply the mean of the gradients given since the last call (nothing pending: nothing done)

        /* setup */
        virtual void weightInit(const brain::FType fallback = 0.01) = 0; // weights at 1.0 by default (not init -> dead network), fallback used on a result that falls on 0
        virtual void biasInit(const brain::FType fallback = 0.01) = 0; // bias at 0.0 by default, fallback used on a result that falls on 0

        /* setter (random) */
        virtual void rand(const brain::FType min, const brain::FType max) = 0; // range of the random weights / bias generation (min <= max)
        virtual void rand(const brain::RandFn& fn) = 0; // function used for the weights / bias generation (wins over the range)

        /* setter (size) */
        virtual void size(const brain::UIType input, const brain::UIType output) = 0; // number of inputs and outputs (named from 0 to n - 1), 0 = not set: the size is not checked
        virtual void input(const brain::UIType input) = 0; // number of inputs
        virtual void output(const brain::UIType output) = 0; // number of outputs

        /* setter (function) */
        virtual void activationFunction(const brain::ActivationFn& fn) = 0; // external activation function (none by default)
        virtual void internalActivationFunction(const brain::ActivationFunction fn) = 0; // internal, more optimized than an external one
        virtual void scoreFunction(const brain::ScoreFn& fn) = 0; // external score function (none by default: the internal one)
        virtual void internalScoreFunction(const brain::ScoreFunction fn) = 0; // internal score function (Difference by default)
        virtual void scoreRate(const brain::FType rate) = 0; // the score rate (> 0, 1 by default): the scores() are multiplied by it
        virtual brain::Values scores(const brain::Values& outputs, const brain::Values& expected) const = 0; // the coefficients of the backpropagation for the outputs and the expected ones: score function, then score rate

        /* getter */
        virtual brain::UIType input(void) const = 0; // number of inputs (0 = not set)
        virtual brain::UIType output(void) const = 0; // number of outputs (0 = not set)
        virtual const std::pair<brain::UIType, brain::UIType>& size(void) const = 0; // <input, output>
        virtual const std::pair<std::pair<brain::FType, brain::FType>, brain::RandFn>& rand(void) const = 0; // <<min, max>, function>
        virtual const std::pair<brain::ActivationFunction, brain::ActivationFn>& activationFunction(void) const = 0; // internal first, the function when set, raw values when none
        virtual std::size_t memory(void) const = 0; // bytes of the memories of the network (on the gpu or in the ram)
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
