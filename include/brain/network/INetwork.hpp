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
##  @file INetwork.hpp

File Description:
##  Interface of every network stored behind a brain::NeuralNetwork (usable without one too)
\**************************************************************/

#ifndef INETWORK_H
    #define INETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../optimizer/IOptimizer.hpp"   // brain::IOptimizer
    #include "../backend/IBackend.hpp"       // brain::IBackend
    #include "../NetworkDefine.hpp"          // brain::NetworkType, brain::PruneTarget, BRAIN_DEFAULT_FALLBACK, BRAIN_DEFAULT_PRUNE
    #include "../NetworkType.hpp"            // brain::UIType, brain::FType, brain::Values, brain::Dataset, brain::Bytes, brain::Rand, brain::Activation
    #include <utility>                       // std::pair
    #include <memory>                        // std::unique_ptr, std::shared_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class INetwork {
    protected:
        // ---------- Pre-Function -------- //
        virtual void init_(void) = 0; // init the neurons, called at the end of the constructors

    public:
        // ---------- Pre-Function -------- //
        /* factory */
        static std::unique_ptr<brain::INetwork> create(const brain::Bytes& data); // reads the kind of the header and builds the right network (load without knowing the type)

        /* serialization */
        virtual brain::Bytes serialize(void) const = 0; // every info needed to rebuild the network (neurons, weights, bias...)
        virtual void deserialize(const brain::Bytes& data) = 0; // rebuild the network from the data given by serialize()

        /* execution */
        virtual void inputs(const brain::Values& inputs) = 0; // training: load the inputs into the network
        virtual brain::Values apply(void) = 0; // training: execute the network with the loaded inputs (keeps what the backpropagation needs)
        virtual brain::Values apply(const brain::Values& inputs) const = 0; // execution: nothing stored in the network
        virtual brain::Dataset applyAll(const brain::Dataset& inputs) const = 0; // execution of several samples: the outputs of each one

        /* training */
        virtual void backpropagation(const brain::FType coef) = 0; // one coefficient for every output (the learning rate is already in it)
        virtual void backpropagation(const brain::Values& coefs) = 0; // same as above with a special coef for each output

        /* init */
        virtual void weightInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) = 0; // weights at 1.0 by default (not init -> dead network), fallback used on a result that falls on 0
        virtual void biasInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) = 0; // bias at 0.0 by default, fallback used on a result that falls on 0

        /* setter (random) */
        virtual void rand(const brain::FType min, const brain::FType max) = 0; // range of the random weights / bias generation (min <= max, RandInit::Uniform / Normal)
        virtual void rand(const brain::Rand& rand) = 0; // RandInit mode or generator fn(fan_in, fan_out)

        /* setter (size) */
        virtual void size(const brain::UIType input, const brain::UIType output) = 0; // number of inputs and outputs (named from 0 to n - 1), 0 = not set: the size is not checked
        virtual void input(const brain::UIType input) = 0; // number of inputs
        virtual void output(const brain::UIType output) = 0; // number of outputs

        /* setter (function) */
        virtual void activation(const brain::Activation& fn) = 0; // internal enum, value fn or vector fn (none by default: raw values)

        /* setter (training) */
        virtual void optimizer(brain::IOptimizer* optimizer) = 0; // not owned (the NeuralNetwork keeps it), nullptr = SGD
        virtual void stepLimit(const brain::FType limit) = 0; // given by the NeuralNetwork from TrainOptions::stepLimit at the start of train()

        /* setter (backend) */
        virtual void backend(std::shared_ptr<brain::IBackend> backend) = 0; // nullptr = CpuBackend::shared(), error when !backend->supports(type())

        /* getter */
        virtual brain::UIType input(void) const = 0; // number of inputs (0 = not set)
        virtual brain::UIType output(void) const = 0; // number of outputs (0 = not set)
        virtual const std::pair<brain::UIType, brain::UIType>& size(void) const = 0; // <input, output>
        virtual const brain::Rand& rand(void) const = 0; // RandInit mode or generator
        virtual const std::pair<brain::FType, brain::FType>& range(void) const = 0; // <min, max> of the random weights / bias
        virtual const brain::Activation& activation(void) const = 0;
        virtual std::pair<brain::UIType, brain::UIType> neurons(void) const = 0; // <layers, neurons>: layers = 0 on a brain network, neurons = total number of neurons
        virtual brain::NetworkType type(void) const = 0;
        virtual const std::shared_ptr<brain::IBackend>& backend(void) const = 0; // never null

        /* shape / state */
        virtual std::unique_ptr<brain::INetwork> clone(void) const = 0; // deep copy (NeuralNetwork::clone + snapshot of the best network), the backend is shared
        virtual void reset(void) = 0; // weights / bias back to their init, same shape
        virtual void prune(const brain::FType influence = BRAIN_DEFAULT_PRUNE, const brain::PruneTarget target = brain::PruneTarget::Both) = 0; // remove the neurons / links with an influence under x%

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
