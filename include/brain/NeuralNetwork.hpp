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
    #include <utils/utils.hpp>              // _hot, _cold, _nodiscard, utils::type::Freezable
    #include "optimizer/IOptimizer.hpp"     // brain::IOptimizer
    #include "backend/CpuBackend.hpp"       // brain::CpuBackend
    #include "backend/IBackend.hpp"         // brain::IBackend
    #include "network/INetwork.hpp"         // brain::INetwork
    #include "hook/ITrainHook.hpp"          // brain::ITrainHook
    #include "NetworkDefine.hpp"            // brain::NetworkType, BRAIN_DEFAULT_FALLBACK
    #include "NetworkType.hpp"              // brain::UIType, brain::IType, brain::FType, brain::Value, brain::Values, brain::Dataset, brain::Layers, brain::Rand, brain::Activation, brain::Loss, brain::Score, brain::*Options, brain::TrainState, brain::TrainResult
    #include "Metrics.hpp"                  // brain::Metrics
    #include <filesystem>                   // std::filesystem::path
    #include <optional>                     // std::optional
    #include <ostream>                      // std::ostream
    #include <utility>                      // std::pair, std::forward
    #include <memory>                       // std::unique_ptr, std::shared_ptr, std::make_unique, std::make_shared

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// Handle every kind of network and its training, frozen (utils::type::Freezable) only exec() is allowed
// A setter without parameter (void) is its getter, each training setter edits a field of _options (no copy of its own)
class NeuralNetwork: public utils::type::Freezable {
    private:
        /* configuration */
        brain::TrainOptions _options; // every setting of the training (loss / score / learning rate / seed...), used by train() without options, loss() and evaluate()

        /* training */
        brain::TrainState _state; // the current / last training (learningRate = the decayed one, updates = the steps of the decay)

        /* storage */
        std::unique_ptr<brain::IOptimizer> _optimizer; // SGD when none
        std::shared_ptr<brain::IBackend> _backend = brain::CpuBackend::shared(); // never null, given to the network at each init / load / from / clone (the NeuralNetwork keeps it between two networks)
        std::unique_ptr<brain::INetwork> _network; // nullptr = no network (init<...> first)

        // ---------- Pre-Function -------- //
        void check_(const brain::TrainOptions& options) const; // the score / loss / activation variants match: error when the train can't run (a variant without its vector / value version), warning when it runs but not as expected (ex: Softmax without CrossEntropy)
        void decay_(const brain::TrainOptions& options); // update _state.learningRate from _state.updates

    public:
        // ---------- Pre-Function -------- //
        /* init (clear any existing network, specialized in the .cpp where the network classes are known) */
        template<brain::NetworkType type> requires (type == brain::NetworkType::Layered)
        void init(const brain::UIType layer, const brain::UIType neuron_per_layer, const brain::LayerOptions& options = {}); // the same number of neurons on every layer
        template<brain::NetworkType type> requires (type == brain::NetworkType::Layered)
        void init(const brain::Layers& layers, const brain::LayerOptions& options = {}); // the number of neurons of each layer
        template<brain::NetworkType type> requires (type == brain::NetworkType::Brain)
        void init(const brain::UIType neurons, const brain::BrainOptions& options = {}); // custom network (more like a brain), the shape of the space in the options (dimension / space)
        void weightInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK); // weights at 1.0 by default (not init -> dead network), fallback used on a result that falls on 0
        void biasInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK); // bias at 0.0 by default, fallback used on a result that falls on 0

        /* setter (network, applied to the current one: NoNetwork without one) */
        void size(const brain::UIType input, const brain::UIType output); // number of inputs and outputs (named from 0 to n - 1), 0 = not set: the size is not checked
        void input(const brain::UIType input);
        void output(const brain::UIType output);
        void rand(const brain::FType min, const brain::FType max); // range of the random weights / bias (min <= max)
        void rand(const brain::Rand& rand); // RandInit mode or generator fn(fan_in, fan_out)
        void activation(const brain::Activation& fn); // internal enum, value fn or vector fn (none by default)
        void backend(std::shared_ptr<brain::IBackend> backend); // nullptr = CpuBackend::shared(), stored in _backend and given to the current network (synced first)

        /* getter (network) */
        _nodiscard const std::pair<brain::UIType, brain::UIType>& size(void) const; // <input, output>
        _nodiscard brain::UIType input(void) const;
        _nodiscard brain::UIType output(void) const;
        _nodiscard const brain::Rand& rand(void) const;
        _nodiscard const std::pair<brain::FType, brain::FType>& range(void) const; // <min, max>
        _nodiscard const brain::Activation& activation(void) const;
        _nodiscard std::pair<brain::UIType, brain::UIType> neurons(void) const; // <layers, neurons> of the network (NoNetwork without one)

        /* setter (training, refused when frozen) */
        void options(const brain::TrainOptions& options); // every setting at once
        void loss(const brain::Loss& fn); // internal enum, value fn or vector fn
        void score(const brain::Score& fn);
        void learningRate(const brain::FType learning_rate);
        void decay(const brain::UIType every, const std::pair<bool, brain::FType>& coef = {true, BRAIN_DEFAULT_DECAY_COEF}); // <smooth, coef>, 0 = none
        void stepLimit(const brain::FType limit); // 0 = none
        void seed(const brain::IType seed); // 0 = random
        void epochs(const brain::UIType epochs); // 0 = no limit
        void batchSize(const brain::UIType size); // 1 = each sample, 0 = the whole set
        void shuffle(const bool shuffle);
        void validation(const brain::FType share); // share of the samples kept apart (0 = none)
        void validation(const brain::Dataset& inputs, const brain::Dataset& expected); // the sets are not owned (kept alive by the caller)
        void patience(const brain::UIType epochs); // 0 = never
        void targetLoss(const brain::FType loss); // < 0 = off
        void keepBest(const bool keep);
        void autosave(const std::optional<std::filesystem::path>& path); // save only when a new best network is found
        void hook(std::shared_ptr<brain::ITrainHook> hook); // nullptr = no hook
        void optimizer(std::unique_ptr<brain::IOptimizer> optimizer); // nullptr = SGD

        /* training (auto) */
        void train(const brain::UIType round, const brain::Dataset& inputs, const brain::Dataset& expected_outputs); // shortcut: _options with epochs = round
        // mini-batch: _backend->parallelFor over the samples of the batch, each worker accumulates in its own gradients / caches, applyGradients at the end of the batch

        /* training (manual) */
        void inputs(const brain::Values& inputs); // load the inputs into the network
        _nodiscard brain::Values apply(void); // execute the network with the loaded inputs (keeps what the backpropagation needs)
        void backpropagation(const brain::FType coef); // one coefficient for every output
        void backpropagation(const brain::Values& coefs); // same as above with a special coef for each output
        _nodiscard brain::FType loss(const brain::Value output, const brain::Value expected) const; // compute the loss of one output
        _nodiscard brain::FType loss(const brain::Values& outputs, const brain::Values& expected) const; // compute the mean loss

        /* usage (the only calls allowed when frozen) */
        _nodiscard brain::Values exec(const brain::Values& inputs) const; // doesn't touch the content at all
        _nodiscard brain::Dataset execAll(const brain::Dataset& inputs) const; // exec of each sample, split between the workers of the backend
        _nodiscard brain::FType evaluate(const brain::Dataset& inputs, const brain::Dataset& expected) const; // mean loss, doesn't touch the network
        _nodiscard brain::Metrics metrics(const brain::Dataset& inputs, const brain::Dataset& expected) const;

        /* file (the backend is not saved: a runtime choice, the loaded network gets _backend) */
        void load(const std::filesystem::path& path); // load a network from a file
        void save(const std::filesystem::path& path) const; // save the actual network in a file
        void loadState(const std::filesystem::path& path); // resume a training: the optimizer and the TrainState (apart from the network file)
        void saveState(const std::filesystem::path& path) const; // save the optimizer and the TrainState
        _nodiscard static std::unique_ptr<brain::NeuralNetwork> from(const std::filesystem::path& path); // ctor + load (CpuBackend::shared(), backend() after to change it)

        /* miscellaneous */
        void clear(void); // remove the network and reset the training state (_backend kept)
        _nodiscard std::unique_ptr<brain::NeuralNetwork> clone(void) const; // deep copy (network + config + optimizer), the backend is shared (same threads), not copied

        // ------------ Function ---------- //
        /* setter (built in place) */
        template<class T, class... Args>
        _cold inline void backend(Args&&... args) {this->backend(std::make_shared<T>(std::forward<Args>(args)...));}; // ex: nn.backend<ThreadBackend>(8)
        template<class T, class... Args>
        _cold inline void hook(Args&&... args) {this->hook(std::make_shared<T>(std::forward<Args>(args)...));}; // ex: nn.hook<ProgressHook>(std::cerr)
        template<class T, class... Args>
        _cold inline void optimizer(Args&&... args) {this->optimizer(std::make_unique<T>(std::forward<Args>(args)...));}; // ex: nn.optimizer<Adam>(0.9, 0.999)

        /* getter (training) */
        _cold _nodiscard inline const brain::TrainOptions& options(void) const                   {return this->_options;};
        _cold _nodiscard inline const brain::Loss& loss(void) const                              {return this->_options.loss;};
        _cold _nodiscard inline const brain::Score& score(void) const                            {return this->_options.score;};
        _cold _nodiscard inline brain::FType learningRate(void) const                            {return this->_options.learningRate;};
        _cold _nodiscard inline brain::UIType decay(void) const                                  {return this->_options.decay;};
        _cold _nodiscard inline const std::pair<bool, brain::FType>& decayCoef(void) const       {return this->_options.decayCoef;};
        _cold _nodiscard inline brain::FType stepLimit(void) const                               {return this->_options.stepLimit;};
        _cold _nodiscard inline brain::IType seed(void) const                                    {return this->_options.seed;};
        _cold _nodiscard inline brain::UIType epochs(void) const                                 {return this->_options.epochs;};
        _cold _nodiscard inline brain::UIType batchSize(void) const                              {return this->_options.batchSize;};
        _cold _nodiscard inline bool shuffle(void) const                                         {return this->_options.shuffle;};
        _cold _nodiscard inline brain::FType validation(void) const                              {return this->_options.validation;};
        _cold _nodiscard inline brain::UIType patience(void) const                               {return this->_options.patience;};
        _cold _nodiscard inline brain::FType targetLoss(void) const                              {return this->_options.targetLoss;};
        _cold _nodiscard inline bool keepBest(void) const                                        {return this->_options.keepBest;};
        _cold _nodiscard inline const std::optional<std::filesystem::path>& autosave(void) const {return this->_options.autosave;};
        _cold _nodiscard inline const std::shared_ptr<brain::ITrainHook>& hook(void) const       {return this->_options.hook;};
        _cold _nodiscard inline brain::IOptimizer* optimizer(void) const                         {return this->_optimizer.get();}; // nullptr = SGD
        _cold _nodiscard inline const std::shared_ptr<brain::IBackend>& backend(void) const      {return this->_backend;}; // never null
        _cold _nodiscard inline const brain::TrainState& state(void) const                       {return this->_state;}; // the current / last training

        /* getter (network) */
        _cold _nodiscard inline brain::INetwork* get(void)             {return this->_network.get();}; // the internal network (nullptr without one), the NeuralNetwork keeps it (no move)
        _cold _nodiscard inline const brain::INetwork* get(void) const {return this->_network.get();}; // read only access to the internal network (nullptr without one)

        /* training (auto) */
        template<bool history_enable = false>
        _hot inline brain::TrainResult<history_enable> train(const brain::Dataset& inputs, const brain::Dataset& expected_outputs) {return {};}; // with _options (the setters)

        template<bool history_enable = false>
        _hot inline brain::TrainResult<history_enable> train(const brain::Dataset& inputs, const brain::Dataset& expected_outputs, const brain::TrainOptions& options) {return {};}; // with these options for this call only (_options untouched)

        // ------------ Operator ---------- //
        friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn); // summary: type, layers, neurons, size, activation, loss, backend

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
