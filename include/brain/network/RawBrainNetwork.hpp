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
##  @file RawBrainNetwork.hpp

File Description:
##  Raw custom network (more like a brain): neurons placed in a 2d space linked in both directions, executes and trains (brain::BrainNetwork is its reduced version)
\**************************************************************/

#ifndef RAWBRAINNETWORK_H
    #define RAWBRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _System
    #include <utils/utils.hpp>      // _cold, _nodiscard, utils::system::HybridMemory
    #include "../NetworkDefine.hpp" // BRAIN_LINKS_CAP_PERCENTAGE, BRAIN_LINKS_CAP_COMPUTING
    #include "../tools/compute.hpp" // brain::tools::gpu::Batch
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::RawBrainNeuron, brain::RawBrainLink, brain::LinksCapFn
    #include "BrainNetwork.hpp"     // brain::BrainNetwork
    #include <utility>              // std::pair
    #include <memory>               // std::unique_ptr
    #include <vector>               // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// The reduced network (the memories of brain::BrainNetwork) is the execution copy of this one: kept up to date by sync_() (deserialize, reduce)
class RawBrainNetwork: public brain::BrainNetwork {
    private:
        /* memory */
        utils::system::HybridMemory<brain::RawBrainNeuron> _rawNeurons; // every neuron (gpu when available)
        utils::system::HybridMemory<brain::RawBrainLink> _rawLinks; // [neuron * _cap + slot] inward and outward links (a free slot is all at 0)

        /* space and limits */
        std::pair<brain::FType, brain::FType> _size = {1.0, 1.0}; // <width, height> of the 2d space
        std::pair<brain::FType, brain::LinksCapFn> _limit; // <percentage, hook> of the links per neuron (the smallest wins)
        bool _customLimit = false; // true: the links hook was replaced (not saved)
        std::pair<brain::FType, brain::StepsCapFn> _stepsLimit; // <percentage, hook> of the time steps (the smallest wins)
        bool _customSteps = false; // true: the steps hook was replaced (not saved)
        brain::UIType _stepsFloor = 0; // least time steps: the farthest input -> output path of the init (0 = none)
        std::pair<brain::FType, brain::FType> _weightLimits = {BRAIN_WEIGHT_MIN, BRAIN_WEIGHT_MAX}; // <min, max> of |weight|: removed at the min, a link grows at the max
        std::pair<brain::UIType, brain::UIType> _placed = {0, 0}; // <inputs, outputs> the neurons were placed for

        /* execution copy */
        std::vector<brain::UIType> _inStart; // neuron i: the inward slots _inSlot[_inStart[i]] to _inSlot[_inStart[i + 1] - 1]
        std::vector<brain::UIType> _inSlot; // slots of the inward links, neuron after neuron
        std::vector<brain::UIType> _clones; // [inward slot] the slot of its clone on the sender (empty = to rebuild)
        bool _clonesChanged = false; // true: _clones changed since its last copy on the gpu
        utils::system::HybridMemory<brain::UIType> _cloneMemory; // _clones on the gpu

        /* links at the init */
        std::vector<std::vector<brain::UIType>> _near; // candidates of each neuron, the nearest first (computed once)
        std::vector<bool> _nearAll; // true: _near holds every candidate of the neuron
        brain::UIType _farLinks = 0; // slots given to random far neurons instead of the nearest
        brain::UIType _pathHops = 0; // 0 = off, else a path of at most this many links from every input to every output
        bool _randomLinks = true; // true: a random number of links per neuron, between _minLinks and the cap
        brain::UIType _minLinks = 1; // a neuron keeps at least this many links

        /* rules of the links (true = on) */
        bool _orphans = false; // the last link of a neuron can be removed
        bool _leftToRight = false; // a link only goes to the right (no loop)
        bool _keepPaths = false; // never cut the last path of an input to an output
        bool _inputPaths = false; // every input keeps a path to at least one output
        bool _lowNewLinks = false; // a new link starts at 5 % of the limits
        bool _moveLastLink = false; // the last weak link of a neuron moves to another neuron
        bool _invertedLinks = false; // removed at weightMax, grown at weightMin

        /* training */
        brain::FType _momentum = 0.0; // step = momentum * last step + gradient (0 = plain gradient)
        brain::FType _adam = 0.0; // 0 = off, else the rate of Adam (replaces the momentum)
        brain::FType _stepLimit = 0.0; // a change of a weight or a bias stays in [-limit, limit] (0 = none)
        brain::FType _rateDecay = 0.0; // steps divided by 1 + rateDecay * updates (0 = constant)
        bool _clampWeights = false; // |weight| and |bias| stay under weightMax
        bool _directOutput = false; // the score reaches the sum of an output without the slope of its activation
        bool _batch = false; // backpropagation() accumulates, applyBatch() applies the mean

        /* training state */
        std::vector<brain::FType> _velocity; // momentum: last step of each slot, then of each bias
        std::vector<brain::FType> _moments; // Adam: <mean, mean of the squares> of each slot, then of each bias
        brain::UIType _adamSteps = 0; // Adam: updates since the moments started
        std::vector<brain::FType> _pending; // mini-batch: gradients not applied yet (layout of _velocity)
        brain::UIType _pendingCount = 0; // mini-batch: number of gradients in _pending
        brain::UIType _updates = 0; // updates since the init (for the rate decay)
        brain::FType _decayScale = 1.0; // 1 / (1 + rateDecay * updates) of the update in progress
        std::vector<std::pair<brain::FType, brain::FType>> _seen; // revive: <min, max> of the last state of each neuron

        /* last execution */
        brain::Dataset _sequence; // inputs of each time step (empty = the same inputs at every step)
        std::vector<brain::Values> _states; // state of every neuron at each time step (the first one is the start)
        std::vector<brain::Values> _sums; // sum of every neuron at each step (host only: the backpropagation reads them)

        /* mini-batch (several samples together) */
        std::unique_ptr<brain::tools::gpu::Batch> _device; // the states of the last applyTrain() on the gpu (created when needed)
        bool _deviceKept = false; // the last applyTrain() kept its samples on the device: backpropagationAll() makes their gradient there

        // ---------- Pre-Function -------- //
        void init_(void) final; // set the memories up: the input neurons on the left edge and the output ones on the right edge (evenly spread), the others placed randomly, each one linked to as many of its nearest neurons as its cap allows (weights at 1.0, bias at 0.0)
        void resized_(void) final; // the first neurons are the inputs and the last ones the outputs: set the neurons up again when their number changes
        brain::UIType linksCap_(void) const; // number of links allowed for a neuron: min(percentage * neurons, hook(neurons)), between 1 and neurons - 1 (0 with a single neuron)
        brain::UIType stepsCap_(void) const; // time steps of the execution: min(percentage * neurons, hook(neurons)), at least 1
        void learn_(const brain::Values& rewards); // backpropagation through the time steps of the last apply(): the scores reinforce (positive) or weaken (negative) the links that made the outputs, then the links are changed (relink_)
        bool relink_(const std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links); // each link of a neuron is removed when its weight is too small, the nearest free neuron is linked when it is too big (false: nothing changed)
        void update_(std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const brain::Values& biasGrad, const brain::Values& weightGrad); // apply a gradient: momentum, step limit and clamp of the weights (the clone of a link keeps the same weight)
        void backward_(const std::vector<brain::Values>& states, const std::vector<brain::Values>* sums, const brain::Values& rewards, const brain::RawBrainNeuron* neurons, const brain::RawBrainLink* links, brain::Values& biasGrad, brain::Values& weightGrad) const; // the gradient of one sample from its states (and sums, nullptr = computed again): the scores of the outputs go back in time through the inward links, added to biasGrad and weightGrad (by inward slot)
        template<typename Activation>
        void sampleForward_(const Activation& activation, const brain::RawBrainNeuron* neurons, const brain::RawBrainLink* links, const brain::Values& inputs, std::vector<brain::Values>& states, std::vector<brain::Values>& sums) const; // the time steps of one sample on one thread, every state and sum kept (the buffers are reused when their shape is the same)
        template<typename Activation>
        void gradientAll_(const Activation& activation, const brain::Dataset& inputs, const brain::Dataset& rewards); // the gradient of every sample, shared between threads (a gradient of its own for each thread, added in the order of the threads), added to the pending one
        void watchAll_(const brain::Values& last, const brain::UIType count); // revive: the last state of each neuron of each sample ([neuron * count + sample]) widens its <min, max>
        template<typename Activation>
        void forward_(const Activation& activation, const brain::RawBrainNeuron* neurons, const brain::RawBrainLink* links); // the time steps of apply() on the host with this activation function, every state and sum kept (apply() chooses the function once)
        void watch_(void); // revive: the last state of each neuron widens its <min, max> since the last revive()
        brain::FType change_(const brain::UIType at, const brain::FType grad, const bool moving, const brain::FType first, const brain::FType second); // the step of a slot (link, then bias): Adam, else the momentum, then the step limit
        void finish_(std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const bool host); // the links change at the limits, the memories are stored and the reduced copy rebuilt
        void ensurePaths_(const std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const bool every) const; // init: a path of at most _pathHops links from every input to every output (the smallest squared distances, a full neuron gives its farthest link up)
        void near_(const std::vector<brain::RawBrainNeuron>& neurons); // fill _near: the first candidates of each neuron (every other one, or the ones strictly on its left in left to right mode), the nearest first
        std::vector<brain::UIType> candidates_(const brain::UIType owner, const std::vector<brain::RawBrainNeuron>& neurons) const; // every candidate of a neuron, the nearest first (the id breaks the ties)
        bool nearest_(const brain::UIType owner, const std::vector<brain::RawBrainNeuron>& neurons, const std::function<bool(const brain::UIType)>& visit, const std::vector<brain::UIType>* used = nullptr) const; // the candidates in order until visit returns true (then true); used = the slots taken of each neuron: a full neuron is never visited and not sorted (the order of the others is the same)
        void sync_(const std::vector<brain::RawBrainNeuron>& rawNeurons, const std::vector<brain::RawBrainLink>& rawLinks); // from the raw memories already read // rebuild the reduced memories (biases and inward links) from the raw ones

        // ------------ Function ---------- //
        _cold _nodiscard inline bool scored_(void) const override {return true;}; // the coefficients of its backpropagation are scores (positive = good)
        _cold _nodiscard inline bool hostTraining_(void) const    {return this->_momentum != 0 || this->_adam != 0 || this->_stepLimit != 0 || this->_clampWeights || this->_batch || this->_directOutput || this->_rateDecay != 0;}; // the options only the host applies (the gpu memories are read back)
        _cold inline void sync_(void)                             {this->sync_(this->_rawNeurons.toVector(), this->_rawLinks.toVector());}; // rebuild the reduced memories from the raw ones
        _cold _nodiscard static inline brain::UIType defaultCap_(const brain::UIType n)   {return static_cast<brain::UIType>(BRAIN_LINKS_CAP_COMPUTING(n));}; // the default hook
        _cold _nodiscard static inline brain::UIType defaultSteps_(const brain::UIType n) {return static_cast<brain::UIType>(BRAIN_STEPS_CAP_COMPUTING(n));}; // the default hook of the time steps

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const final;
        void deserialize(const brain::Bytes& data) final;

        /* execution */
        using brain::BrainNetwork::apply;
        void inputs(const brain::Values& inputs) final; // the same inputs at every step (cancels a sequence)
        void sequence(const brain::Dataset& steps); // the inputs of each time step: step t reads steps[t], the inputs are 0 after the last one (a memory: the network keeps what it saw); runs on the host
        brain::Values apply(void) final;
        brain::Dataset applyTrain(const brain::Dataset& inputs) override; // mini-batch: the outputs of the samples together (the gpu keeps their states there in the Fast profile)
        void backpropagationAll(const brain::Dataset& coefs) override; // mini-batch: the gradient of the samples of the last applyTrain() (a sample per thread on the host, or all together on the gpu), added to the pending one
        brain::Values applySequence(const brain::Dataset& steps) const; // execution of a sequence, nothing kept (the outputs after the last step)
        void backpropagation(const brain::FType coef) final;
        void backpropagation(const brain::Values& coefs) final;

        /* setup */
        void weightInit(const brain::FType fallback = 0.01) final;
        void biasInit(const brain::FType fallback = 0.01) final;

        /* setter (weights) */
        void weightLimits(const brain::FType min, const brain::FType max); // both at once, 0 <= min < max

        /* setter (links) */
        void linksCap(const brain::FType percentage); // set the maximum allowed links using the percentage of the neurons (0 < percentage <= 1), sets the neurons and the links up again
        void linksCap(const brain::LinksCapFn& fn); // replace the function to compute the number of allowed links, sets the neurons and the links up again
        void resetLinksCap(void); // reset the links cap used
        void stepsCap(const brain::FType percentage); // set the maximum time steps using the percentage of the neurons (0 < percentage <= 1)
        void stepsCap(const brain::StepsCapFn& fn); // replace the function to compute the time steps
        void resetStepsCap(void); // reset the steps cap used
        void farLinks(const brain::UIType count); // slots of the cap given to random far neurons at the init (0 = only the nearest ones), sets the neurons and the links up again
        void pathInit(const brain::UIType hops); // 0 = off, else (>= 2) a path of at most hops links from every input to every output at the init, sets the neurons and the links up again
        void leftToRight(const bool enable); // true: the links only go from the left to the right (a neuron only receives from the neurons on its left), sets the neurons and the links up again

        /* setter (training) */
        void momentum(const brain::FType value); // 0 <= value < 1
        void adam(const brain::FType rate); // 0 = off, else Adam with this rate (beta 0.9 / 0.999): the step of a slot is about the rate whatever the size of the score, replaces the momentum
        void rateDecay(const brain::FType decay); // decay >= 0: the steps of the weights are divided by 1 + decay * (updates so far), 0 = constant (with batch() an update is a round)
        void inputPaths(const bool enable); // true: every input keeps a path to at least one output (built at the init, never cut by the training), sets the neurons and the links up again
        void setRules(const std::uint32_t rules); // exactly these rules on (brain::Rule, brain::rules::*), the others off: the neurons and the links are set up again once when a rule of the init changed
        void randomLinks(const bool enable); // true: a random number of links per neuron at the init (between minLinks() and the cap), sets the neurons and the links up again
        void minLinks(const brain::UIType count); // a link is never removed when a neuron would keep count links or fewer (sets the neurons and the links up again when randomLinks() is on)
        void stepLimit(const brain::FType limit); // limit >= 0 (0 = none)
        void batch(const bool enable) override; // true: backpropagation() accumulates, applyBatch() applies the mean (the pending gradients are dropped when it changes)
        brain::UIType revive(void); // a neuron stuck at 0 or 1 on every execution since the last call (saturated) gets its bias at 0 and its inward weights halved, returns the number revived (call it at the end of a round)
        void applyBatch(void) override; // mini-batch: apply the mean of the gradients given since the last call, then the links change at the limits (nothing pending: nothing done)

        /* reduction */
        std::unique_ptr<brain::BrainNetwork> reduce(void); // the reduced network (execution only), from the inward links and the biases

        // ------------ Function ---------- //
        /* getter (weights) */
        _cold _nodiscard inline brain::FType weightMin(void) const {return this->_weightLimits.first;}; // a link is removed when the absolute value of its weight falls to it
        _cold _nodiscard inline brain::FType weightMax(void) const {return this->_weightLimits.second;}; // the nearest free neuron is linked when the absolute value of the weight of a link rises to it
        _cold _nodiscard inline bool orphans(void) const           {return this->_orphans;}; // true: the last links of a neuron can be removed
        _cold _nodiscard inline bool leftToRight(void) const       {return this->_leftToRight;};
        _cold _nodiscard inline brain::UIType farLinks(void) const {return this->_farLinks;}; // slots of the cap given to random far neurons at the init
        _cold _nodiscard inline brain::UIType pathInit(void) const {return this->_pathHops;}; // 0 = off, else the max links of the path of every input to every output at the init

        /* getter (training) */
        _cold _nodiscard inline brain::FType momentum(void) const      {return this->_momentum;}; // step = momentum * last step + gradient
        _cold _nodiscard inline brain::FType rateDecay(void) const     {return this->_rateDecay;}; // decay of the steps of the weights (0 = none)
        _cold _nodiscard inline bool invertedLinks(void) const         {return this->_invertedLinks;}; // true: removed at weightMax, grown at weightMin
        _cold _nodiscard inline bool randomLinks(void) const           {return this->_randomLinks;}; // true: a random number of links per neuron at the init
        _cold _nodiscard inline brain::UIType minLinks(void) const     {return this->_minLinks;}; // links a neuron keeps at least
        _cold _nodiscard inline brain::FType adam(void) const          {return this->_adam;}; // 0 = off, else the rate of Adam
        _cold _nodiscard inline brain::FType stepLimit(void) const     {return this->_stepLimit;}; // each change of a weight or a bias in [-limit, limit] (0 = none)
        _cold _nodiscard inline bool clampWeights(void) const          {return this->_clampWeights;}; // true: |weight| and |bias| never above weightMax
        _cold _nodiscard inline bool lowNewLinks(void) const           {return this->_lowNewLinks;}; // true: a link added at weightMax starts at 5 % of the limits
        _cold _nodiscard inline bool directOutput(void) const          {return this->_directOutput;}; // true: the score reaches the sum of an output without the slope of its activation
        _cold _nodiscard inline bool keepPaths(void) const             {return this->_keepPaths;}; // true: the last path of an input to an output is never cut
        _cold _nodiscard inline std::size_t trainingMemory(void) const {return this->_device ? this->_device->bytes() : 0;}; // bytes the mini-batches hold on the device (the states of the last applyTrain())
        _cold _nodiscard bool batchTraining(void) const override       {return this->_batch && this->_sequence.empty() && !this->_activationFunction.second && !this->_rawNeurons.empty() && this->_ioSize.first != 0 && this->_ioSize.second != 0;}; // the samples of a mini-batch are computed together (threads, or the gpu)
        _cold _nodiscard inline bool batch(void) const override        {return this->_batch;}; // true: backpropagation() accumulates, applyBatch() applies
        _cold _nodiscard inline brain::UIType pending(void) const      {return this->_pendingCount;}; // mini-batch: gradients given since the last applyBatch()

        /* setter (training) */
        _cold inline void clampWeights(const bool enable)          {this->_clampWeights = enable;}; // true: |weight| and |bias| kept under weightMax
        _cold inline void directOutput(const bool enable)          {this->_directOutput = enable;}; // true: a saturated output still learns (cross-entropy form of the score)
        _cold inline void lowNewLinks(const bool enable)           {this->_lowNewLinks = enable;}; // true: a link added at weightMax starts at 5 % of the limits
        _cold inline void keepPaths(const bool enable)             {this->_keepPaths = enable;};
        _cold inline void addRules(const std::uint32_t rules)      {this->setRules(this->getRules() | rules);}; // turn these rules on
        _cold inline void subRules(const std::uint32_t rules)      {this->setRules(this->getRules() & ~rules);}; // turn these rules off
        _cold inline void resetRules(void)                         {this->setRules(brain::rules::DEFAULT);}; // the rules of a new network
        _cold inline void clearRules(void)                         {this->setRules(brain::rules::NONE);}; // every rule off
        _cold _nodiscard inline std::uint32_t getRules(void) const {return (this->_orphans ? brain::ORPHANS : 0) | (this->_leftToRight ? brain::LEFT_TO_RIGHT : 0) | (this->_clampWeights ? brain::CLAMP_WEIGHTS : 0) | (this->_lowNewLinks ? brain::LOW_NEW_LINKS : 0) | (this->_directOutput ? brain::DIRECT_OUTPUT : 0) | (this->_keepPaths ? brain::KEEP_PATHS : 0) | (this->_batch ? brain::BATCH : 0) | (this->_randomLinks ? brain::RANDOM_LINKS : 0) | (this->_invertedLinks ? brain::INVERTED_LINKS : 0) | (this->_inputPaths ? brain::INPUT_PATHS : 0) | (this->_moveLastLink ? brain::MOVE_LAST_LINK : 0);}; // the rules on
        _cold _nodiscard inline bool inputPaths(void) const        {return this->_inputPaths;}; // true: every input keeps a path to an output
        _cold _nodiscard inline bool moveLastLink(void) const      {return this->_moveLastLink;}; // true: the last weak link of a neuron is moved
        _cold inline void moveLastLink(const bool enable)          {this->_moveLastLink = enable;}; // true: the last link of a neuron under weightMin is moved to another neuron
        _cold inline void invertedLinks(const bool enable)         {this->_invertedLinks = enable;}; // true: at the limits a link is removed at weightMax and a new one grows at weightMin // true: a link is never removed when it takes the last path of an input to an output away

        /* setter (weights) */
        _cold inline void weightMin(const brain::FType min) {this->weightLimits(min, this->_weightLimits.second);}; // 0 <= min < max
        _cold inline void weightMax(const brain::FType max) {this->weightLimits(this->_weightLimits.first, max);}; // max > min

        /* setter (orphans) */
        _cold inline void allowOrphans(void)    {this->_orphans = true;}; // the backpropagation can remove the last link of a neuron (an input without outward link, an other without inward link, even alone)
        _cold inline void disallowOrphans(void) {this->_orphans = false;}; // default: an input keeps an outward link, the others an inward one

        _cold _nodiscard std::size_t memory(void) const final {return this->brain::BrainNetwork::memory() + this->_rawNeurons.bytes() + this->_rawLinks.bytes();}; // the execution copy and the raw memories

        /* getter (raw memories) */
        _cold _nodiscard inline std::vector<brain::RawBrainNeuron> rawNeurons(void) const {return this->_rawNeurons.toVector();}; // a copy of every neuron (position, bias)
        _cold _nodiscard inline std::vector<brain::RawBrainLink> rawLinks(void) const     {return this->_rawLinks.toVector();}; // a copy of every slot of link ([neuron * cap + slot], a free slot is all at 0)
        _cold _nodiscard inline const std::vector<brain::Values>& states(void) const      {return this->_states;}; // the state of every neuron at each time step of the last apply() (the first one is the start)

        /* getter (links, steps) */
        _cold _nodiscard inline const std::pair<brain::FType, brain::LinksCapFn>& linksCap(void) const {return this->_limit;}; // <percentage, hook> of the links limit
        _cold _nodiscard inline const std::pair<brain::FType, brain::StepsCapFn>& stepsCap(void) const {return this->_stepsLimit;}; // <percentage, hook> of the time steps limit

        // ------------ Operator ---------- //
        RawBrainNetwork& operator=(const RawBrainNetwork& other) = delete;
        RawBrainNetwork& operator=(RawBrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        RawBrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size = {1.0, 1.0}, const utils::system::MemoryMode mode = utils::system::MemoryMode::Auto); // size = 2d space of the neurons, calls init_ at the end
        RawBrainNetwork(const RawBrainNetwork& other) = delete;
        RawBrainNetwork(RawBrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~RawBrainNetwork() = default;
};

} // namespace end
#endif /* RAWBRAINNETWORK_H */
