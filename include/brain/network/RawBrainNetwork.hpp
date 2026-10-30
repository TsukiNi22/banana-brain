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
        utils::system::HybridMemory<brain::RawBrainNeuron> _rawNeurons; // every neuron of the 2d space (gpu when available)
        utils::system::HybridMemory<brain::RawBrainLink> _rawLinks; // [neuron * _cap + slot] the links of every neuron, inward and outward (a free slot is all at 0)
        std::pair<brain::FType, brain::FType> _size = {1.0, 1.0}; // <width, height> of the 2d space (could be applied in 3d in the future)
        std::pair<brain::FType, brain::LinksCapFn> _limit; // <percentage, hook> limits of linking: the smallest of the two is the number of links of a neuron
        bool _customLimit = false; // true: the hook was replaced (can't be saved)
        std::pair<brain::FType, brain::StepsCapFn> _stepsLimit; // <percentage, hook> limits of the time steps: the smallest of the two is the number of steps of the execution
        bool _customSteps = false; // true: the steps hook was replaced (can't be saved)
        std::pair<brain::FType, brain::FType> _weightLimits = {BRAIN_WEIGHT_MIN, BRAIN_WEIGHT_MAX}; // <min, max> of the absolute weight: a link is removed at the min and the nearest free neuron is linked at the max (backpropagation)
        bool _orphans = false; // true: the last links of a neuron can be removed (it can end up alone), false: an input keeps an outward link, the others an inward one
        bool _leftToRight = false; // true: a link only goes from a neuron to a neuron on its right (no loop, the signal always moves to the outputs)
        std::vector<std::vector<brain::UIType>> _near; // the first candidates of each neuron for a link, the nearest first (only the left ones in left to right mode), computed once (init / load): the others are sorted only when these are all used
        std::vector<bool> _nearAll; // true: _near holds every candidate of the neuron
        std::vector<brain::UIType> _inStart; // the inward links of neuron i are the slots _inSlot[_inStart[i]] to _inSlot[_inStart[i + 1] - 1] (rebuilt with the execution copy)
        std::vector<brain::UIType> _inSlot;
        std::vector<brain::UIType> _clones; // [slot of an inward link] the slot of its clone on the sender (rebuilt by sync_ when the links change, empty = to rebuild)
        bool _clonesChanged = false; // a new link changed _clones since its last copy on the gpu
        utils::system::HybridMemory<brain::UIType> _cloneMemory; // the same on the gpu of the network (the update of the weights on the device writes the clones with it)
        std::vector<brain::Values> _sums; // the sum of every neuron at each step of the last apply() on the host (the backpropagation reads them back, empty after a gpu apply)
        brain::UIType _stepsFloor = 0; // the least time steps: BRAIN_STEPS_PATH_MARGIN times the farthest path from an input to an output at the init (0 = none)
        brain::UIType _farLinks = 0; // slots of the cap given at the init to neurons drawn at random among the allowed ones (not the nearest)
        brain::UIType _pathHops = 0; // 0 = off, else at the init every input gets a path of at most this number of links to every output (the missing links are added)
        brain::FType _momentum = 0.0; // step = momentum * last step + gradient (0 = plain gradient)
        brain::FType _stepLimit = 0.0; // each change of a weight or a bias kept in [-limit, limit] (0 = no limit)
        bool _clampWeights = false; // true: the absolute value of a weight and of a bias never goes above weightMax (no saturated neuron that can't learn anymore)
        bool _lowNewLinks = false; // true: a link added at weightMax starts at 5 % of the limits instead of the size of the nearest links (no shock on the outputs)
        bool _directOutput = false; // true: the score reaches the sum of an output as it is, without the slope of its activation (cross-entropy form: a saturated output still learns)
        bool _keepPaths = false; // true: a link is never removed when it takes the last path of an input to an output away
        bool _batch = false; // true: backpropagation() only adds the gradient to the pending one, applyBatch() applies their mean
        brain::FType _adam = 0.0; // 0 = off, else the rate of Adam: each slot moves by about this rate, the direction from the mean of its gradients (replaces the momentum)
        std::vector<brain::FType> _moments; // Adam: <mean, mean of the squares> of the gradients of each slot of link, then of each bias
        brain::FType _rateDecay = 0.0; // the steps of the weights are divided by 1 + rateDecay * (number of updates before) (0 = constant)
        brain::UIType _updates = 0; // updates of the weights since the init (the decay of the rate)
        brain::FType _decayScale = 1.0; // 1 / (1 + rateDecay * updates) of the update in progress
        bool _inputPaths = false; // true: every input keeps a path to at least one output (built at the init, a removal that would cut the last one is put back)
        bool _moveLastLink = false; // true: the last link of a neuron under weightMin is moved to another neuron (a new link, the weak one removed) instead of being kept
        bool _invertedLinks = false; // true: at the limits a link is removed at weightMax and a new one grows at weightMin (the opposite of the default)
        bool _randomLinks = true; // true (default): each neuron gets a random number of links at the init, between minLinks and the cap (false: as many as the cap allows)
        brain::UIType _minLinks = 1; // (1 by default) a link is never removed when it leaves one of its two neurons with this number of links or fewer (also the least of the random init)
        brain::UIType _adamSteps = 0; // Adam: number of updates since the moments started (corrections of the first steps)
        std::vector<std::pair<brain::FType, brain::FType>> _seen; // revive: <min, max> of the last state of each neuron over the executions since the last revive()
        std::vector<brain::FType> _velocity; // momentum: the last step of each slot of link, then of each bias (neurons * cap + neurons)
        std::vector<brain::FType> _pending; // mini-batch: the sum of the gradients not applied yet (same layout as _velocity)
        brain::UIType _pendingCount = 0; // mini-batch: number of gradients in _pending
        std::pair<brain::UIType, brain::UIType> _placed = {0, 0}; // number of inputs and outputs the neurons were placed for
        brain::Dataset _sequence; // inputs of each time step (sequence()), empty = the inputs of inputs() at every step
        std::vector<brain::Values> _states; // training: the state of every neuron at each time step of the last apply() (the first one is the start)

        // ---------- Pre-Function -------- //
        void init_(void) final; // set the memories up: the input neurons on the left edge and the output ones on the right edge (evenly spread), the others placed randomly, each one linked to as many of its nearest neurons as its cap allows (weights at 1.0, bias at 0.0)
        void resized_(void) final; // the first neurons are the inputs and the last ones the outputs: set the neurons up again when their number changes
        brain::UIType linksCap_(void) const; // number of links allowed for a neuron: min(percentage * neurons, hook(neurons)), between 1 and neurons - 1 (0 with a single neuron)
        brain::UIType stepsCap_(void) const; // time steps of the execution: min(percentage * neurons, hook(neurons)), at least 1
        void learn_(const brain::Values& rewards); // backpropagation through the time steps of the last apply(): the scores reinforce (positive) or weaken (negative) the links that made the outputs, then the links are changed (relink_)
        bool relink_(const std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links); // each link of a neuron is removed when its weight is too small, the nearest free neuron is linked when it is too big (false: nothing changed)
        void update_(std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const brain::Values& biasGrad, const brain::Values& weightGrad); // apply a gradient: momentum, step limit and clamp of the weights (the clone of a link keeps the same weight)
        template<typename Activation>
        void forward_(const Activation& activation, const brain::RawBrainNeuron* neurons, const brain::RawBrainLink* links); // the time steps of apply() on the host with this activation function, every state and sum kept (apply() chooses the function once)
        void watch_(void); // revive: the last state of each neuron widens its <min, max> since the last revive()
        brain::FType change_(const brain::UIType at, const brain::FType grad, const bool moving, const brain::FType first, const brain::FType second); // the step of a slot (link, then bias): Adam, else the momentum, then the step limit
        void finish_(std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const bool host); // the links change at the limits, the memories are stored and the reduced copy rebuilt
        void ensurePaths_(const std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const bool every) const; // init: a path of at most _pathHops links from every input to every output (the smallest squared distances, a full neuron gives its farthest link up)
        void near_(const std::vector<brain::RawBrainNeuron>& neurons); // fill _near: the first candidates of each neuron (every other one, or the ones strictly on its left in left to right mode), the nearest first
        std::vector<brain::UIType> candidates_(const brain::UIType owner, const std::vector<brain::RawBrainNeuron>& neurons) const; // every candidate of a neuron, the nearest first (the id breaks the ties)
        bool nearest_(const brain::UIType owner, const std::vector<brain::RawBrainNeuron>& neurons, const std::function<bool(const brain::UIType)>& visit) const; // the candidates in order until visit returns true (then true)
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
        _cold _nodiscard inline brain::FType momentum(void) const  {return this->_momentum;}; // step = momentum * last step + gradient
        _cold _nodiscard inline brain::FType rateDecay(void) const {return this->_rateDecay;}; // decay of the steps of the weights (0 = none)
        _cold _nodiscard inline bool invertedLinks(void) const     {return this->_invertedLinks;}; // true: removed at weightMax, grown at weightMin
        _cold _nodiscard inline bool randomLinks(void) const       {return this->_randomLinks;}; // true: a random number of links per neuron at the init
        _cold _nodiscard inline brain::UIType minLinks(void) const {return this->_minLinks;}; // links a neuron keeps at least
        _cold _nodiscard inline brain::FType adam(void) const      {return this->_adam;}; // 0 = off, else the rate of Adam
        _cold _nodiscard inline brain::FType stepLimit(void) const {return this->_stepLimit;}; // each change of a weight or a bias in [-limit, limit] (0 = none)
        _cold _nodiscard inline bool clampWeights(void) const      {return this->_clampWeights;}; // true: |weight| and |bias| never above weightMax
        _cold _nodiscard inline bool lowNewLinks(void) const       {return this->_lowNewLinks;}; // true: a link added at weightMax starts at 5 % of the limits
        _cold _nodiscard inline bool directOutput(void) const      {return this->_directOutput;}; // true: the score reaches the sum of an output without the slope of its activation
        _cold _nodiscard inline bool keepPaths(void) const         {return this->_keepPaths;}; // true: the last path of an input to an output is never cut
        _cold _nodiscard inline bool batch(void) const override    {return this->_batch;}; // true: backpropagation() accumulates, applyBatch() applies
        _cold _nodiscard inline brain::UIType pending(void) const  {return this->_pendingCount;}; // mini-batch: gradients given since the last applyBatch()

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
