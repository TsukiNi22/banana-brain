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
##  @file BrainNetwork.hpp

File Description:
##  Reduced custom network (more like a brain): execution only, made by brain::RawBrainNetwork::reduce()
\**************************************************************/

#ifndef BRAINNETWORK_H
    #define BRAINNETWORK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::NetworkType, brain::PruneTarget, brain::flags, BRAIN_DEFAULT_FALLBACK, BRAIN_DEFAULT_PRUNE
    #include "../NetworkType.hpp"      // brain::UIType, brain::FType, brain::Values, brain::Bytes, brain::BrainNeuron, brain::BrainLink, brain::BrainOptions, brain::LinksCapFn, brain::StepsCapFn
    #include "INetwork.hpp"            // brain::INetwork
    #include "ANetwork.hpp"            // brain::ANetwork
    #include <utility>                 // std::pair
    #include <variant>                 // std::variant
    #include <cstdint>                 // std::uint16_t, std::uint32_t
    #include <memory>                  // std::unique_ptr
    #include <vector>                  // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// The time steps go through _backend->propagate (links)
class BrainNetwork: public brain::ANetwork {
    protected:
        /* structure */
        std::vector<brain::BrainNeuron> _neurons; // every neuron (its bias)
        std::vector<brain::BrainLink> _links; // the input links of every neuron, _cap slots each
        brain::UIType _neuron = 0; // number of neurons
        brain::UIType _cap = 0; // slots of links of each neuron (from _options.links)

        /* configuration */
        brain::BrainOptions _options; // the flags included (_options.flags), edited by the setters below (no copy of its own)

        // ---------- Pre-Function -------- //
        void init_(void) override; // create _neuron neurons with _cap free slots of links each

    public:
        // ---------- Pre-Function -------- //
        /* serialization */
        brain::Bytes serialize(void) const override;
        void deserialize(const brain::Bytes& data) override;

        /* execution */
        using brain::ANetwork::apply;
        brain::Values apply(void) override; // execution only: ReadOnlyNetwork

        /* training */
        void backpropagation(const brain::FType coef) override; // execution only: ReadOnlyNetwork
        void backpropagation(const brain::Values& coefs) override; // execution only: ReadOnlyNetwork

        /* init */
        void weightInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) override; // execution only: ReadOnlyNetwork
        void biasInit(const brain::FType fallback = BRAIN_DEFAULT_FALLBACK) override; // execution only: ReadOnlyNetwork

        /* options (each one edits a field of _options) */
        void options(const brain::BrainOptions& options); // every option at once
        void links(const std::variant<brain::FType, brain::LinksCapFn>& links); // links per neuron: share of the neurons or fn
        void minLinks(const brain::UIType links);
        void farLinks(const brain::UIType links);
        void pathHops(const brain::UIType hops); // 0 = off
        void steps(const std::variant<brain::FType, brain::StepsCapFn>& steps); // time steps: share of the neurons or fn
        void limits(const brain::FType weight_min, const brain::FType weight_max);
        void threshold(const brain::FType threshold);
        virtual void dimension(const std::uint16_t dimension); // after the init: the neurons placed again in the new space (links kept)
        virtual void space(const std::vector<brain::FType>& space); // same (one value = the same size on every axis)
        _nodiscard bool validate(void) const; // the rules of the flags are respected (paths input -> output, at least one link on each neuron, in / out only, space.size() == dimension or 1...)

        /* shape */
        void prune(const brain::FType influence = BRAIN_DEFAULT_PRUNE, const brain::PruneTarget target = brain::PruneTarget::Both) override; // execution only: ReadOnlyNetwork

        /* state */
        std::unique_ptr<brain::INetwork> clone(void) const override;
        void reset(void) override; // execution only: ReadOnlyNetwork

        /* getter */
        std::pair<brain::UIType, brain::UIType> neurons(void) const final; // <0, total neurons> (no layer)

        // ------------ Function ---------- //
        /* flags (like the cli of libutils, on _options.flags) */
        _cold inline void resetFlags(void)                         {this->_options.flags = brain::flags::DEFAULT;};
        _cold inline void subFlags(const std::uint32_t flags)      {this->_options.flags &= ~flags;};
        _cold inline void addFlags(const std::uint32_t flags)      {this->_options.flags |= flags;};
        _cold inline void setFlags(const std::uint32_t flags)      {this->_options.flags = flags;};
        _cold _nodiscard inline std::uint32_t getFlags(void) const {return this->_options.flags;};

        /* getter (options) */
        _cold _nodiscard brain::NetworkType type(void) const override                    {return brain::NetworkType::Brain;};
        _cold _nodiscard inline const brain::BrainOptions& options(void) const           {return this->_options;};
        _cold _nodiscard inline const std::variant<brain::FType, brain::LinksCapFn>& links(void) const {return this->_options.links;};
        _cold _nodiscard inline brain::UIType minLinks(void) const                       {return this->_options.minLinks;};
        _cold _nodiscard inline brain::UIType farLinks(void) const                       {return this->_options.farLinks;};
        _cold _nodiscard inline brain::UIType pathHops(void) const                       {return this->_options.pathHops;};
        _cold _nodiscard inline const std::variant<brain::FType, brain::StepsCapFn>& steps(void) const {return this->_options.steps;};
        _cold _nodiscard inline std::pair<brain::FType, brain::FType> limits(void) const {return {this->_options.weightMin, this->_options.weightMax};}; // <weight min, weight max>
        _cold _nodiscard inline brain::FType threshold(void) const                       {return this->_options.threshold;};
        _cold _nodiscard inline std::uint16_t dimension(void) const                      {return this->_options.dimension;};
        _cold _nodiscard inline const std::vector<brain::FType>& space(void) const       {return this->_options.space;};

        // ------------ Operator ---------- //
        BrainNetwork& operator=(const BrainNetwork& other) = delete;
        BrainNetwork& operator=(BrainNetwork&& other) = delete;

        // ---------- Constructor --------- //
        BrainNetwork(const brain::UIType neuron, const brain::BrainOptions& options = {}); // calls init_ at the end
        BrainNetwork(const BrainNetwork& other) = delete;
        BrainNetwork(BrainNetwork&& other) = delete;

        // ----------- Destructor --------- //
        ~BrainNetwork() = default;
};

} // namespace end
#endif /* BRAINNETWORK_H */
