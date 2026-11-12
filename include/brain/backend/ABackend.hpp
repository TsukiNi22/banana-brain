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
##  @file ABackend.hpp

File Description:
##  Abstract backend: the common state and a plain cpu default of every method (a child only overrides what it speeds up)
\**************************************************************/

#ifndef ABACKEND_H
    #define ABACKEND_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _hot, _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::NetworkType, BRAIN_DEFAULT_GRAIN
    #include "../NetworkType.hpp"      // brain::UIType, brain::FType, brain::Slice, brain::CSlice, brain::Dataset, brain::BrainLink, brain::RawBrainLink
    #include "IBackend.hpp"            // brain::IBackend, brain::INetwork
    #include <functional>              // std::function
    #include <span>                    // std::span

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

// type() stays pure: CpuBackend / ThreadBackend
class ABackend: public brain::IBackend {
    protected:
        /* state */
        bool _ready = false;
        brain::UIType _threads = 1; // workers used by parallelFor
        brain::UIType _grain = BRAIN_DEFAULT_GRAIN; // under n items per worker parallelFor stays on the calling thread

    public:
        // ---------- Pre-Function -------- //
        /* parallel */
        void parallelFor(const brain::UIType count, const std::function<void(const brain::UIType begin, const brain::UIType end, const brain::UIType worker)>& fn) override; // fn(0, count, 0) on the calling thread

        /* compute (plain loops: the reference code, also used by the tests to check the other backends) */
        void matVec(const brain::CSlice weights, const brain::CSlice inputs, const brain::Slice outputs, const brain::UIType rows, const brain::UIType cols) override;
        void outer(const brain::CSlice deltas, const brain::CSlice inputs, const brain::Slice gradients) override;
        void axpy(const brain::FType a, const brain::CSlice x, const brain::Slice y) override;
        void propagate(std::span<const brain::RawBrainLink> links, const brain::UIType cap, const brain::CSlice inputs, const brain::Slice outputs) override;
        void propagate(std::span<const brain::BrainLink> links, const brain::UIType cap, const brain::CSlice inputs, const brain::Slice outputs) override;

        /* batch */
        brain::Dataset applyAll(const brain::INetwork& network, const brain::Dataset& inputs) override; // parallelFor over the samples + network.apply(inputs[i]) (apply const: no cache touched)

        // ------------ Function ---------- //
        /* lifecycle */
        _cold void init(void) override    {this->_ready = true;};
        _cold void release(void) override {this->_ready = false;};
        _cold void sync(void) override    { /* Nothing */ }; // nothing (everything is done at the return)

        /* info */
        _cold _nodiscard brain::UIType threads(void) const override                  {return this->_threads;};
        _cold _nodiscard bool ready(void) const override                             {return this->_ready;};
        _cold _nodiscard bool supports(const brain::NetworkType type) const override {return true;}; // every network

        /* setter */
        _cold inline void grain(const brain::UIType grain) {this->_grain = grain;};

        /* getter */
        _cold _nodiscard inline brain::UIType grain(void) const {return this->_grain;};

        // ------------ Operator ---------- //
        ABackend& operator=(const ABackend& other) = delete;
        ABackend& operator=(ABackend&& other) = delete;

        // ---------- Constructor --------- //
        ABackend() = default;
        ABackend(const ABackend& other) = delete;
        ABackend(ABackend&& other) = delete;

        // ----------- Destructor --------- //
        ~ABackend() { /* Nothing */ }; // release() when still ready
};

} // namespace end
#endif /* ABACKEND_H */
