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
##  @file IBackend.hpp

File Description:
##  Interface of the backends: where / how the compute of a network runs (the networks only describe it)
\**************************************************************/

#ifndef IBACKEND_H
    #define IBACKEND_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkDefine.hpp"   // brain::BackendType, brain::NetworkType
    #include "../NetworkType.hpp"     // brain::UIType, brain::FType, brain::Slice, brain::CSlice, brain::Dataset, brain::BrainLink, brain::RawBrainLink
    #include <functional>             // std::function
    #include <span>                   // std::span

namespace brain { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

class INetwork; // network/INetwork.hpp (applyAll)

//----------------------------------------------------------------//
/* CLASS */

// Every method pure virtual: no thread code in the networks, they only call it
class IBackend {
    public:
        // ---------- Pre-Function -------- //
        /* lifecycle */
        virtual void init(void) = 0; // threads started, called on the first use (lazy)
        virtual void release(void) = 0; // threads stopped
        virtual void sync(void) = 0; // wait for every pending job

        /* info */
        virtual brain::BackendType type(void) const = 0;
        virtual brain::UIType threads(void) const = 0; // workers really used (1 on Cpu), the number of caches / gradients of a network
        virtual bool ready(void) const = 0; // init() done
        virtual bool supports(const brain::NetworkType type) const = 0; // the network can run on it

        /* parallel */
        virtual void parallelFor(const brain::UIType count, const std::function<void(const brain::UIType begin, const brain::UIType end, const brain::UIType worker)>& fn) = 0; // [0, count) split between the workers, worker = index of the worker (its own gradients / caches)

        /* dense (LayerNetwork) */
        virtual void matVec(const brain::CSlice weights, const brain::CSlice inputs, const brain::Slice outputs, const brain::UIType rows, const brain::UIType cols) = 0; // outputs = weights * inputs + bias (rows x (cols + 1): the last column of a row is its bias)
        virtual void outer(const brain::CSlice deltas, const brain::CSlice inputs, const brain::Slice gradients) = 0; // gradients += deltas x inputs (one layer of a backpropagation)
        virtual void axpy(const brain::FType a, const brain::CSlice x, const brain::Slice y) = 0; // y += a * x (gradients sum / update)

        /* sparse (BrainNetwork) */
        virtual void propagate(std::span<const brain::RawBrainLink> links, const brain::UIType cap, const brain::CSlice inputs, const brain::Slice outputs) = 0; // one time step over the inward links (cap slots per neuron) of a raw network
        virtual void propagate(std::span<const brain::BrainLink> links, const brain::UIType cap, const brain::CSlice inputs, const brain::Slice outputs) = 0; // same on the links of a reduced network

        /* batch */
        virtual brain::Dataset applyAll(const brain::INetwork& network, const brain::Dataset& inputs) = 0; // network.apply() of each sample, split between the workers

        // ------------ Operator ---------- //
        IBackend& operator=(const IBackend& other) = delete;
        IBackend& operator=(IBackend&& other) = delete;

        // ---------- Constructor --------- //
        IBackend() = default;
        IBackend(const IBackend& other) = delete;
        IBackend(IBackend&& other) = delete;

        // ----------- Destructor --------- //
        virtual ~IBackend() = default;
};

} // namespace end
#endif /* IBACKEND_H */
