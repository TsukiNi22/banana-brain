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
##  @file IOptimizer.hpp

File Description:
##  Interface of the optimizers: how a gradient becomes the change of a weight
\**************************************************************/

#ifndef IOPTIMIZER_H
    #define IOPTIMIZER_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Bytes
    #include <memory>               // std::unique_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class IOptimizer {
    public:
        // ---------- Pre-Function -------- //
        /* storage */
        virtual void init(const brain::UIType weights) = 0; // allocate the storage (one slot per weight / bias)
        virtual void reset(void) = 0; // every slot back to its start

        /* update */
        virtual brain::FType step(const brain::UIType index, const brain::FType gradient, const brain::FType learning_rate) = 0; // the change to apply to the weight index

        /* serialization (resume a training: NeuralNetwork::saveState / loadState, apart from the network file) */
        virtual brain::Bytes serialize(void) const = 0;
        virtual void deserialize(const brain::Bytes& data) = 0;

        /* copy */
        virtual std::unique_ptr<brain::IOptimizer> clone(void) const = 0; // deep copy (NeuralNetwork::clone)

        // ------------ Operator ---------- //
        IOptimizer& operator=(const IOptimizer& other) = delete;
        IOptimizer& operator=(IOptimizer&& other) = delete;

        // ---------- Constructor --------- //
        IOptimizer() = default;
        IOptimizer(const IOptimizer& other) = delete;
        IOptimizer(IOptimizer&& other) = delete;

        // ----------- Destructor --------- //
        virtual ~IOptimizer() = default;
};

} // namespace end
#endif /* IOPTIMIZER_H */
