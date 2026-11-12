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
##  @file SGD.hpp

File Description:
##  Stochastic gradient descent: the default optimizer (no storage)
\**************************************************************/

#ifndef SGD_H
    #define SGD_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkType.hpp"   // brain::UIType, brain::FType, brain::Bytes
    #include "IOptimizer.hpp"       // brain::IOptimizer
    #include <memory>               // std::unique_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class SGD: public brain::IOptimizer {
    public:
        // ---------- Pre-Function -------- //
        /* storage */
        void init(const brain::UIType weights) override; // nothing to store
        void reset(void) override; // nothing to reset

        /* update */
        brain::FType step(const brain::UIType index, const brain::FType gradient, const brain::FType learning_rate) override; // -learning_rate * gradient

        /* serialization */
        brain::Bytes serialize(void) const override; // nothing to save
        void deserialize(const brain::Bytes& data) override;

        /* copy */
        std::unique_ptr<brain::IOptimizer> clone(void) const override;

        // ------------ Operator ---------- //
        SGD& operator=(const SGD& other) = delete;
        SGD& operator=(SGD&& other) = delete;

        // ---------- Constructor --------- //
        SGD() = default;
        SGD(const SGD& other) = delete;
        SGD(SGD&& other) = delete;

        // ----------- Destructor --------- //
        ~SGD() = default;
};

} // namespace end
#endif /* SGD_H */
