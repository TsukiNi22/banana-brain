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
##  @file Momentum.hpp

File Description:
##  Gradient descent with momentum: a velocity per weight keeps the direction of the last updates
\**************************************************************/

#ifndef MOMENTUM_H
    #define MOMENTUM_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // BRAIN_DEFAULT_MOMENTUM
    #include "../NetworkType.hpp"      // brain::UIType, brain::FType, brain::Values, brain::Bytes
    #include "IOptimizer.hpp"          // brain::IOptimizer
    #include <memory>                  // std::unique_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class Momentum: public brain::IOptimizer {
    private:
        /* params */
        brain::FType _momentum = BRAIN_DEFAULT_MOMENTUM;

        /* storage */
        brain::Values _velocity; // one per weight / bias

    public:
        // ---------- Pre-Function -------- //
        /* storage */
        void init(const brain::UIType weights) override;
        void reset(void) override;

        /* update */
        brain::FType step(const brain::UIType index, const brain::FType gradient, const brain::FType learning_rate) override; // velocity = momentum * velocity - learning_rate * gradient

        /* serialization */
        brain::Bytes serialize(void) const override; // the params and the velocities
        void deserialize(const brain::Bytes& data) override;

        /* copy */
        std::unique_ptr<brain::IOptimizer> clone(void) const override;

        // ------------ Function ---------- //
        /* setter */
        _cold inline void momentum(const brain::FType momentum) {this->_momentum = momentum;};

        /* getter */
        _cold _nodiscard inline brain::FType momentum(void) const {return this->_momentum;};

        // ------------ Operator ---------- //
        Momentum& operator=(const Momentum& other) = delete;
        Momentum& operator=(Momentum&& other) = delete;

        // ---------- Constructor --------- //
        Momentum(const brain::FType momentum = BRAIN_DEFAULT_MOMENTUM): _momentum{momentum} { /* Nothing */ };

        Momentum(const Momentum& other) = delete;
        Momentum(Momentum&& other) = delete;

        // ----------- Destructor --------- //
        ~Momentum() = default;
};

} // namespace end
#endif /* MOMENTUM_H */
