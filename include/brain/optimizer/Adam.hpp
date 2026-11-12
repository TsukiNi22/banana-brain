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
##  @file Adam.hpp

File Description:
##  Adam optimizer: a mean and a variance of the gradients per weight, corrected by the steps done
\**************************************************************/

#ifndef ADAM_H
    #define ADAM_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // BRAIN_DEFAULT_BETA1, BRAIN_DEFAULT_BETA2, BRAIN_DEFAULT_EPSILON
    #include "../NetworkType.hpp"      // brain::UIType, brain::FType, brain::Values, brain::Bytes
    #include "IOptimizer.hpp"          // brain::IOptimizer
    #include <memory>                  // std::unique_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class Adam: public brain::IOptimizer {
    private:
        /* params */
        brain::FType _beta1 = BRAIN_DEFAULT_BETA1; // decay of the mean
        brain::FType _beta2 = BRAIN_DEFAULT_BETA2; // decay of the variance
        brain::FType _epsilon = BRAIN_DEFAULT_EPSILON; // no division by 0

        /* storage */
        brain::Values _m; // mean of the gradients, one per weight / bias
        brain::Values _v; // variance of the gradients, one per weight / bias
        brain::UIType _t = 0; // steps done (bias correction)

    public:
        // ---------- Pre-Function -------- //
        /* storage */
        void init(const brain::UIType weights) override;
        void reset(void) override;

        /* update */
        brain::FType step(const brain::UIType index, const brain::FType gradient, const brain::FType learning_rate) override; // -learning_rate * m^ / (sqrt(v^) + epsilon)

        /* serialization */
        brain::Bytes serialize(void) const override; // the params, m, v and t
        void deserialize(const brain::Bytes& data) override;

        /* copy */
        std::unique_ptr<brain::IOptimizer> clone(void) const override;

        // ------------ Function ---------- //
        /* setter */
        _cold inline void beta1(const brain::FType beta)      {this->_beta1 = beta;};
        _cold inline void beta2(const brain::FType beta)      {this->_beta2 = beta;};
        _cold inline void epsilon(const brain::FType epsilon) {this->_epsilon = epsilon;};

        /* getter */
        _cold _nodiscard inline brain::FType beta1(void) const   {return this->_beta1;};
        _cold _nodiscard inline brain::FType beta2(void) const   {return this->_beta2;};
        _cold _nodiscard inline brain::FType epsilon(void) const {return this->_epsilon;};

        // ------------ Operator ---------- //
        Adam& operator=(const Adam& other) = delete;
        Adam& operator=(Adam&& other) = delete;

        // ---------- Constructor --------- //
        Adam(const brain::FType beta1 = BRAIN_DEFAULT_BETA1, const brain::FType beta2 = BRAIN_DEFAULT_BETA2, const brain::FType epsilon = BRAIN_DEFAULT_EPSILON): _beta1{beta1}, _beta2{beta2}, _epsilon{epsilon} { /* Nothing */ };

        Adam(const Adam& other) = delete;
        Adam(Adam&& other) = delete;

        // ----------- Destructor --------- //
        ~Adam() = default;
};

} // namespace end
#endif /* ADAM_H */
