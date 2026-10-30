/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 19/10/2026 by @author Tsukini

File Name:
##  @file activation.hpp

File Description:
##  Tools of the internal activation functions: sigmoid, softmax, linear and ReLU
\**************************************************************/

#ifndef ACTIVATION_H
    #define ACTIVATION_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _hot, _nodiscard
    #include "../NetworkType.hpp"   // brain::FType, brain::Values
    #include <functional>           // std::function
    #include <cstdint>              // std::int32_t, std::uint32_t
    #include <cmath>                // std::exp
    #include <bit>                  // std::bit_cast

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* activation (brain::ActivationFunction) */
_hot _nodiscard inline brain::FType sigmoid(const brain::FType value) {return static_cast<brain::FType>(1.0) / (static_cast<brain::FType>(1.0) + std::exp(-value));}; // 1 / (1 + e^-value)
//_hot _nodiscard brain::Values softmax(const brain::Values& values); // e^value / sum of every e^value (the max is removed first to avoid an overflow)
_hot _nodiscard inline brain::FType linear(const brain::FType value) {return value;}; // value unchanged
_hot _nodiscard inline brain::FType relu(const brain::FType value)   {return value > 0 ? value : static_cast<brain::FType>(0.0);}; // max(0, value)

// e^value for a float with no call to the math library (a loop over an array is vectorized): range reduction to [-ln2 / 2, ln2 / 2], a polynomial of degree 6, then the power of two by the exponent bits; about 1 ulp of error (e^-87 and under, e^88 and over give the value of the clamp)
_hot _nodiscard inline brain::FType fastExp(const brain::FType value)
{
    const brain::FType x = value < static_cast<brain::FType>(-87.0) ? static_cast<brain::FType>(-87.0) : (value > static_cast<brain::FType>(88.0) ? static_cast<brain::FType>(88.0) : value);
    const brain::FType scaled = x * static_cast<brain::FType>(1.44269504088896341); // x / ln2
    const std::int32_t whole = static_cast<std::int32_t>(scaled < 0 ? scaled - static_cast<brain::FType>(0.5) : scaled + static_cast<brain::FType>(0.5)); // nearest integer
    const brain::FType n = static_cast<brain::FType>(whole);
    const brain::FType r = (x - n * static_cast<brain::FType>(0.693359375)) - n * static_cast<brain::FType>(-2.12194440e-4); // x - n ln2, in two parts (the product stays exact)
    brain::FType p = static_cast<brain::FType>(1.9875691500e-4);

    p = p * r + static_cast<brain::FType>(1.3981999507e-3);
    p = p * r + static_cast<brain::FType>(8.3334519073e-3);
    p = p * r + static_cast<brain::FType>(4.1665795894e-2);
    p = p * r + static_cast<brain::FType>(1.6666665459e-1);
    p = p * r + static_cast<brain::FType>(5.0000001201e-1);
    return (p * r * r + r + static_cast<brain::FType>(1.0)) * std::bit_cast<brain::FType>(static_cast<std::uint32_t>((whole + 127) << 23));
}
_hot _nodiscard inline brain::FType fastSigmoid(const brain::FType value) {return static_cast<brain::FType>(1.0) / (static_cast<brain::FType>(1.0) + brain::tools::fastExp(-value));}; // 1 / (1 + e^-value) with fastExp

/* the same functions with their derivative, as types: a loop chooses one once (ANetwork::withActivation_) and the compiler inlines it, instead of a test and a call for every neuron */
struct SigmoidActivation {
    _hot _nodiscard inline brain::FType operator()(const brain::FType value) const                      {return brain::tools::fastSigmoid(value);}; // the fast one: a loop over the neurons is vectorized
    _hot _nodiscard inline brain::FType derivative(const brain::FType, const brain::FType output) const {return output * (static_cast<brain::FType>(1.0) - output);}; // from the output (the activation of the sum)
};
struct ReluActivation {
    _hot _nodiscard inline brain::FType operator()(const brain::FType value) const                   {return brain::tools::relu(value);};
    _hot _nodiscard inline brain::FType derivative(const brain::FType sum, const brain::FType) const {return sum > 0 ? static_cast<brain::FType>(1.0) : static_cast<brain::FType>(0.0);};
};
struct LinearActivation {
    _hot _nodiscard inline brain::FType operator()(const brain::FType value) const               {return value;};
    _hot _nodiscard inline brain::FType derivative(const brain::FType, const brain::FType) const {return static_cast<brain::FType>(1.0);};
};
struct ExternalActivation { // a function of the user: its derivative is estimated around the sum
    const brain::ActivationFn& fn;
    _hot _nodiscard inline brain::FType operator()(const brain::FType value) const                                {return this->fn(value);};
    _hot _nodiscard inline brain::FType derivative(const brain::FType sum, const brain::FType) const
    {
        const brain::FType step = static_cast<brain::FType>(1e-3);

        return (this->fn(sum + step) - this->fn(sum - step)) / (static_cast<brain::FType>(2.0) * step);
    };
};

} // namespace end
#endif /* ACTIVATION_H */
