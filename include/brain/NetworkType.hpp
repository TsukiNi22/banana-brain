/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 10/11/2026 by @author Tsukini

File Name:
##  @file NetworkType.hpp

File Description:
##  Value types, data / hook aliases and the neuron structs of the neural network
\**************************************************************/

#ifndef BRAIN_NETWORKTYPE_H
    #define BRAIN_NETWORKTYPE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _CustomType
    #include <utils/utils.hpp>   // _hot, _nodiscard, utils::type::OVector2
    #include <type_traits>       // std::is_floating_point_v, std::is_integral_v, std::is_signed_v, std::is_unsigned_v
    #include <functional>        // std::function
    #include <cstdint>           // std::int64_t, std::uint8_t
    #include <cstddef>           // std::size_t
    #include <vector>            // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* TYPEDEF */

/* values */
using IType = std::int64_t; // signed integer type (seeds)
using UIType = std::size_t; // unsigned integer type (sizes, counts, ids)
using FType = float; // floating type of every neuron / coefficient value (float, double, ...)
static_assert(std::is_integral_v<brain::IType> && std::is_signed_v<brain::IType>, "IType must be a signed integral type");
static_assert(std::is_integral_v<brain::UIType> && std::is_unsigned_v<brain::UIType>, "UIType must be an unsigned integral type");
static_assert(std::is_floating_point_v<brain::FType>, "FType must be a floating point type");

/* data */
using Layers = std::vector<brain::UIType>; // number of neurons of each layer
using Bytes = std::vector<std::uint8_t>; // raw data of a saved network
using Values = std::vector<brain::FType>; // inputs / outputs / coefficients of one sample
using Dataset = std::vector<brain::Values>; // several samples

/* hook */
using ActivationFn = std::function<brain::FType(const brain::FType value)>; // external activation function
using LossFn = std::function<brain::FType(const brain::FType output, const brain::FType expected)>; // external loss function
using RandFn = std::function<brain::FType(const brain::IType seed)>; // weight / bias generator from the seed

//----------------------------------------------------------------//
/* STRUCT */

// Neuron of a layered network, its weights are in the weights of the network
struct LayerNeuron {
    brain::UIType incoming = 0; // index of the first weight of the neuron (one weight per input of the neuron)
    brain::UIType count = 0; // number of weights
    brain::FType bias = 0.0;
};

// Input link of a reduced brain network neuron: the output of the neuron id times the weight (a free slot is all at 0: no effect)
struct BrainLink {
    brain::UIType id = 0; // id of the neuron sending its output
    brain::FType weight = 0.0;
};

// Neuron of a reduced brain network (execution only): the bias, its input links are in the links of the network
struct BrainNeuron {
    brain::FType bias = 0.0;
};

// Link of a raw brain network neuron, a link between two neurons is found 2 times (the receiver '<-' for the execution, the sender '->' a clone for the backpropagation)
struct RawBrainLink {
    brain::UIType id = 0; // id of the other neuron
    brain::FType weight = 0.0; // weight of the link
    bool inward = false; // true: input from the other neuron (execution) | false: output to the other neuron (only used for the backpropagation)
};

// Neuron of a raw brain network (execution and training), placed in a 2d space
struct RawBrainNeuron {
    utils::type::OVector2<brain::FType> pos; // position in the 2d space of the network
    brain::FType bias = 0.0;

    _hot _nodiscard inline brain::FType distance2(const RawBrainNeuron& other) const {return (this->pos - other.pos).lengthSquared();}; // squared distance between the positions (no sqrt: only the order matters)
    _hot _nodiscard inline brain::FType operator|(const RawBrainNeuron& other) const {return this->distance2(other);}; // squared distance (same as distance2)

    // Copy only (defaulted): without move operations, the non trivial move assignment of utils::type::OVector2 doesn't make the neuron non trivially copyable
    RawBrainNeuron() = default;
    RawBrainNeuron(const RawBrainNeuron& other) = default;
    RawBrainNeuron& operator=(const RawBrainNeuron& other) = default;
};

} // namespace end
#endif /* BRAIN_NETWORKTYPE_H */
