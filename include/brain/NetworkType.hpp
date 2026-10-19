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
##  @file NetworkType.hpp

File Description:
##  Value types, data / hook aliases and the neuron structs of the neural network
\**************************************************************/

#ifndef NETWORKTYPE_H
    #define NETWORKTYPE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _CustomType
    #include <utils/utils.hpp>   // _alignas, _hot, _nodiscard, utils::type::OVector2, std::hardware_destructive_interference_size
    #include <type_traits>       // std::is_floating_point_v, std::is_integral_v, std::is_signed_v, std::is_unsigned_v
    #include <functional>        // std::function
    #include <cstdint>           // std::int64_t
    #include <cstddef>           // std::size_t
    #include <utility>           // std::pair
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
using Values = std::vector<brain::FType>; // inputs / outputs / coefficients of one sample
using Dataset = std::vector<brain::Values>; // several samples

/* hook */
using ActivationFn = std::function<brain::FType(const brain::FType value)>; // external activation function
using LossFn = std::function<brain::FType(const brain::FType output, const brain::FType expected)>; // external loss function
using RandFn = std::function<brain::FType(const brain::IType seed)>; // weight / bias generator from the seed

//----------------------------------------------------------------//
/* STRUCT */

// Neuron of a layered network, aligned on the cache line: no false sharing when the neurons are edited concurrently (cpu / gpu) during the backpropagation
struct _alignas(std::hardware_destructive_interference_size) LayerNeuron {
    std::vector<brain::FType> incoming; // weight of each input (one per neuron of the previous layer)
    brain::FType bias = 0.0;
};

// Neuron of a brain network (2d space), same alignment as brain::LayerNeuron
struct _alignas(std::hardware_destructive_interference_size) BrainNeuron {
    std::vector<std::pair<brain::FType, brain::UIType>> incoming; // <weight, id of the incoming neuron linked to this weight>
    std::vector<std::pair<brain::UIType, brain::FType>> ordered; // every other neuron as <id, squared distance>, closest first once set up (brain::tools::sort)
    utils::type::OVector2<brain::FType> pos; // position in the 2d space of the network
    brain::UIType id = 0; // identifier of the neuron (unique in its network)
    brain::FType bias = 0.0;

    _hot _nodiscard brain::FType distance2(const BrainNeuron& other) const // squared distance between the positions (no sqrt: only the order matters, every distance stays squared and comparable)
    {return (this->pos - other.pos).lengthSquared();};
    _hot _nodiscard inline brain::FType operator|(const BrainNeuron& other) const {return this->distance2(other);}; // squared distance (same as distance2)
    _hot _nodiscard inline bool operator==(const BrainNeuron& other) const        {return this->id == other.id;}; // same neuron: ids only
};

} // namespace end
#endif /* NETWORKTYPE_H */
