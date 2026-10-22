/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 22/10/2026 by @author Tsukini

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
    #define _System
    #include <utils/utils.hpp>   // _alignas, _hot, _nodiscard, utils::type::OVector2, std::hardware_destructive_interference_size
    #include <type_traits>       // std::is_floating_point_v, std::is_integral_v, std::is_signed_v, std::is_unsigned_v, std::is_trivially_copyable_v
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
using Bytes = std::vector<std::uint8_t>; // raw data of a saved network
using Values = std::vector<brain::FType>; // inputs / outputs / coefficients of one sample
using Dataset = std::vector<brain::Values>; // several samples

/* hook */
using LinksCapFn = std::function<brain::UIType(const brain::UIType n)>; // external links limit computing function
using ActivationFn = std::function<brain::FType(const brain::FType value)>; // external activation function
using LossFn = std::function<brain::FType(const brain::FType output, const brain::FType expected)>; // external loss function
using RandFn = std::function<brain::FType(const brain::IType seed)>; // weight / bias generator from the seed

//----------------------------------------------------------------//
/* STRUCT */

// Neuron of a layered network, aligned on the cache line: no false sharing when the neurons are edited concurrently (cpu / gpu) during the backpropagation
// Trivially copyable (stored in a utils::system::HybridMemory), its weights are in the weights memory of the network
struct _alignas(std::hardware_destructive_interference_size) LayerNeuron {
    brain::UIType incoming = 0; // index of the first weight in the weights memory (one weight per input of the neuron)
    brain::UIType count = 0; // number of weights
    brain::FType bias = 0.0;
};

// Link of a brain network neuron (trivially copyable: stored in a utils::system::HybridMemory)
struct BrainLink { // can be found 2 time, one on each neuron using this link (for execution '->' and backpropagation '<-')
    brain::UIType id = 0; // id of the other neuron
    brain::FType weight = 0.0; // weight of the links
    bool inward = false; // does the link is an input from another neuron (true) or only used for backpropagation (false)
};

// Neuron of a brain network (2d space), same alignment as brain::LayerNeuron
// Trivially copyable (stored in a utils::system::HybridMemory), its links and distances are in the memories of the network
struct _alignas(std::hardware_destructive_interference_size) BrainNeuron {
    utils::type::OVector2<brain::FType> pos; // position in the 2d space of the network
    brain::FType bias = 0.0;

    _hot _nodiscard inline brain::FType distance2(const BrainNeuron& other) const {return (this->pos - other.pos).lengthSquared();}; // squared distance between the positions (no sqrt: only the order matters, every distance stays squared and comparable)
    _hot _nodiscard inline brain::FType operator|(const BrainNeuron& other) const {return this->distance2(other);}; // squared distance (same as distance2)

    // Copy only (defaulted): without move operations, the non trivial move assignment of utils::type::OVector2 doesn't make the neuron non trivially copyable
    BrainNeuron() = default;
    BrainNeuron(const BrainNeuron& other) = default;
    BrainNeuron& operator=(const BrainNeuron& other) = default;
};
static_assert(std::is_trivially_copyable_v<brain::BrainLink> && std::is_trivially_copyable_v<brain::BrainDistance>, "the links and the distances are stored in a HybridMemory");
static_assert(std::is_trivially_copyable_v<brain::LayerNeuron> && std::is_trivially_copyable_v<brain::BrainNeuron>, "the neurons are stored in a HybridMemory");

} // namespace end
#endif /* NETWORKTYPE_H */
