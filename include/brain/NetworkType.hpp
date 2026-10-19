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
using ActivationFn = std::function<brain::FType(const brain::FType value)>; // external activation function
using LossFn = std::function<brain::FType(const brain::FType output, const brain::FType expected)>; // external loss function
using RandFn = std::function<brain::FType(const brain::IType seed)>; // weight / bias generator from the seed

//----------------------------------------------------------------//
/* STRUCT */

// Link of a brain network neuron (trivially copyable: stored in a utils::system::HybridMemory)
struct BrainLink {
    brain::FType weight = 0.0; // weight of the link
    brain::UIType id = 0; // id of the incoming neuron linked to this weight
};

// Distance of a brain network neuron to another one (trivially copyable: stored in a utils::system::HybridMemory)
struct BrainDistance {
    brain::UIType id = 0; // id of the other neuron
    brain::FType distance = 0.0; // squared distance (no sqrt: only the order matters)
};

// Neuron of a layered network, aligned on the cache line: no false sharing when the neurons are edited concurrently (cpu / gpu) during the backpropagation
// Trivially copyable (stored in a utils::system::HybridMemory), its weights are in the weights memory of the network
struct _alignas(std::hardware_destructive_interference_size) LayerNeuron {
    brain::UIType incoming = 0; // index of the first weight in the weights memory (one weight per input of the neuron)
    brain::UIType count = 0; // number of weights
    brain::FType bias = 0.0;
};

// Neuron of a brain network (2d space), same alignment as brain::LayerNeuron
// Trivially copyable (stored in a utils::system::HybridMemory), its links and distances are in the memories of the network
struct _alignas(std::hardware_destructive_interference_size) BrainNeuron {
    utils::type::OVector2<brain::FType> pos; // position in the 2d space of the network
    brain::UIType incoming = 0; // index of the first brain::BrainLink in the links memory
    brain::UIType incomingCount = 0; // number of links
    brain::UIType ordered = 0; // index of the first brain::BrainDistance in the distances memory (every other neuron, closest first once set up with brain::tools::sort)
    brain::UIType orderedCount = 0; // number of distances
    brain::UIType id = 0; // identifier of the neuron (unique in its network)
    brain::FType bias = 0.0;

    _hot _nodiscard inline brain::FType distance2(const BrainNeuron& other) const        {return (this->pos - other.pos).lengthSquared();}; // squared distance between the positions (no sqrt: only the order matters, every distance stays squared and comparable)
    _hot _nodiscard inline brain::FType operator|(const BrainNeuron& other) const {return this->distance2(other);}; // squared distance (same as distance2)
    _hot _nodiscard inline bool operator==(const BrainNeuron& other) const        {return this->id == other.id;}; // same neuron: ids only

    // Copy only (defaulted): without move operations, the non trivial move assignment of utils::type::OVector2 doesn't make the neuron non trivially copyable
    BrainNeuron() = default;
    BrainNeuron(const BrainNeuron& other) = default;
    BrainNeuron& operator=(const BrainNeuron& other) = default;
};
static_assert(std::is_trivially_copyable_v<brain::BrainLink> && std::is_trivially_copyable_v<brain::BrainDistance>, "the links and the distances are stored in a HybridMemory");
static_assert(std::is_trivially_copyable_v<brain::LayerNeuron> && std::is_trivially_copyable_v<brain::BrainNeuron>, "the neurons are stored in a HybridMemory");

} // namespace end
#endif /* NETWORKTYPE_H */
