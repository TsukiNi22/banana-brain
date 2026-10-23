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
##  @file compute.hpp

File Description:
##  Computing of the networks on the gpu that stores their memories (CUDA driver + NVRTC loaded at runtime)
\**************************************************************/

#ifndef COMPUTE_H
    #define COMPUTE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _System
    #include <utils/utils.hpp>      // _hot, _cold, _nodiscard, utils::system::HybridMemory, utils::system::IGpu
    #include "../NetworkDefine.hpp" // brain::ActivationFunction
    #include "../NetworkType.hpp"   // brain::FType, brain::UIType, brain::Values, brain::Layers
    #include <vector>               // std::vector

namespace brain::tools::gpu { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* availability */
_cold _nodiscard bool available(const utils::system::IGpu* handler); // the memories are stored by a CUDA gpu and the driver and NVRTC are loaded (else the caller computes on the host)

/* computing (false: nothing computed, the caller computes on the host; the pointers are the ones of the memories stored on the gpu) */
_hot bool layers(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Values& inputs, std::vector<brain::Values>& sums, std::vector<brain::Values>& outputs); // layer after layer: every sum and every output of each layer
_hot bool steps(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const bool raw, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::UIType steps, const brain::ActivationFunction activation, const brain::FType threshold, const brain::Values& inputs, std::vector<brain::Values>& states, const bool keep); // time steps of a brain network (raw: the RawBrain structs, only the inward links): the last state, or every state when keep
_hot bool layersBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Values& inputs, const std::vector<brain::Values>& sums, const std::vector<brain::Values>& outputs, const brain::Values& errors); // classic backpropagation of the error of each output, the weights and the biases are changed on the device (the sums and outputs are the ones of the forward pass)
_hot bool stepsBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::ActivationFunction activation, const std::vector<brain::Values>& states, const brain::Values& rewards); // backpropagation of the scores of the outputs through the time steps of a raw brain network, the weights (the clones too) and the biases are changed on the device
_hot bool triggered(const utils::system::IGpu* handler, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::FType low, const brain::FType high, bool& any); // does a link of the raw memory have an absolute weight at or under low, or at or over high (any), without reading the links back
_hot bool reduce(const utils::system::IGpu* handler, const utils::system::GpuBuffer rawNeurons, const utils::system::GpuBuffer rawLinks, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap); // rebuild the reduced memories (biases and inward links) from the raw ones, on the device

} // namespace end
#endif /* COMPUTE_H */
