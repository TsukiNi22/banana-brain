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
##  @file ABackend.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/INetwork.hpp"
#include "brain/backend/ABackend.hpp"
#include <functional>
#include <span>

_hot void brain::ABackend::parallelFor(const brain::UIType count, const std::function<void(const brain::UIType begin, const brain::UIType end, const brain::UIType worker)>& fn)
{
    /* Nothing */
}

_hot void brain::ABackend::matVec(const brain::CSlice weights, const brain::CSlice inputs, const brain::Slice outputs, const brain::UIType rows, const brain::UIType cols)
{
    /* Nothing */
}

_hot void brain::ABackend::outer(const brain::CSlice deltas, const brain::CSlice inputs, const brain::Slice gradients)
{
    /* Nothing */
}

_hot void brain::ABackend::axpy(const brain::FType a, const brain::CSlice x, const brain::Slice y)
{
    /* Nothing */
}

_hot void brain::ABackend::propagate(std::span<const brain::RawBrainLink> links, const brain::UIType cap, const brain::CSlice inputs, const brain::Slice outputs)
{
    /* Nothing */
}

_hot void brain::ABackend::propagate(std::span<const brain::BrainLink> links, const brain::UIType cap, const brain::CSlice inputs, const brain::Slice outputs)
{
    /* Nothing */
}

_hot brain::Dataset brain::ABackend::applyAll(const brain::INetwork& network, const brain::Dataset& inputs)
{
    /* Nothing */
    return {};
}
