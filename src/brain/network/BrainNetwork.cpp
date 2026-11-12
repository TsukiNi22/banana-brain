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
##  @file BrainNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/network/BrainNetwork.hpp"
#include <utility>
#include <variant>
#include <cstdint>
#include <memory>
#include <vector>

brain::BrainNetwork::BrainNetwork(const brain::UIType neuron, const brain::BrainOptions& options)
: _neuron{neuron}, _options{options}
{
    /* Nothing */
}

_cold void brain::BrainNetwork::init_(void)
{
    /* Nothing */
}

_cold brain::Bytes brain::BrainNetwork::serialize(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::BrainNetwork::deserialize(const brain::Bytes& data)
{
    /* Nothing */
}

_hot brain::Values brain::BrainNetwork::apply(void)
{
    /* Nothing */
    return {};
}

_hot void brain::BrainNetwork::backpropagation(const brain::FType coef)
{
    /* Nothing */
}

_hot void brain::BrainNetwork::backpropagation(const brain::Values& coefs)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::weightInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::biasInit(const brain::FType fallback)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::options(const brain::BrainOptions& options)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::links(const std::variant<brain::FType, brain::LinksCapFn>& links)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::minLinks(const brain::UIType links)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::farLinks(const brain::UIType links)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::pathHops(const brain::UIType hops)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::steps(const std::variant<brain::FType, brain::StepsCapFn>& steps)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::limits(const brain::FType weight_min, const brain::FType weight_max)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::threshold(const brain::FType threshold)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::dimension(const std::uint16_t dimension)
{
    /* Nothing */
}

_cold void brain::BrainNetwork::space(const std::vector<brain::FType>& space)
{
    /* Nothing */
}

_cold bool brain::BrainNetwork::validate(void) const
{
    /* Nothing */
    return false;
}

_cold void brain::BrainNetwork::prune(const brain::FType influence, const brain::PruneTarget target)
{
    /* Nothing */
}

_cold std::unique_ptr<brain::INetwork> brain::BrainNetwork::clone(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::BrainNetwork::reset(void)
{
    /* Nothing */
}

_cold std::pair<brain::UIType, brain::UIType> brain::BrainNetwork::neurons(void) const
{
    /* Nothing */
    return {};
}
