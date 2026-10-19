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
##  @file BrainNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#include <utils/utils.hpp>
#include "brain/network/BrainNetwork.hpp"
#include "brain/NetworkDefine.hpp"
#include <utility>

brain::BrainNetwork::BrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size)
: _size{size}, _neuron{neuron}
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
    return {static_cast<std::uint8_t>(BRAIN_MAGIC), static_cast<std::uint8_t>(BRAIN_MAGIC_BRAIN)};
}

_cold void brain::BrainNetwork::deserialize(const brain::Bytes& data)
{
    // check the magic bytes
    if (data.size() < 2 || data[0] != static_cast<std::uint8_t>(BRAIN_MAGIC) || data[1] != static_cast<std::uint8_t>(BRAIN_MAGIC_BRAIN)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "bad magic bytes for a brain network");
    }
    /* Nothing */
}

_hot brain::Values brain::BrainNetwork::apply(void) const
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
