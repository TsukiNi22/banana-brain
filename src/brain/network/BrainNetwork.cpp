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
##  @file BrainNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#define _System
#include <utils/utils.hpp>
#include "brain/network/BrainNetwork.hpp"
#include "brain/NetworkDefine.hpp"
#include "brain/tools/bytes.hpp"
#include <cstdint>
#include <vector>

brain::BrainNetwork::BrainNetwork(const brain::UIType neuron, const brain::UIType cap, const utils::system::MemoryMode mode, const bool setup)
: ANetwork{mode}, _neurons{this->makeMemory_<brain::BrainNeuron>()}, _links{this->makeMemory_<brain::BrainLink>()}, _neuron{neuron}, _cap{cap}
{
    // check inputs
    if (neuron == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the number of neurons should be above 0");
    }
    if (setup) this->init_();
}

brain::BrainNetwork::BrainNetwork(const brain::UIType neuron, const brain::UIType cap, const utils::system::MemoryMode mode)
: BrainNetwork{neuron, cap, mode, true}
{
    /* Nothing */
}

_cold void brain::BrainNetwork::init_(void)
{
    brain::UIType links;

    // check inputs (the memories are allocated in bytes, a count that overflows is refused here)
    if (__builtin_mul_overflow(this->_neuron, this->_cap, &links)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfMemory, "the links of the neurons overflow the size");
    }

    // every neuron at 0 and every slot of link free (all at 0: no effect), the weights come from deserialize or from the reduction of a raw network
    this->_neurons.allocate(this->_neuron);
    this->_neurons.clear();
    this->_links.allocate(links);
    this->_links.clear();
}

_cold void brain::BrainNetwork::serializeBody_(brain::Bytes& out) const
{
    const std::vector<brain::BrainNeuron> neurons = this->_neurons.toVector();
    const std::vector<brain::BrainLink> links = this->_links.toVector();

    brain::tools::put<std::uint64_t>(out, this->_neuron);
    brain::tools::put<std::uint64_t>(out, this->_cap);
    brain::tools::put<std::uint64_t>(out, neurons.size());
    for (const brain::BrainNeuron& neuron: neurons)
        brain::tools::put<brain::FType>(out, neuron.bias);
    brain::tools::put<std::uint64_t>(out, links.size());
    for (const brain::BrainLink& link: links) {
        brain::tools::put<std::uint64_t>(out, link.id);
        brain::tools::put<brain::FType>(out, link.weight);
    }
}

_cold void brain::BrainNetwork::deserializeBody_(const brain::Bytes& data, brain::UIType offset)
{
    const brain::UIType neuron = brain::tools::get<std::uint64_t>(data, offset);
    const brain::UIType cap = brain::tools::get<std::uint64_t>(data, offset);
    brain::UIType count = brain::tools::get<std::uint64_t>(data, offset);

    // the neurons (none: not allocated yet)
    if (neuron == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the number of neurons should be above 0");
    } else if ((count != 0 && count != neuron) || !brain::tools::fits(data, offset, count, sizeof(brain::FType))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid neurons");
    }
    std::vector<brain::BrainNeuron> neurons(count);
    for (brain::BrainNeuron& value: neurons)
        value.bias = brain::tools::get<brain::FType>(data, offset);

    // the links (none: not allocated yet)
    count = brain::tools::get<std::uint64_t>(data, offset);
    if ((count != 0 && (cap == 0 || count / cap != neuron || count % cap != 0)) || !brain::tools::fits(data, offset, count, sizeof(std::uint64_t) + sizeof(brain::FType))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid links");
    }
    std::vector<brain::BrainLink> links(count);
    for (brain::BrainLink& link: links) {
        link.id = brain::tools::get<std::uint64_t>(data, offset);
        link.weight = brain::tools::get<brain::FType>(data, offset);
        if (link.id >= neuron) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "a link goes to the neuron " + std::to_string(link.id) + " of " + std::to_string(neuron));
        }
    }

    if (offset != data.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "unexpected data after the links");
    }

    // everything is valid: the memories are replaced
    this->_neuron = neuron;
    this->_cap = cap;
    this->_neurons.allocate(neurons.size());
    this->_neurons.store(neurons);
    this->_links.allocate(links.size());
    this->_links.store(links);
}

_cold brain::Bytes brain::BrainNetwork::serialize(void) const
{
    brain::Bytes out;

    this->serializeHeader_(out, static_cast<std::uint8_t>(BRAIN_MAGIC_BRAIN));
    this->serializeBody_(out);
    return out;
}

_hot brain::Values brain::BrainNetwork::compute_(const brain::Values& inputs) const
{
    /* Nothing */
    return {};
}
