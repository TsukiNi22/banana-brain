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
##  @file RawBrainNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#define _System
#include <utils/utils.hpp>
#include "brain/network/RawBrainNetwork.hpp"
#include "brain/NetworkDefine.hpp"
#include "brain/tools/neuron.hpp"
#include "brain/tools/random.hpp"
#include "brain/tools/bytes.hpp"
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <vector>
#include <cmath>

brain::RawBrainNetwork::RawBrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size, const utils::system::MemoryMode mode)
: BrainNetwork{neuron, 0, mode, false}, _rawNeurons{this->makeMemory_<brain::RawBrainNeuron>()}, _rawLinks{this->makeMemory_<brain::RawBrainLink>()}, _size{size}, _limit{BRAIN_LINKS_CAP_PERCENTAGE, &brain::RawBrainNetwork::defaultCap_}
{
    // check inputs
    if (!(size.first > 0) || !(size.second > 0)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the size of the space should be above 0");
    }
    this->init_();
}

_cold void brain::RawBrainNetwork::init_(void)
{
    std::vector<brain::RawBrainNeuron> neurons(this->_neuron);
    std::vector<brain::RawBrainLink> links;
    std::vector<brain::UIType> used(this->_neuron, 0); // slots of links used by each neuron
    brain::UIType total;

    // check inputs (the memories are allocated in bytes, a count that overflows is refused here)
    this->_cap = this->linksCap_();
    if (__builtin_mul_overflow(this->_neuron, this->_cap, &total)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfMemory, "the links of the neurons overflow the size");
    }
    links.resize(total);

    // the neurons are placed in the space (with the generator of the library: brain::tools::rand(seed) to choose the seed)
    for (brain::RawBrainNeuron& neuron: neurons)
        neuron.pos = {brain::tools::rand_floating() * this->_size.first, brain::tools::rand_floating() * this->_size.second};

    // each neuron receives from its nearest neurons (they send to it), a link takes a slot on both of them
    for (brain::UIType receiver = 0; receiver < neurons.size(); ++receiver) {
        for (const brain::UIType sender: brain::tools::sort(receiver, neurons, this->_cap)) {
            bool linked = false;

            if (used[receiver] >= this->_cap) break;
            if (used[sender] >= this->_cap) continue;
            for (brain::UIType slot = 0; slot < used[receiver] && !linked; ++slot)
                linked = links[receiver * this->_cap + slot].id == sender;
            if (linked) continue;
            links[receiver * this->_cap + used[receiver]++] = {sender, 1.0, true}; // weight at 1.0 by default: weightInit gives the random ones
            links[sender * this->_cap + used[sender]++] = {receiver, 1.0, false};
        }
    }
    this->_rawNeurons.allocate(neurons.size());
    this->_rawNeurons.store(neurons);
    this->_rawLinks.allocate(links.size());
    this->_rawLinks.store(links);
    this->sync_();
}

_cold brain::UIType brain::RawBrainNetwork::linksCap_(void) const
{
    brain::UIType cap;

    // a single neuron has nobody to link to
    if (this->_neuron < 2) return 0;
    cap = static_cast<brain::UIType>(std::floor(static_cast<double>(this->_neuron) * static_cast<double>(this->_limit.first)));
    if (this->_limit.second) cap = std::min(cap, this->_limit.second(this->_neuron));
    return std::clamp<brain::UIType>(cap, 1, this->_neuron - 1);
}

_cold void brain::RawBrainNetwork::sync_(void)
{
    const std::vector<brain::RawBrainNeuron> rawNeurons = this->_rawNeurons.toVector();
    const std::vector<brain::RawBrainLink> rawLinks = this->_rawLinks.toVector();
    std::vector<brain::BrainNeuron> neurons(rawNeurons.size());
    std::vector<brain::BrainLink> links(rawLinks.size()); // free slots at 0: no effect

    // the biases, then the inward links of each neuron packed from its first slot
    for (brain::UIType i = 0; i < rawNeurons.size(); ++i) {
        brain::UIType slot = 0;

        neurons[i].bias = rawNeurons[i].bias;
        for (brain::UIType k = 0; k < this->_cap && i * this->_cap + k < rawLinks.size(); ++k) {
            const brain::RawBrainLink& link = rawLinks[i * this->_cap + k];
            if (link.inward) links[i * this->_cap + slot++] = {link.id, link.weight};
        }
    }
    this->_neurons.allocate(neurons.size());
    this->_neurons.store(neurons);
    this->_links.allocate(links.size());
    this->_links.store(links);
}

_cold brain::Bytes brain::RawBrainNetwork::serialize(void) const
{
    const std::vector<brain::RawBrainNeuron> neurons = this->_rawNeurons.toVector();
    const std::vector<brain::RawBrainLink> links = this->_rawLinks.toVector();
    brain::Bytes out;

    // common part, then the space, the limits, the neurons and the links (the reduced memories are rebuilt by the load)
    this->serializeHeader_(out, static_cast<std::uint8_t>(BRAIN_MAGIC_RAW));
    brain::tools::put<std::uint64_t>(out, this->_neuron);
    brain::tools::put<std::uint64_t>(out, this->_cap);
    brain::tools::put<brain::FType>(out, this->_size.first);
    brain::tools::put<brain::FType>(out, this->_size.second);
    brain::tools::put<brain::FType>(out, this->_limit.first);
    brain::tools::put<std::uint8_t>(out, this->_customLimit ? static_cast<std::uint8_t>(BRAIN_ACTIVATION_CUSTOM) : static_cast<std::uint8_t>(BRAIN_ACTIVATION_NONE));
    brain::tools::put<std::uint64_t>(out, neurons.size());
    for (const brain::RawBrainNeuron& neuron: neurons) {
        brain::tools::put<brain::FType>(out, neuron.pos.x);
        brain::tools::put<brain::FType>(out, neuron.pos.y);
        brain::tools::put<brain::FType>(out, neuron.bias);
    }
    brain::tools::put<std::uint64_t>(out, links.size());
    for (const brain::RawBrainLink& link: links) {
        brain::tools::put<std::uint64_t>(out, link.id);
        brain::tools::put<brain::FType>(out, link.weight);
        brain::tools::put<bool>(out, link.inward);
    }
    return out;
}

// Every link has its clone on the other neuron with the same weight: the inward one on the receiver and the outward one on the sender (the free slots, all at 0, are not links)
_cold static bool clonesMatch(const std::vector<brain::RawBrainLink>& links, const brain::UIType neuron, const brain::UIType cap)
{
    std::vector<std::vector<std::pair<brain::UIType, brain::FType>>> inward(neuron);
    std::vector<std::vector<std::pair<brain::UIType, brain::FType>>> outward(neuron);

    for (brain::UIType owner = 0; owner < neuron; ++owner) {
        for (brain::UIType slot = 0; slot < cap; ++slot) {
            const brain::RawBrainLink& link = links[owner * cap + slot];

            if (link.inward) inward[owner].push_back({link.id, link.weight});
            else if (link.id != 0 || link.weight != 0) outward[owner].push_back({link.id, link.weight});
        }
        std::sort(inward[owner].begin(), inward[owner].end());
        std::sort(outward[owner].begin(), outward[owner].end());
    }
    for (brain::UIType owner = 0; owner < neuron; ++owner) {
        for (const std::pair<brain::UIType, brain::FType>& link: inward[owner]) {
            if (!std::binary_search(outward[link.first].begin(), outward[link.first].end(), std::pair<brain::UIType, brain::FType>{owner, link.second})) return false;
        }
        for (const std::pair<brain::UIType, brain::FType>& link: outward[owner]) {
            if (!std::binary_search(inward[link.first].begin(), inward[link.first].end(), std::pair<brain::UIType, brain::FType>{owner, link.second})) return false;
        }
    }
    return true;
}

_cold void brain::RawBrainNetwork::deserialize(const brain::Bytes& data)
{
    brain::UIType offset = this->deserializeHeader_(data, static_cast<std::uint8_t>(BRAIN_MAGIC_RAW));
    const brain::UIType neuron = brain::tools::get<std::uint64_t>(data, offset);
    const brain::UIType cap = brain::tools::get<std::uint64_t>(data, offset);
    const brain::FType width = brain::tools::get<brain::FType>(data, offset);
    const brain::FType height = brain::tools::get<brain::FType>(data, offset);
    const brain::FType percentage = brain::tools::get<brain::FType>(data, offset);
    const std::uint8_t custom = brain::tools::get<std::uint8_t>(data, offset);
    brain::UIType count = brain::tools::get<std::uint64_t>(data, offset);

    // the space and the limits
    if (neuron == 0 || !(width > 0) || !(height > 0) || !std::isfinite(width) || !std::isfinite(height)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid number of neurons or size of the space");
    } else if (!std::isfinite(percentage) || percentage <= 0 || percentage > 1 || (custom != BRAIN_ACTIVATION_NONE && custom != BRAIN_ACTIVATION_CUSTOM)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid links limit");
    }

    // the neurons (none: not allocated yet)
    if ((count != 0 && count != neuron) || !brain::tools::fits(data, offset, count, 3 * sizeof(brain::FType))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid neurons");
    }
    std::vector<brain::RawBrainNeuron> neurons(count);
    for (brain::RawBrainNeuron& value: neurons) {
        value.pos.x = brain::tools::get<brain::FType>(data, offset);
        value.pos.y = brain::tools::get<brain::FType>(data, offset);
        value.bias = brain::tools::get<brain::FType>(data, offset);
    }

    // the links (none: not allocated yet)
    count = brain::tools::get<std::uint64_t>(data, offset);
    if ((count != 0 && (cap == 0 || count / cap != neuron || count % cap != 0)) || !brain::tools::fits(data, offset, count, sizeof(std::uint64_t) + sizeof(brain::FType) + sizeof(bool))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid links");
    }
    std::vector<brain::RawBrainLink> links(count);
    for (brain::RawBrainLink& link: links) {
        link.id = brain::tools::get<std::uint64_t>(data, offset);
        link.weight = brain::tools::get<brain::FType>(data, offset);
        link.inward = brain::tools::get<bool>(data, offset);
        if (link.id >= neuron) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "a link goes to the neuron " + std::to_string(link.id) + " of " + std::to_string(neuron));
        }
    }
    if (offset != data.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "unexpected data after the links");
    } else if (!links.empty() && !clonesMatch(links, neuron, cap)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "a link has no clone with the same weight on the other neuron");
    }

    // everything is valid: the network is replaced, then its reduced copy is rebuilt
    this->_neuron = neuron;
    this->_cap = cap;
    this->_size = {width, height};
    this->_limit = {percentage, &brain::RawBrainNetwork::defaultCap_};
    this->_customLimit = false;
    if (custom == BRAIN_ACTIVATION_CUSTOM) {
        this->_customLimit = true;
        std::cerr << utils::exception::WarningException(utils::exception::ExternalCode::CustomLinksCapRequired).formated() << std::endl;
    }
    this->_rawNeurons.allocate(neurons.size());
    this->_rawNeurons.store(neurons);
    this->_rawLinks.allocate(links.size());
    this->_rawLinks.store(links);
    this->sync_();
}

_hot brain::Values brain::RawBrainNetwork::apply(void)
{
    // check inputs
    if (this->_inputs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no inputs loaded (call inputs() first)");
    }
    /* Nothing */
    return {};
}

_hot void brain::RawBrainNetwork::backpropagation(const brain::FType coef)
{
    // check inputs
    if (!std::isfinite(coef)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficient should be a finite number");
    }
    /* Nothing */
}

_hot void brain::RawBrainNetwork::backpropagation(const brain::Values& coefs)
{
    // check inputs
    if (coefs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "no coefficients given");
    } else if (this->_ioSize.second != 0 && coefs.size() != this->_ioSize.second) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "given " + std::to_string(coefs.size()) + ", expected " + std::to_string(this->_ioSize.second));
    }
    /* Nothing */
}

_cold void brain::RawBrainNetwork::weightInit(const brain::FType fallback)
{
    std::vector<brain::RawBrainLink> links;

    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }

    // a random value for every link, the same one on the receiver (inward) and on the sender (outward)
    links = this->_rawLinks.toVector();
    for (brain::UIType receiver = 0; receiver * this->_cap < links.size(); ++receiver) {
        for (brain::UIType slot = 0; slot < this->_cap; ++slot) {
            brain::RawBrainLink& link = links[receiver * this->_cap + slot];

            if (!link.inward) continue;
            link.weight = this->random_(fallback);
            for (brain::UIType other = 0; other < this->_cap; ++other) {
                brain::RawBrainLink& mirror = links[link.id * this->_cap + other];

                if (!mirror.inward && mirror.id == receiver) {
                    mirror.weight = link.weight;
                    break;
                }
            }
        }
    }
    this->_rawLinks.store(links);
    this->sync_();
}

_cold void brain::RawBrainNetwork::biasInit(const brain::FType fallback)
{
    std::vector<brain::RawBrainNeuron> neurons;

    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }

    // a random value for every bias
    neurons = this->_rawNeurons.toVector();
    for (brain::RawBrainNeuron& neuron: neurons)
        neuron.bias = this->random_(fallback);
    this->_rawNeurons.store(neurons);
    this->sync_();
}

_cold void brain::RawBrainNetwork::linksCap(const brain::FType percentage)
{
    // check inputs
    if (!std::isfinite(percentage) || percentage <= 0 || percentage > 1) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the percentage of links should be above 0 and under or equal to 1");
    }
    this->_limit.first = percentage;
    this->init_(); // the slots of the links changed: the neurons and the links are set up again
}

_cold void brain::RawBrainNetwork::linksCap(const brain::LinksCapFn& fn)
{
    // check inputs
    if (!fn) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the links cap function is empty");
    }
    this->_limit.second = fn;
    this->_customLimit = true;
    this->init_(); // the slots of the links changed: the neurons and the links are set up again
}

_cold void brain::RawBrainNetwork::resetLinksCap(void)
{
    this->_limit = {BRAIN_LINKS_CAP_PERCENTAGE, &brain::RawBrainNetwork::defaultCap_};
    this->_customLimit = false;
    this->init_(); // the slots of the links changed: the neurons and the links are set up again
}

_cold std::unique_ptr<brain::BrainNetwork> brain::RawBrainNetwork::reduce(void)
{
    std::unique_ptr<brain::BrainNetwork> network = std::make_unique<brain::BrainNetwork>(this->_neuron, this->_cap, this->_mode);

    // the reduced copy is made from the raw memories, then given through its own deserialize
    this->sync_();
    network->deserialize(this->BrainNetwork::serialize());
    return network;
}
