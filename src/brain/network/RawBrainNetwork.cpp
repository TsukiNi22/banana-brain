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
#include "brain/tools/compute.hpp"
#include "brain/NetworkDefine.hpp"
#include "brain/tools/neuron.hpp"
#include "brain/tools/random.hpp"
#include "brain/tools/bytes.hpp"
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>

brain::RawBrainNetwork::RawBrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size, const utils::system::MemoryMode mode)
: BrainNetwork{neuron, 0, mode, false}, _rawNeurons{this->makeMemory_<brain::RawBrainNeuron>()}, _rawLinks{this->makeMemory_<brain::RawBrainLink>()}, _size{size}, _limit{BRAIN_LINKS_CAP_PERCENTAGE, &brain::RawBrainNetwork::defaultCap_}, _stepsLimit{BRAIN_STEPS_CAP_PERCENTAGE, &brain::RawBrainNetwork::defaultSteps_}
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
    const brain::UIType input = this->_ioSize.first; // the first neurons are the inputs, the last ones the outputs
    const brain::UIType output = this->_ioSize.second;

    // check inputs (the memories are allocated in bytes, a count that overflows is refused here)
    this->_cap = this->linksCap_();
    this->_steps = this->stepsCap_();
    if (__builtin_mul_overflow(this->_neuron, this->_cap, &total)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfMemory, "the links of the neurons overflow the size");
    }
    links.resize(total);

    // the inputs are on the left edge and the outputs on the right edge, evenly spread, the others are placed randomly (with the generator of the library: brain::tools::rand(seed) to choose the seed)
    for (brain::UIType i = 0; i < neurons.size(); ++i) {
        if (i < input) neurons[i].pos = {0.0, (static_cast<brain::FType>(i) + 0.5f) * this->_size.second / static_cast<brain::FType>(input)};
        else if (i >= neurons.size() - output) neurons[i].pos = {this->_size.first, (static_cast<brain::FType>(i - (neurons.size() - output)) + 0.5f) * this->_size.second / static_cast<brain::FType>(output)};
        else neurons[i].pos = {brain::tools::rand_floating() * this->_size.first, brain::tools::rand_floating() * this->_size.second};
    }

    // each neuron gets a random number of links (1 -> cap) with its nearest neurons, the closest first without skipping one (but the ones that can't be linked: a full neuron, two inputs or two outputs)
    // the direction of each link is random too, except for the inputs that only send and the outputs that only receive
    for (brain::UIType owner = 0; owner < neurons.size(); ++owner) {
        const brain::UIType target = std::min<brain::UIType>(this->_cap, 1 + static_cast<brain::UIType>(brain::tools::rand_floating() * static_cast<brain::FType>(this->_cap)));
        const bool ownerInput = owner < input;
        const bool ownerOutput = owner >= neurons.size() - output;

        if (this->_cap == 0) break;
        for (const brain::UIType other: brain::tools::sort(owner, neurons, neurons.size() - 1)) {
            const bool otherInput = other < input;
            const bool otherOutput = other >= neurons.size() - output;
            const bool canReceive = !ownerInput && !otherOutput; // owner <- other
            const bool canSend = !ownerOutput && !otherInput; // owner -> other
            bool linked = false;

            if (used[owner] >= target) break;
            for (brain::UIType slot = 0; slot < used[owner] && !linked; ++slot)
                linked = links[owner * this->_cap + slot].id == other;
            if (linked || used[owner] >= this->_cap || used[other] >= this->_cap || (!canReceive && !canSend)) continue;
            const bool receive = canReceive && (!canSend || brain::tools::rand_floating() < 0.5f);

            links[owner * this->_cap + used[owner]++] = {other, 1.0, receive}; // weight at 1.0 by default: weightInit gives the random ones
            links[other * this->_cap + used[other]++] = {owner, 1.0, !receive};
        }
    }
    this->_rawNeurons.allocate(neurons.size());
    this->_rawNeurons.store(neurons);
    this->_rawLinks.allocate(links.size());
    this->_rawLinks.store(links);
    this->_placed = this->_ioSize;
    this->sync_();
}

_cold void brain::RawBrainNetwork::resized_(void)
{
    this->BrainNetwork::resized_(); // the inputs and the outputs fit in the neurons
    if (this->_ioSize != this->_placed) this->init_();
}

_cold brain::UIType brain::RawBrainNetwork::stepsCap_(void) const
{
    brain::UIType steps = static_cast<brain::UIType>(std::floor(static_cast<double>(this->_neuron) * static_cast<double>(this->_stepsLimit.first)));

    if (this->_stepsLimit.second) steps = std::min(steps, this->_stepsLimit.second(this->_neuron));
    return std::max<brain::UIType>(steps, 1);
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
    brain::tools::put<brain::FType>(out, this->_stepsLimit.first);
    brain::tools::put<std::uint8_t>(out, this->_customSteps ? static_cast<std::uint8_t>(BRAIN_ACTIVATION_CUSTOM) : static_cast<std::uint8_t>(BRAIN_ACTIVATION_NONE));
    brain::tools::put<std::uint64_t>(out, this->_steps);
    brain::tools::put<brain::FType>(out, this->_threshold);
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
    const brain::FType stepsPercentage = brain::tools::get<brain::FType>(data, offset);
    const std::uint8_t stepsCustom = brain::tools::get<std::uint8_t>(data, offset);
    const brain::UIType steps = brain::tools::get<std::uint64_t>(data, offset);
    const brain::FType threshold = brain::tools::get<brain::FType>(data, offset);
    brain::UIType count = brain::tools::get<std::uint64_t>(data, offset);

    // the space and the limits
    if (neuron == 0 || !(width > 0) || !(height > 0) || !std::isfinite(width) || !std::isfinite(height)
        || this->_ioSize.first > neuron || this->_ioSize.second > neuron - this->_ioSize.first) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid number of neurons, size of the space or inputs and outputs");
    } else if (!std::isfinite(percentage) || percentage <= 0 || percentage > 1 || (custom != BRAIN_ACTIVATION_NONE && custom != BRAIN_ACTIVATION_CUSTOM)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid links limit");
    } else if (!std::isfinite(stepsPercentage) || stepsPercentage <= 0 || stepsPercentage > 1 || steps == 0 || (stepsCustom != BRAIN_ACTIVATION_NONE && stepsCustom != BRAIN_ACTIVATION_CUSTOM)
        || !std::isfinite(threshold) || threshold < 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid time steps limit or signal threshold");
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
    this->_stepsLimit = {stepsPercentage, &brain::RawBrainNetwork::defaultSteps_};
    this->_customSteps = stepsCustom == BRAIN_ACTIVATION_CUSTOM;
    this->_steps = steps;
    this->_threshold = threshold;
    this->_placed = this->_ioSize;
    if (this->_customSteps) {
        std::cerr << utils::exception::WarningException(utils::exception::ExternalCode::CustomStepsCapRequired).formated() << std::endl;
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
    } else if (this->_ioSize.first == 0 || this->_rawNeurons.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "the number of inputs of the network is not set (size() / input())");
    } else if (this->_ioSize.second == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "the number of outputs of the network is not set (size() / output())");
    }

    // the memories stored on the gpu are computed there (an external activation function can only run on the host)
    if (this->_rawNeurons.isOnGpu() && !this->_activationFunction.second) {
        if (brain::tools::gpu::steps(this->_gpu.get(), this->_rawNeurons.native(), this->_rawLinks.native(), true, this->_neuron, this->_cap, this->_ioSize.first, this->_steps, this->_activationFunction.first, this->_threshold, this->_inputs, this->_states, true))
            return brain::Values(this->_states.back().end() - static_cast<std::ptrdiff_t>(this->_ioSize.second), this->_states.back().end());
    }

    // the memories are read on the host (a gpu memory is fetched once)
    std::vector<brain::RawBrainNeuron> fetchedNeurons;
    std::vector<brain::RawBrainLink> fetchedLinks;
    const brain::RawBrainNeuron* neurons = this->_rawNeurons.data();
    const brain::RawBrainLink* links = this->_rawLinks.data();
    const brain::UIType input = this->_ioSize.first;
    if (neurons == nullptr) {fetchedNeurons = this->_rawNeurons.toVector(); neurons = fetchedNeurons.data();}
    if (links == nullptr && !this->_rawLinks.empty()) {fetchedLinks = this->_rawLinks.toVector(); links = fetchedLinks.data();}

    // same time steps as the reduced network, but on the raw links (only the inward ones read), every step is kept for the backpropagation
    this->_states.assign(1, brain::Values(this->_neuron, 0.0));
    std::copy(this->_inputs.begin(), this->_inputs.end(), this->_states[0].begin());
    for (brain::UIType step = 0; step < this->_steps; ++step) {
        brain::Values next(this->_neuron, 0.0);
        const brain::Values& state = this->_states.back();

        for (brain::UIType i = 0; i < this->_neuron; ++i) {
            brain::FType sum = neurons[i].bias;

            if (i < input) {
                next[i] = this->_inputs[i];
                continue;
            }
            for (brain::UIType slot = 0; links != nullptr && slot < this->_cap; ++slot) {
                if (links[i * this->_cap + slot].inward) sum += links[i * this->_cap + slot].weight * state[links[i * this->_cap + slot].id];
            }
            next[i] = this->activate_(sum);
            if (std::abs(next[i]) < this->_threshold) next[i] = 0.0; // too weak to matter
        }
        this->_states.push_back(std::move(next));
    }
    return brain::Values(this->_states.back().end() - static_cast<std::ptrdiff_t>(this->_ioSize.second), this->_states.back().end());
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

_cold void brain::RawBrainNetwork::stepsCap(const brain::FType percentage)
{
    // check inputs
    if (!std::isfinite(percentage) || percentage <= 0 || percentage > 1) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the percentage of time steps should be above 0 and under or equal to 1");
    }
    this->_stepsLimit.first = percentage;
    this->_steps = this->stepsCap_();
}

_cold void brain::RawBrainNetwork::stepsCap(const brain::StepsCapFn& fn)
{
    // check inputs
    if (!fn) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the steps cap function is empty");
    }
    this->_stepsLimit.second = fn;
    this->_customSteps = true;
    this->_steps = this->stepsCap_();
}

_cold void brain::RawBrainNetwork::resetStepsCap(void)
{
    this->_stepsLimit = {BRAIN_STEPS_CAP_PERCENTAGE, &brain::RawBrainNetwork::defaultSteps_};
    this->_customSteps = false;
    this->_steps = this->stepsCap_();
}
