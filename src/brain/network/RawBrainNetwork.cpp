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
#include <functional>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <tuple>
#include <cmath>

brain::RawBrainNetwork::RawBrainNetwork(const brain::UIType neuron, const std::pair<brain::FType, brain::FType>& size, const utils::system::MemoryMode mode)
: BrainNetwork{neuron, 0, mode, false}, _rawNeurons{this->makeMemory_<brain::RawBrainNeuron>()}, _rawLinks{this->makeMemory_<brain::RawBrainLink>()}, _size{size}, _limit{BRAIN_LINKS_CAP_PERCENTAGE, &brain::RawBrainNetwork::defaultCap_}, _stepsLimit{BRAIN_STEPS_CAP_PERCENTAGE, &brain::RawBrainNetwork::defaultSteps_}, _cloneMemory{this->makeMemory_<brain::UIType>()}
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

    // each neuron is linked to as many of its nearest neurons as its cap allows, the closest first without skipping one (but the ones that can't be linked: a full neuron, two inputs or two outputs)
    // the direction of each link is random, except for the inputs that only send and the outputs that only receive; left to right: only the neurons on the left, the owner receives from them
    this->near_(neurons);
    const brain::UIType nearCap = this->_cap - std::min(this->_farLinks, this->_cap); // the slots left for the far links drawn below
    for (brain::UIType owner = 0; owner < neurons.size(); ++owner) {
        const brain::UIType target = this->_randomLinks ? std::min(nearCap, std::max(std::min(this->_minLinks, this->_cap), 1 + std::min<brain::UIType>(static_cast<brain::UIType>(brain::tools::rand_floating() * static_cast<brain::FType>(this->_cap)), this->_cap - 1))) : nearCap; // random: between minLinks and the cap
        const bool ownerInput = owner < input;
        const bool ownerOutput = owner >= neurons.size() - output;

        if (this->_cap == 0) break;
        (void)this->nearest_(owner, neurons, [&](const brain::UIType other) {
            const bool otherInput = other < input;
            const bool otherOutput = other >= neurons.size() - output;
            const bool canReceive = !ownerInput && !otherOutput; // owner <- other
            const bool canSend = !ownerOutput && !otherInput; // owner -> other
            bool linked = false;

            if (used[owner] >= target) return true;
            for (brain::UIType slot = 0; slot < used[owner] && !linked; ++slot)
                linked = links[owner * this->_cap + slot].id == other;
            if (linked || used[owner] >= nearCap || used[other] >= nearCap || (!canReceive && !canSend) || (this->_leftToRight && !canReceive)) return false;
            const bool receive = this->_leftToRight || (canReceive && (!canSend || brain::tools::rand_floating() < 0.5f)); // left to right: the other one is on the left, it sends

            links[owner * this->_cap + used[owner]++] = {other, 1.0, receive}; // weight at 1.0 by default: weightInit gives the random ones
            links[other * this->_cap + used[other]++] = {owner, 1.0, !receive};
            return false;
        });
    }

    // far links: the free slots go to neurons drawn at random among the ones the owner can link (the left ones in left to right)
    for (brain::UIType owner = 0; owner < neurons.size() && nearCap < this->_cap; ++owner) {
        const bool ownerInput = owner < input;
        const bool ownerOutput = owner >= neurons.size() - output;
        std::vector<brain::UIType> pool = this->candidates_(owner, neurons);

        while (used[owner] < this->_cap && !pool.empty()) {
            const brain::UIType at = std::min<brain::UIType>(static_cast<brain::UIType>(brain::tools::rand_floating() * static_cast<brain::FType>(pool.size())), pool.size() - 1);
            const brain::UIType other = pool[at];
            const bool canReceive = !ownerInput && other < neurons.size() - output;
            const bool canSend = !ownerOutput && other >= input;
            bool linked = false;

            pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(at));
            for (brain::UIType slot = 0; slot < used[owner] && !linked; ++slot)
                linked = links[owner * this->_cap + slot].id == other;
            if (linked || used[other] >= this->_cap || (!canReceive && !canSend) || (this->_leftToRight && !canReceive)) continue;
            const bool receive = this->_leftToRight || (canReceive && (!canSend || brain::tools::rand_floating() < 0.5f));

            links[owner * this->_cap + used[owner]++] = {other, 1.0, receive};
            links[other * this->_cap + used[other]++] = {owner, 1.0, !receive};
        }
    }
    if (this->_pathHops > 0 && this->_cap > 0) this->ensurePaths_(neurons, links);
    this->_velocity.clear();
    this->_moments.clear();
    this->_adamSteps = 0;
    this->_updates = 0;
    this->_seen.clear();
    this->_clones.clear();
    this->_pending.clear();
    this->_pendingCount = 0;
    this->_rawNeurons.allocate(neurons.size());
    this->_rawNeurons.store(neurons);
    this->_rawLinks.allocate(links.size());
    this->_rawLinks.store(links);
    this->_placed = this->_ioSize;
    this->sync_();
}

_cold void brain::RawBrainNetwork::resized_(void)
{
    this->fits_(); // the inputs and the outputs fit in the neurons (the links are set up again for the new sizes)
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

_cold static brain::UIType cloneSlot(const std::vector<brain::RawBrainLink>& links, const brain::UIType cap, const brain::UIType owner, const brain::RawBrainLink& link); // defined with the helpers of the links below
_cold static bool occupied(const brain::RawBrainLink& link);

_cold void brain::RawBrainNetwork::sync_(const std::vector<brain::RawBrainNeuron>& rawNeurons, const std::vector<brain::RawBrainLink>& rawLinks)
{
    std::vector<brain::BrainNeuron> neurons(rawNeurons.size());
    std::vector<brain::BrainLink> links(rawLinks.size()); // free slots at 0: no effect

    // the biases, then the inward links of each neuron packed from its first slot (and their slots, read by the raw execution and the backpropagation)
    this->_inStart.assign(rawNeurons.size() + 1, 0);
    this->_inSlot.clear();
    for (brain::UIType i = 0; i < rawNeurons.size(); ++i) {
        brain::UIType slot = 0;

        this->_inStart[i] = this->_inSlot.size();
        neurons[i].bias = rawNeurons[i].bias;
        for (brain::UIType k = 0; k < this->_cap && i * this->_cap + k < rawLinks.size(); ++k) {
            const brain::RawBrainLink& link = rawLinks[i * this->_cap + k];

            if (!link.inward) continue;
            links[i * this->_cap + slot++] = {link.id, link.weight};
            this->_inSlot.push_back(i * this->_cap + k);
        }
    }
    this->_inStart[rawNeurons.size()] = this->_inSlot.size();

    // the slot of the clone of every inward link, found once per change of the links (not once per update), on the gpu too when the network is there
    if (this->_clones.size() != rawLinks.size()) {
        std::vector<std::vector<std::pair<brain::UIType, brain::UIType>>> outward(rawNeurons.size()); // <receiver, slot> of the outward links of each sender, sorted by receiver (a pair of neurons has one link at most)

        this->_clones.assign(rawLinks.size(), 0);
        for (brain::UIType j = 0; j < rawNeurons.size(); ++j) {
            for (brain::UIType slot = 0; slot < this->_cap && j * this->_cap + slot < rawLinks.size(); ++slot) {
                const brain::RawBrainLink& link = rawLinks[j * this->_cap + slot];
                if (!link.inward && occupied(link)) outward[j].push_back({link.id, slot});
            }
            std::sort(outward[j].begin(), outward[j].end());
        }
        for (brain::UIType i = 0; i < rawNeurons.size(); ++i) {
            for (brain::UIType slot = 0; slot < this->_cap && i * this->_cap + slot < rawLinks.size(); ++slot) {
                const brain::RawBrainLink& link = rawLinks[i * this->_cap + slot];
                if (!link.inward) continue;
                const std::vector<std::pair<brain::UIType, brain::UIType>>& list = outward[link.id];
                const std::vector<std::pair<brain::UIType, brain::UIType>>::const_iterator found = std::lower_bound(list.begin(), list.end(), std::pair<brain::UIType, brain::UIType>{i, 0});

                this->_clones[i * this->_cap + slot] = link.id * this->_cap + (found != list.end() && found->first == i ? found->second : cloneSlot(rawLinks, this->_cap, i, link));
            }
        }
        this->_clonesChanged = true;
    }
    if (this->_clonesChanged && this->_cloneMemory.isOnGpu()) {
        if (this->_cloneMemory.size() != this->_clones.size()) this->_cloneMemory.allocate(this->_clones.size());
        this->_cloneMemory.store(this->_clones);
    }
    this->_clonesChanged = false;
    if (this->_neurons.size() != neurons.size()) this->_neurons.allocate(neurons.size()); // kept when the size is the same (sync_ runs after every update: no new memory each time)
    this->_neurons.store(neurons);
    if (this->_links.size() != links.size()) this->_links.allocate(links.size());
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
    brain::tools::put<brain::FType>(out, this->_weightLimits.first);
    brain::tools::put<brain::FType>(out, this->_weightLimits.second);
    brain::tools::put<std::uint8_t>(out, this->_orphans ? 1 : 0);
    brain::tools::put<std::uint8_t>(out, this->_leftToRight ? 1 : 0);
    brain::tools::put<std::uint64_t>(out, this->_farLinks);
    brain::tools::put<std::uint64_t>(out, this->_pathHops);
    brain::tools::put<brain::FType>(out, this->_momentum);
    brain::tools::put<brain::FType>(out, this->_stepLimit);
    brain::tools::put<std::uint8_t>(out, this->_clampWeights ? 1 : 0);
    brain::tools::put<std::uint8_t>(out, this->_lowNewLinks ? 1 : 0);
    brain::tools::put<std::uint8_t>(out, this->_batch ? 1 : 0);
    brain::tools::put<std::uint8_t>(out, this->_directOutput ? 1 : 0);
    brain::tools::put<std::uint8_t>(out, this->_keepPaths ? 1 : 0);
    brain::tools::put<brain::FType>(out, this->_adam);
    brain::tools::put<brain::FType>(out, this->_rateDecay);
    brain::tools::put<std::uint8_t>(out, this->_invertedLinks ? 1 : 0);
    brain::tools::put<std::uint8_t>(out, this->_randomLinks ? 1 : 0);
    brain::tools::put<std::uint64_t>(out, this->_minLinks);
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
    const brain::FType weightMin = brain::tools::get<brain::FType>(data, offset);
    const brain::FType weightMax = brain::tools::get<brain::FType>(data, offset);
    const std::uint8_t orphans = brain::tools::get<std::uint8_t>(data, offset);
    const std::uint8_t leftToRight = brain::tools::get<std::uint8_t>(data, offset);
    const brain::UIType farLinks = brain::tools::get<std::uint64_t>(data, offset);
    const brain::UIType pathHops = brain::tools::get<std::uint64_t>(data, offset);
    const brain::FType momentum = brain::tools::get<brain::FType>(data, offset);
    const brain::FType stepLimit = brain::tools::get<brain::FType>(data, offset);
    const std::uint8_t clampWeights = brain::tools::get<std::uint8_t>(data, offset);
    const std::uint8_t lowNewLinks = brain::tools::get<std::uint8_t>(data, offset);
    const std::uint8_t batch = brain::tools::get<std::uint8_t>(data, offset);
    const std::uint8_t directOutput = brain::tools::get<std::uint8_t>(data, offset);
    const std::uint8_t keepPaths = brain::tools::get<std::uint8_t>(data, offset);
    const brain::FType adam = brain::tools::get<brain::FType>(data, offset);
    const brain::FType rateDecay = brain::tools::get<brain::FType>(data, offset);
    const std::uint8_t invertedLinks = brain::tools::get<std::uint8_t>(data, offset);
    const std::uint8_t randomLinks = brain::tools::get<std::uint8_t>(data, offset);
    const brain::UIType minLinks = brain::tools::get<std::uint64_t>(data, offset);
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

    if (!std::isfinite(weightMin) || !std::isfinite(weightMax) || weightMin < 0 || weightMax <= weightMin || orphans > 1 || leftToRight > 1) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid limits of the weights, orphans or left to right mode");
    } else if (pathHops == 1 || !std::isfinite(momentum) || momentum < 0 || momentum >= 1 || !std::isfinite(stepLimit) || stepLimit < 0 || clampWeights > 1 || lowNewLinks > 1 || batch > 1 || directOutput > 1 || keepPaths > 1 || !std::isfinite(adam) || adam < 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid path, momentum, step limit, clamp, new links, mini-batch, direct output, paths or Adam mode");
    }

    if (!std::isfinite(rateDecay) || rateDecay < 0 || invertedLinks > 1 || randomLinks > 1 || minLinks > cap) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid rate decay, links mode or minimum of links");
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

    // an input only sends (outward links), an output only receives (inward links)
    for (brain::UIType owner = 0; owner < neuron && cap > 0; ++owner) {
        for (brain::UIType slot = 0; slot < cap; ++slot) {
            const brain::RawBrainLink& link = links[owner * cap + slot];

            if (owner < this->_ioSize.first && link.inward) _unlikely {
                throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the input " + std::to_string(owner) + " receives a link (an input only sends)");
            } else if (owner >= neuron - this->_ioSize.second && !link.inward && (link.id != 0 || link.weight != 0)) _unlikely {
                throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the output " + std::to_string(owner) + " sends a link (an output only receives)");
            } else if (leftToRight == 1 && link.inward && !(neurons[link.id].pos.x < neurons[owner].pos.x)) _unlikely {
                throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the neuron " + std::to_string(owner) + " receives from the neuron " + std::to_string(link.id) + " on its right (left to right mode)");
            }
        }
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
    this->_weightLimits = {weightMin, weightMax};
    this->_orphans = orphans == 1;
    this->_leftToRight = leftToRight == 1;
    this->near_(neurons);
    this->_farLinks = farLinks;
    this->_pathHops = pathHops;
    this->_momentum = momentum;
    this->_stepLimit = stepLimit;
    this->_clampWeights = clampWeights == 1;
    this->_lowNewLinks = lowNewLinks == 1;
    this->_batch = batch == 1;
    this->_directOutput = directOutput == 1;
    this->_keepPaths = keepPaths == 1;
    this->_adam = adam;
    this->_rateDecay = rateDecay;
    this->_invertedLinks = invertedLinks == 1;
    this->_randomLinks = randomLinks == 1;
    this->_minLinks = minLinks;
    this->_updates = 0;
    this->_moments.clear();
    this->_adamSteps = 0;
    this->_seen.clear();
    this->_clones.clear();
    this->_velocity.clear();
    this->_pending.clear();
    this->_pendingCount = 0;
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

    // the memories stored on the gpu are computed there (an external activation function can only run on the host, a sequence too)
    if (this->_rawNeurons.isOnGpu() && !this->_activationFunction.second && this->_sequence.empty()) {
        if (brain::tools::gpu::steps(this->_gpu.get(), this->_rawNeurons.native(), this->_rawLinks.native(), true, this->_neuron, this->_cap, this->_ioSize.first, this->_steps, this->_activationFunction.first, this->_threshold, this->_inputs, this->_states, true)) {
            this->_sums.clear(); // computed on the device: the backpropagation computes them again
            this->watch_();
            return brain::Values(this->_states.back().end() - static_cast<std::ptrdiff_t>(this->_ioSize.second), this->_states.back().end());
        }
    }

    // the memories are read on the host (a gpu memory is fetched once)
    std::vector<brain::RawBrainNeuron> fetchedNeurons;
    std::vector<brain::RawBrainLink> fetchedLinks;
    const brain::RawBrainNeuron* neurons = this->_rawNeurons.data();
    const brain::RawBrainLink* links = this->_rawLinks.data();
    const brain::UIType input = this->_ioSize.first;
    if (neurons == nullptr) {fetchedNeurons = this->_rawNeurons.toVector(); neurons = fetchedNeurons.data();}
    if (links == nullptr && !this->_rawLinks.empty()) {fetchedLinks = this->_rawLinks.toVector(); links = fetchedLinks.data();}

    // same time steps as the reduced network, but on the raw links (only the inward ones read, packed in the order of the slots), every step and its sums kept for the backpropagation
    this->_states.assign(1, brain::Values(this->_neuron, 0.0));
    this->_sums.assign(this->_steps + 1, brain::Values(this->_neuron, 0.0));
    std::copy(this->_inputs.begin(), this->_inputs.end(), this->_states[0].begin());
    for (brain::UIType step = 0; step < this->_steps; ++step) {
        brain::Values next(this->_neuron, 0.0);
        const brain::Values& state = this->_states.back();
        brain::Values& sums = this->_sums[step + 1];

        for (brain::UIType i = 0; i < this->_neuron; ++i) {
            brain::FType sum = neurons[i].bias;

            if (i < input) {
                next[i] = this->_sequence.empty() ? this->_inputs[i] : (step + 1 < this->_sequence.size() ? this->_sequence[step + 1][i] : 0.0f); // a sequence: its next step, then 0
                continue;
            }
            for (brain::UIType k = this->_inStart[i]; links != nullptr && k < this->_inStart[i + 1]; ++k)
                sum += links[this->_inSlot[k]].weight * state[links[this->_inSlot[k]].id];
            sums[i] = sum;
            next[i] = this->activate_(sum);
            if (std::abs(next[i]) < this->_threshold) next[i] = 0.0; // too weak to matter
        }
        this->_states.push_back(std::move(next));
    }
    this->watch_();
    return brain::Values(this->_states.back().end() - static_cast<std::ptrdiff_t>(this->_ioSize.second), this->_states.back().end());
}

_hot void brain::RawBrainNetwork::backpropagation(const brain::FType coef)
{
    // check inputs
    if (!std::isfinite(coef)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficient should be a finite number");
    } else if (this->_ioSize.second == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "the number of outputs of the network is not set (size() / output())");
    }
    this->learn_(brain::Values(this->_ioSize.second, coef));
}

_hot void brain::RawBrainNetwork::backpropagation(const brain::Values& coefs)
{
    // check inputs
    if (coefs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "no coefficients given");
    } else if (this->_ioSize.second == 0 || coefs.size() != this->_ioSize.second) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "given " + std::to_string(coefs.size()) + ", expected " + std::to_string(this->_ioSize.second));
    }
    for (const brain::FType coef: coefs) {
        if (!std::isfinite(coef)) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficients should be finite numbers");
        }
    }
    this->learn_(coefs);
}

// A slot holds a link when it is an inward one or has an id or a weight (a free slot is all at 0)
_cold static bool occupied(const brain::RawBrainLink& link)
{
    return link.inward || link.id != 0 || link.weight != 0;
}

// The first free slot of a neuron (cap: none)
_cold static brain::UIType freeSlot(const std::vector<brain::RawBrainLink>& links, const brain::UIType cap, const brain::UIType owner)
{
    brain::UIType slot = 0;

    while (slot < cap && occupied(links[owner * cap + slot])) ++slot;
    return slot;
}

// The slot of the other neuron that holds the clone of a link
_cold static brain::UIType cloneSlot(const std::vector<brain::RawBrainLink>& links, const brain::UIType cap, const brain::UIType owner, const brain::RawBrainLink& link)
{
    brain::UIType slot = 0;

    while (slot < cap && !(occupied(links[link.id * cap + slot]) && links[link.id * cap + slot].id == owner && links[link.id * cap + slot].inward != link.inward)) ++slot;
    return slot;
}

// Number of inward and outward links of a neuron
_cold static std::pair<brain::UIType, brain::UIType> degree(const std::vector<brain::RawBrainLink>& links, const brain::UIType cap, const brain::UIType owner)
{
    std::pair<brain::UIType, brain::UIType> count = {0, 0};

    for (brain::UIType slot = 0; slot < cap; ++slot) {
        const brain::RawBrainLink& link = links[owner * cap + slot];

        if (!occupied(link)) continue;
        if (link.inward) ++count.first;
        else ++count.second;
    }
    return count;
}

// Can a neuron lose a link? Never its last one, nor the last link of its kind (an input keeps an outward link, the others an inward one) when it had one
_cold static bool survives(const std::pair<brain::UIType, brain::UIType>& count, const bool isInput, const bool inward)
{
    const brain::UIType in = count.first - (inward ? 1 : 0);
    const brain::UIType out = count.second - (inward ? 0 : 1);
    const brain::UIType before = isInput ? count.second : count.first;
    const brain::UIType after = isInput ? out : in;

    return in + out >= 1 && !(before >= 1 && after == 0);
}

// Shortest numbers of inward links from a neuron to every other one (-1: not reached), along the outward links of each neuron
_cold static std::vector<brain::IType> reach(const std::vector<brain::RawBrainLink>& links, const brain::UIType cap, const brain::UIType neuron, const brain::UIType from)
{
    std::vector<brain::IType> distance(neuron, -1);
    std::vector<brain::UIType> queue = {from};

    distance[from] = 0;
    for (brain::UIType head = 0; head < queue.size(); ++head) {
        const brain::UIType at = queue[head];

        for (brain::UIType slot = 0; slot < cap; ++slot) {
            const brain::RawBrainLink& link = links[at * cap + slot];

            if (!occupied(link) || link.inward || distance[link.id] >= 0) continue;
            distance[link.id] = distance[at] + 1;
            queue.push_back(link.id);
        }
    }
    return distance;
}

// Number of pairs input -> output joined by a path of inward links
_cold static brain::UIType pairs(const std::vector<brain::RawBrainLink>& links, const brain::UIType cap, const brain::UIType neuron, const std::pair<brain::UIType, brain::UIType>& io)
{
    brain::UIType count = 0;

    for (brain::UIType from = 0; from < io.first; ++from) {
        const std::vector<brain::IType> distance = reach(links, cap, neuron, from);

        for (brain::UIType to = neuron - io.second; to < neuron; ++to)
            if (distance[to] >= 0) ++count;
    }
    return count;
}

_hot void brain::RawBrainNetwork::learn_(const brain::Values& rewards)
{
    // the states of the last apply() are what the gradient is made of
    if (this->_states.size() != this->_steps + 1 || this->_states[0].size() != this->_neuron || this->_rawNeurons.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "nothing to learn from, call apply() first");
    }
    const brain::UIType input = this->_ioSize.first;
    const brain::UIType output = this->_ioSize.second;

    // the memories stored on the gpu are changed there (an external activation function can only run on the host)
    const bool onGpu = this->_rawNeurons.isOnGpu() && !this->_activationFunction.second && !this->hostTraining_() && brain::tools::gpu::stepsBack(this->_gpu.get(), this->_rawNeurons.native(), this->_rawLinks.native(), this->_neuron, this->_cap, input, this->_activationFunction.first, this->_states, rewards, this->_cloneMemory.native());

    // no link at a limit: nothing is linked or unlinked, the execution copy is rebuilt on the device and nothing is read back
    bool any = true;
    if (onGpu && brain::tools::gpu::triggered(this->_gpu.get(), this->_rawLinks.native(), this->_neuron, this->_cap, this->_weightLimits.first, this->_weightLimits.second, any) && !any
        && brain::tools::gpu::reduce(this->_gpu.get(), this->_rawNeurons.native(), this->_rawLinks.native(), this->_neurons.native(), this->_links.native(), this->_neuron, this->_cap)) return;
    std::vector<brain::RawBrainNeuron> neurons = this->_rawNeurons.toVector();
    std::vector<brain::RawBrainLink> links = this->_rawLinks.toVector();
    brain::Values grad(this->_neuron, 0.0); // gain of the score for the output of each neuron, at the step read
    brain::Values biasGrad(this->_neuron, 0.0);
    brain::Values weightGrad(links.size(), 0.0); // by inward slot

    if (!onGpu) {
        const bool known = this->_sums.size() == this->_steps + 1; // the sums of the apply on the host (else computed again)

        // the scores are given to the outputs of the last step, then they go back in time through the inward links
        for (brain::UIType k = 0; k < output; ++k)
            grad[this->_neuron - output + k] = rewards[k];
        brain::Values below(this->_neuron, 0.0); // the gradient of the step before (swapped with grad: no new buffer at each step)
        for (brain::UIType step = this->_steps; step > 0; --step) {
            const brain::Values& before = this->_states[step - 1];

            std::fill(below.begin(), below.end(), 0.0f);
            for (brain::UIType i = input; i < this->_neuron; ++i) {
                brain::FType sum = neurons[i].bias;

                if (grad[i] == 0) continue;
                if (known) sum = this->_sums[step][i];
                else for (brain::UIType k = this->_inStart[i]; k < this->_inStart[i + 1]; ++k) sum += links[this->_inSlot[k]].weight * before[links[this->_inSlot[k]].id];
                const bool direct = this->_directOutput && step == this->_steps && i >= this->_neuron - output; // the score as it is on the sum of an output
                const brain::FType slope = direct ? grad[i] : grad[i] * this->derivative_(sum, this->activate_(sum));

                biasGrad[i] += slope;
                for (brain::UIType k = this->_inStart[i]; k < this->_inStart[i + 1]; ++k) {
                    const brain::RawBrainLink& link = links[this->_inSlot[k]];

                    weightGrad[this->_inSlot[k]] += slope * before[link.id];
                    below[link.id] += slope * link.weight;
                }
            }
            std::swap(grad, below);
        }

        // mini-batch: kept for applyBatch(), nothing changes yet
        if (this->_batch) {
            this->_pending.resize(links.size() + this->_neuron, 0.0);
            for (brain::UIType k = 0; k < links.size(); ++k)
                if (std::isfinite(weightGrad[k])) this->_pending[k] += weightGrad[k];
            for (brain::UIType i = 0; i < this->_neuron; ++i)
                if (std::isfinite(biasGrad[i])) this->_pending[links.size() + i] += biasGrad[i];
            ++this->_pendingCount;
            return;
        }
        this->update_(neurons, links, biasGrad, weightGrad);
    }
    this->finish_(neurons, links, !onGpu);
}

_hot void brain::RawBrainNetwork::inputs(const brain::Values& inputs)
{
    this->brain::ANetwork::inputs(inputs);
    this->_sequence.clear();
}

_hot void brain::RawBrainNetwork::sequence(const brain::Dataset& steps)
{
    // check inputs (every step has the inputs of the network)
    if (steps.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no step given");
    }
    for (const brain::Values& step: steps) {
        if (this->_ioSize.first != 0 && step.size() != this->_ioSize.first) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "a step has " + std::to_string(step.size()) + " inputs, expected " + std::to_string(this->_ioSize.first));
        }
    }
    this->brain::ANetwork::inputs(steps.front());
    this->_sequence = steps;
}

_hot brain::Values brain::RawBrainNetwork::applySequence(const brain::Dataset& steps) const
{
    const brain::UIType input = this->_ioSize.first;

    // check inputs
    if (steps.empty() || input == 0 || this->_rawNeurons.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no step given, or the number of inputs of the network is not set (size() / input())");
    }
    for (const brain::Values& step: steps) {
        if (step.size() != input) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "a step has " + std::to_string(step.size()) + " inputs, expected " + std::to_string(input));
        }
    }
    const std::vector<brain::RawBrainNeuron> neurons = this->_rawNeurons.toVector();
    const std::vector<brain::RawBrainLink> links = this->_rawLinks.toVector();
    brain::Values state(this->_neuron, 0.0);
    brain::Values next(this->_neuron, 0.0);

    // the same steps as apply(), the inputs of the sequence then 0
    std::copy(steps.front().begin(), steps.front().end(), state.begin());
    for (brain::UIType step = 0; step < this->_steps; ++step) {
        for (brain::UIType i = 0; i < this->_neuron; ++i) {
            brain::FType sum = neurons[i].bias;

            if (i < input) {
                next[i] = step + 1 < steps.size() ? steps[step + 1][i] : 0.0f;
                continue;
            }
            for (brain::UIType k = this->_inStart[i]; k < this->_inStart[i + 1]; ++k)
                sum += links[this->_inSlot[k]].weight * state[links[this->_inSlot[k]].id];
            next[i] = this->activate_(sum);
            if (std::abs(next[i]) < this->_threshold) next[i] = 0.0; // too weak to matter
        }
        state.swap(next);
    }
    return brain::Values(state.end() - static_cast<std::ptrdiff_t>(this->_ioSize.second), state.end());
}

_hot void brain::RawBrainNetwork::watch_(void)
{
    const brain::Values& last = this->_states.back();

    if (this->_seen.size() != this->_neuron) this->_seen.assign(this->_neuron, {std::numeric_limits<brain::FType>::infinity(), -std::numeric_limits<brain::FType>::infinity()});
    for (brain::UIType i = this->_ioSize.first; i < this->_neuron; ++i) {
        this->_seen[i].first = std::min(this->_seen[i].first, last[i]);
        this->_seen[i].second = std::max(this->_seen[i].second, last[i]);
    }
}

_cold brain::UIType brain::RawBrainNetwork::revive(void)
{
    std::vector<brain::RawBrainNeuron> neurons;
    std::vector<brain::RawBrainLink> links;
    brain::UIType count = 0;

    if (this->_seen.size() != this->_neuron) return 0;
    neurons = this->_rawNeurons.toVector();
    links = this->_rawLinks.toVector();

    // a neuron stuck at 0 or at 1 on every execution since the last call (saturated: no gradient) gets its bias back to 0 and its inward weights halved
    for (brain::UIType i = this->_ioSize.first; i < this->_neuron; ++i) {
        const std::pair<brain::FType, brain::FType>& seen = this->_seen[i];

        if (!(seen.second < static_cast<brain::FType>(0.02) || seen.first > static_cast<brain::FType>(0.98)) || seen.second - seen.first > static_cast<brain::FType>(1e-3)) continue;
        neurons[i].bias = 0;
        ++count;
        for (brain::UIType slot = 0; slot < this->_cap; ++slot) {
            brain::RawBrainLink& link = links[i * this->_cap + slot];

            if (!link.inward) continue;
            link.weight = link.weight / 2 != 0 ? link.weight / 2 : std::numeric_limits<brain::FType>::min(); // a weight at 0 would look like a free slot
            links[link.id * this->_cap + cloneSlot(links, this->_cap, i, link)].weight = link.weight;
        }
    }
    this->_seen.clear();
    if (count == 0) return 0;
    this->_rawNeurons.store(neurons);
    this->_rawLinks.store(links);
    this->sync_(neurons, links);
    return count;
}

_hot void brain::RawBrainNetwork::update_(std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const brain::Values& biasGrad, const brain::Values& weightGrad)
{
    const brain::FType bound = this->_clampWeights ? this->_weightLimits.second : std::numeric_limits<brain::FType>::infinity();
    const brain::UIType size = links.size() + this->_neuron;
    const bool moving = this->_momentum != 0 && this->_adam == 0;
    brain::FType first = 1.0;
    this->_decayScale = 1 / (1 + this->_rateDecay * static_cast<brain::FType>(this->_updates++)); // the first update is not decayed // Adam: the corrections of the first steps (the means start at 0)
    brain::FType second = 1.0;

    if (moving && this->_velocity.size() != size) this->_velocity.assign(size, 0.0);
    if (this->_adam != 0) {
        if (this->_moments.size() != 2 * size) {
            this->_moments.assign(2 * size, 0.0);
            this->_adamSteps = 0;
        }
        ++this->_adamSteps;
        first = 1 - std::pow(static_cast<brain::FType>(0.9), static_cast<brain::FType>(this->_adamSteps));
        second = 1 - std::pow(static_cast<brain::FType>(0.999), static_cast<brain::FType>(this->_adamSteps));
    }

    // a positive score strengthens what made the outputs, a negative one weakens it (the link and its clone keep the same weight)
    for (brain::UIType i = this->_ioSize.first; i < this->_neuron; ++i) {
        if (std::isfinite(biasGrad[i])) {
            neurons[i].bias = std::clamp(neurons[i].bias + this->change_(links.size() + i, biasGrad[i], moving, first, second), -bound, bound);
        }
        for (brain::UIType slot = 0; slot < this->_cap; ++slot) {
            const brain::UIType at = i * this->_cap + slot;
            brain::RawBrainLink& link = links[at];
            brain::FType weight;

            if (!link.inward || !std::isfinite(weightGrad[at])) continue;
            weight = std::clamp(link.weight + this->change_(at, weightGrad[at], moving, first, second), -bound, bound);
            if (weight == 0) weight = std::numeric_limits<brain::FType>::min(); // a weight at 0 would look like a free slot
            link.weight = weight;
            links[this->_clones[at]].weight = weight;
        }
    }
}

_hot brain::FType brain::RawBrainNetwork::change_(const brain::UIType at, const brain::FType grad, const bool moving, const brain::FType first, const brain::FType second)
{
    brain::FType value = grad;

    // Adam (rate * mean / sqrt(mean of the squares): the score only gives the direction), else the momentum, then the step limit
    if (this->_adam != 0) {
        brain::FType& mean = this->_moments[2 * at];
        brain::FType& square = this->_moments[2 * at + 1];

        mean = static_cast<brain::FType>(0.9) * mean + static_cast<brain::FType>(0.1) * grad;
        square = static_cast<brain::FType>(0.999) * square + static_cast<brain::FType>(0.001) * grad * grad;
        value = this->_adam * this->_decayScale * (mean / first) / (std::sqrt(square / second) + static_cast<brain::FType>(1e-8));
    } else {
        value *= this->_decayScale;
        if (moving) value += this->_momentum * this->_velocity[at];
    }
    if (this->_stepLimit > 0) value = std::clamp(value, -this->_stepLimit, this->_stepLimit);
    if (moving) this->_velocity[at] = value;
    return value;
}

_cold void brain::RawBrainNetwork::finish_(std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links, const bool host)
{
    std::vector<brain::RawBrainLink> before;
    bool changed;

    if (!this->_velocity.empty() || !this->_moments.empty()) before = links;
    changed = this->relink_(neurons, links); // keeps _clones up to date itself

    // momentum / Adam: a slot whose link was removed or replaced starts again from rest
    for (brain::UIType k = 0; k < before.size() && changed; ++k) {
        if (before[k].id == links[k].id && before[k].inward == links[k].inward && occupied(before[k]) == occupied(links[k])) continue;
        if (!this->_velocity.empty()) this->_velocity[k] = 0;
        if (!this->_moments.empty()) this->_moments[2 * k] = this->_moments[2 * k + 1] = 0;
    }
    if (host) this->_rawNeurons.store(neurons);
    if (host || changed) this->_rawLinks.store(links);
    this->sync_(neurons, links);
}

_cold bool brain::RawBrainNetwork::relink_(const std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links)
{
    const brain::UIType input = this->_ioSize.first;
    const brain::UIType output = this->_ioSize.second;
    const brain::UIType count = neurons.size();
    const brain::FType min = this->_weightLimits.first;
    const brain::FType max = this->_weightLimits.second;
    bool changed = false;

    for (brain::UIType owner = 0; owner < count && this->_cap > 0; ++owner) {
        std::vector<std::pair<brain::FType, brain::UIType>> list; // <squared distance, slot> of the links of the neuron, the nearest first
        bool limit = false;

        // no link of the neuron at a limit: nothing changes for it
        for (brain::UIType slot = 0; slot < this->_cap && !limit; ++slot) {
            const brain::RawBrainLink& link = links[owner * this->_cap + slot];
            limit = occupied(link) && (std::abs(link.weight) <= min || std::abs(link.weight) >= max);
        }
        if (!limit) continue;

        for (brain::UIType slot = 0; slot < this->_cap; ++slot) {
            const brain::RawBrainLink& link = links[owner * this->_cap + slot];
            if (occupied(link)) list.push_back({neurons[owner] | neurons[link.id], slot});
        }
        std::sort(list.begin(), list.end(), [&](const std::pair<brain::FType, brain::UIType>& a, const std::pair<brain::FType, brain::UIType>& b) {
            return a.first != b.first ? a.first < b.first : links[owner * this->_cap + a.second].id < links[owner * this->_cap + b.second].id;
        });

        // the links as they are before any change of this neuron: <slot, id, absolute weight, too weak | too strong | neither>
        std::vector<std::tuple<brain::UIType, brain::UIType, brain::FType, int>> snapshot;
        for (const std::pair<brain::FType, brain::UIType>& entry: list) {
            const brain::RawBrainLink& link = links[owner * this->_cap + entry.second];
            const brain::FType weight = std::abs(link.weight);

            const int low = this->_invertedLinks ? 1 : -1; // inverted: a link too weak makes room for a new one, a link too strong is removed

            snapshot.push_back({entry.second, link.id, weight, weight <= min ? low : (weight >= max ? -low : 0)});
        }
        for (brain::UIType k = 0; k < snapshot.size(); ++k) {
            const brain::UIType slot = std::get<0>(snapshot[k]);
            const brain::RawBrainLink link = links[owner * this->_cap + slot];

            // removed by the other neuron since (its clone was)
            if (!occupied(link) || link.id != std::get<1>(snapshot[k]) || std::get<3>(snapshot[k]) == 0) continue;
            if (std::get<3>(snapshot[k]) < 0) {
                // too weak: removed (with its clone), unless it is the last link of one of the two neurons
                const std::pair<brain::UIType, brain::UIType> ownerDegree = degree(links, this->_cap, owner);
                const std::pair<brain::UIType, brain::UIType> otherDegree = degree(links, this->_cap, link.id);

                if (this->_minLinks > 0 && (ownerDegree.first + ownerDegree.second <= this->_minLinks || otherDegree.first + otherDegree.second <= this->_minLinks)) continue; // the forced minimum of links
                if (this->_orphans || (survives(degree(links, this->_cap, owner), owner < input, link.inward) && survives(degree(links, this->_cap, link.id), link.id < input, !link.inward))) {
                    const brain::UIType other = cloneSlot(links, this->_cap, owner, link);
                    const brain::RawBrainLink clone = links[link.id * this->_cap + other];
                    const brain::UIType joined = this->_keepPaths ? pairs(links, this->_cap, count, this->_ioSize) : 0;

                    links[link.id * this->_cap + other] = brain::RawBrainLink{};
                    links[owner * this->_cap + slot] = brain::RawBrainLink{};
                    if (this->_keepPaths && pairs(links, this->_cap, count, this->_ioSize) < joined) { // the last path of an input to an output: put back
                        links[link.id * this->_cap + other] = clone;
                        links[owner * this->_cap + slot] = link;
                        continue;
                    }
                    changed = true;
                }
                continue;
            }

            // too strong: the nearest neuron that is still free gets a link, a weight like the two nearest links of the list that did not cause it (or fewer)
            if (freeSlot(links, this->_cap, owner) == this->_cap) continue;
            brain::FType reference = 0.0;
            brain::UIType used = 0;
            for (brain::UIType distance = 1; distance < snapshot.size() && used < 2; ++distance) {
                for (const brain::IType side: {-1, 1}) {
                    const brain::IType at = static_cast<brain::IType>(k) + side * static_cast<brain::IType>(distance);

                    if (at < 0 || at >= static_cast<brain::IType>(snapshot.size()) || used >= 2 || std::get<3>(snapshot[at]) != 0) continue;
                    reference += std::get<2>(snapshot[at]);
                    ++used;
                }
            }
            reference = used == 0 ? (min + max) / 2 : reference / static_cast<brain::FType>(used);
            (void)this->nearest_(owner, neurons, [&](const brain::UIType other) { // the lists of the init (left to right: only the neurons on the left)
                const bool canReceive = owner >= input && other < count - output; // owner <- other
                const bool canSend = owner < count - output && other >= input; // owner -> other
                brain::UIType found = 0;
                brain::UIType otherSlot;

                while (found < this->_cap && !(occupied(links[owner * this->_cap + found]) && links[owner * this->_cap + found].id == other)) ++found;
                otherSlot = freeSlot(links, this->_cap, other);
                if (found < this->_cap || otherSlot == this->_cap || (!canReceive && !canSend) || (this->_leftToRight && !canReceive)) return false;
                const bool receive = this->_leftToRight || (canReceive && (!canSend || brain::tools::rand_floating() < 0.5f));
                const brain::FType low = min + (max - min) * static_cast<brain::FType>(0.05);
                const brain::FType high = min + (max - min) * static_cast<brain::FType>(0.95);
                const brain::FType start = this->random_(static_cast<brain::FType>(0.01)); // only the sign: the size comes from the nearest links
                const brain::FType value = (start < 0 ? -1 : 1) * (this->_lowNewLinks ? low : std::clamp(reference * (static_cast<brain::FType>(0.5) + static_cast<brain::FType>(0.5) * brain::tools::rand_floating()), low, high)); // between half and all of the reference (low: 5 % of the limits)

                const brain::UIType ownerSlot = freeSlot(links, this->_cap, owner);

                links[owner * this->_cap + ownerSlot] = {other, value, receive};
                links[other * this->_cap + otherSlot] = {owner, value, !receive};
                if (this->_clones.size() == links.size()) { // the clone of the new inward link (a removed one needs nothing: its slot is not inward anymore)
                    if (receive) this->_clones[owner * this->_cap + ownerSlot] = other * this->_cap + otherSlot;
                    else this->_clones[other * this->_cap + otherSlot] = owner * this->_cap + ownerSlot;
                    this->_clonesChanged = true;
                }
                changed = true;
                return true;
            });
        }
    }
    return changed;
}

_cold void brain::RawBrainNetwork::ensurePaths_(const std::vector<brain::RawBrainNeuron>& neurons, std::vector<brain::RawBrainLink>& links) const
{
    const brain::UIType count = neurons.size();
    const brain::UIType input = this->_ioSize.first;
    const brain::UIType output = this->_ioSize.second;
    const brain::UIType hops = std::max<brain::UIType>(2, std::min(this->_pathHops, this->_steps));
    const brain::FType none = std::numeric_limits<brain::FType>::max(); // no path: a finite mark (the Optimized build has -ffast-math, where a comparison with the infinity is folded away)
    std::vector<std::pair<brain::UIType, brain::UIType>> kept; // <sender, receiver> of the links of the paths already built: never given up

    // frees a slot of a neuron: its farthest link that is not on the path being built nor on an earlier one (with its clone)
    const std::function<void(const brain::UIType, const std::vector<brain::UIType>&)> drop = [&](const brain::UIType owner, const std::vector<brain::UIType>& path) {
        brain::UIType best = this->_cap;
        brain::FType far = -1;

        for (brain::UIType slot = 0; slot < this->_cap; ++slot) {
            const brain::RawBrainLink& link = links[owner * this->_cap + slot];

            const std::pair<brain::UIType, brain::UIType> pair = link.inward ? std::pair<brain::UIType, brain::UIType>{link.id, owner} : std::pair<brain::UIType, brain::UIType>{owner, link.id};

            if (!occupied(link) || std::find(path.begin(), path.end(), link.id) != path.end() || std::find(kept.begin(), kept.end(), pair) != kept.end() || (neurons[owner] | neurons[link.id]) <= far) continue;
            far = neurons[owner] | neurons[link.id];
            best = slot;
        }
        if (best == this->_cap) return;
        const brain::RawBrainLink link = links[owner * this->_cap + best];
        links[link.id * this->_cap + cloneSlot(links, this->_cap, owner, link)] = brain::RawBrainLink{};
        links[owner * this->_cap + best] = brain::RawBrainLink{};
    };

    // a removed link can cut an earlier path: checked again (3 passes at most)
    for (brain::UIType pass = 0; pass < 3; ++pass) {
        bool added = false;

        for (brain::UIType from = 0; from < input; ++from) {
            for (brain::UIType to = count - output; to < count; ++to) {
                const std::vector<brain::IType> distance = reach(links, this->_cap, count, from);
                std::vector<brain::Values> cost(hops + 1, brain::Values(count, none)); // smallest sum of squared distances to a neuron in k links (an existing inward link costs 0)
                std::vector<std::vector<brain::UIType>> previous(hops + 1, std::vector<brain::UIType>(count, count));
                brain::UIType best = 0;

                std::vector<brain::UIType> locked(count, 0); // links of the earlier paths on each neuron: a neuron full of them can't take a new link

                if (distance[to] >= 0 && static_cast<brain::UIType>(distance[to]) <= this->_steps) continue;
                for (const std::pair<brain::UIType, brain::UIType>& pair: kept) {
                    ++locked[pair.first];
                    ++locked[pair.second];
                }
                cost[0][from] = 0;
                for (brain::UIType k = 1; k <= hops; ++k) {
                    for (brain::UIType a = 0; a < count; ++a) {
                        if (cost[k - 1][a] == none || a >= count - output) continue;
                        for (brain::UIType b = input; b < count; ++b) {
                            brain::UIType slot = 0;
                            brain::FType step;

                            if (b == a || (b >= count - output && b != to) || (this->_leftToRight && !(neurons[a].pos.x < neurons[b].pos.x))) continue;
                            while (slot < this->_cap && !(occupied(links[b * this->_cap + slot]) && links[b * this->_cap + slot].id == a)) ++slot;
                            step = slot == this->_cap ? (neurons[a] | neurons[b]) : (links[b * this->_cap + slot].inward ? 0 : none);
                            if (slot == this->_cap && (locked[a] >= this->_cap || locked[b] >= this->_cap)) step = none;
                            if (step == none) continue;
                            if (cost[k - 1][a] + step < cost[k][b]) {
                                cost[k][b] = cost[k - 1][a] + step;
                                previous[k][b] = a;
                            }
                        }
                    }
                }
                for (brain::UIType k = 2; k <= hops; ++k) // at least one hidden neuron on the way when it can
                    if (cost[k][to] != none && (best == 0 || cost[k][to] < cost[best][to])) best = k;
                if (best == 0 && cost[1][to] != none) best = 1;
                if (best == 0) continue;
                std::vector<brain::UIType> path = {to};
                for (brain::UIType k = best, at = to; k > 0; --k) {
                    at = previous[k][at];
                    path.insert(path.begin(), at);
                }

                // the missing links of the path (weight 1.0 like the init: weightInit gives the random ones)
                for (brain::UIType p = 0; p + 1 < path.size(); ++p) {
                    const brain::UIType a = path[p];
                    const brain::UIType b = path[p + 1];
                    brain::UIType slot = 0;

                    while (slot < this->_cap && !(occupied(links[b * this->_cap + slot]) && links[b * this->_cap + slot].id == a)) ++slot;
                    kept.push_back({a, b});
                    if (slot < this->_cap) continue;
                    if (freeSlot(links, this->_cap, a) == this->_cap) drop(a, path);
                    if (freeSlot(links, this->_cap, b) == this->_cap) drop(b, path);
                    if (freeSlot(links, this->_cap, a) == this->_cap || freeSlot(links, this->_cap, b) == this->_cap) continue;
                    links[b * this->_cap + freeSlot(links, this->_cap, b)] = {a, 1.0, true};
                    links[a * this->_cap + freeSlot(links, this->_cap, a)] = {b, 1.0, false};
                    added = true;
                }
            }
        }
        if (!added) break;
    }
}

_cold void brain::RawBrainNetwork::farLinks(const brain::UIType count)
{
    this->_farLinks = count;
    this->init_(); // the links change: the neurons and the links are set up again
}

_cold void brain::RawBrainNetwork::pathInit(const brain::UIType hops)
{
    // check inputs
    if (hops == 1) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "a path needs at least 2 links (0 = off)");
    }
    this->_pathHops = hops;
    this->init_(); // the links change: the neurons and the links are set up again
}

_cold void brain::RawBrainNetwork::momentum(const brain::FType value)
{
    // check inputs
    if (!std::isfinite(value) || value < 0 || value >= 1) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the momentum should be in [0, 1)");
    }
    this->_momentum = value;
    this->_velocity.clear();
}

_cold void brain::RawBrainNetwork::adam(const brain::FType rate)
{
    // check inputs
    if (!std::isfinite(rate) || rate < 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the rate of Adam should be a finite number >= 0 (0 = off)");
    }
    this->_adam = rate;
    this->_moments.clear();
    this->_adamSteps = 0;
}

_cold void brain::RawBrainNetwork::rateDecay(const brain::FType decay)
{
    // check inputs
    if (!std::isfinite(decay) || decay < 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the rate decay should be a finite number >= 0 (0 = none)");
    }
    this->_rateDecay = decay;
}

_cold void brain::RawBrainNetwork::randomLinks(const bool enable)
{
    this->_randomLinks = enable;
    this->init_(); // the links change: the neurons and the links are set up again
}

_cold void brain::RawBrainNetwork::minLinks(const brain::UIType count)
{
    this->_minLinks = count;
    if (this->_randomLinks) this->init_(); // the least of the random init
}

_cold void brain::RawBrainNetwork::stepLimit(const brain::FType limit)
{
    // check inputs
    if (!std::isfinite(limit) || limit < 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the step limit should be a finite number >= 0");
    }
    this->_stepLimit = limit;
}

_cold void brain::RawBrainNetwork::batch(const bool enable)
{
    this->_batch = enable;
    this->_pending.clear();
    this->_pendingCount = 0;
}

_hot void brain::RawBrainNetwork::applyBatch(void)
{
    if (this->_pendingCount == 0) return;
    std::vector<brain::RawBrainNeuron> neurons = this->_rawNeurons.toVector();
    std::vector<brain::RawBrainLink> links = this->_rawLinks.toVector();
    const brain::FType count = static_cast<brain::FType>(this->_pendingCount);
    brain::Values weightGrad(links.size(), 0.0);
    brain::Values biasGrad(this->_neuron, 0.0);

    // the mean of the gradients given since the last call
    for (brain::UIType k = 0; k < links.size(); ++k)
        weightGrad[k] = this->_pending[k] / count;
    for (brain::UIType i = 0; i < this->_neuron; ++i)
        biasGrad[i] = this->_pending[links.size() + i] / count;
    this->_pending.clear();
    this->_pendingCount = 0;
    this->update_(neurons, links, biasGrad, weightGrad);
    this->finish_(neurons, links, true);
}

_cold void brain::RawBrainNetwork::leftToRight(const bool enable)
{
    if (enable == this->_leftToRight) return;
    this->_leftToRight = enable;
    this->init_(); // the links change: the neurons and the links are set up again
}

_cold void brain::RawBrainNetwork::near_(const std::vector<brain::RawBrainNeuron>& neurons)
{
    const brain::UIType kept = 2 * this->_cap + 8; // enough for the init and the new links in most cases (a full neighbour is skipped)

    this->_near.assign(neurons.size(), {});
    this->_nearAll.assign(neurons.size(), true);
    for (brain::UIType owner = 0; owner < neurons.size(); ++owner) {
        std::vector<std::pair<brain::FType, brain::UIType>> distances; // <squared distance, id> of the candidates
        brain::UIType size;

        for (brain::UIType other = 0; other < neurons.size(); ++other) {
            if (other != owner && (!this->_leftToRight || neurons[other].pos.x < neurons[owner].pos.x)) distances.push_back({neurons[owner] | neurons[other], other});
        }
        size = std::min(kept, distances.size());
        std::nth_element(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(size == 0 ? 0 : size - 1), distances.end()); // the <size> nearest in front (the id breaks the ties: a total order)
        std::sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(size)); // then sorted, the nearest first
        this->_nearAll[owner] = size == distances.size();
        this->_near[owner].reserve(size);
        for (brain::UIType k = 0; k < size; ++k)
            this->_near[owner].push_back(distances[k].second);
    }
}

_cold std::vector<brain::UIType> brain::RawBrainNetwork::candidates_(const brain::UIType owner, const std::vector<brain::RawBrainNeuron>& neurons) const
{
    std::vector<std::pair<brain::FType, brain::UIType>> distances;
    std::vector<brain::UIType> out;

    for (brain::UIType other = 0; other < neurons.size(); ++other) {
        if (other != owner && (!this->_leftToRight || neurons[other].pos.x < neurons[owner].pos.x)) distances.push_back({neurons[owner] | neurons[other], other});
    }
    std::sort(distances.begin(), distances.end());
    out.reserve(distances.size());
    for (const std::pair<brain::FType, brain::UIType>& entry: distances)
        out.push_back(entry.second);
    return out;
}

_hot bool brain::RawBrainNetwork::nearest_(const brain::UIType owner, const std::vector<brain::RawBrainNeuron>& neurons, const std::function<bool(const brain::UIType)>& visit) const
{
    const std::vector<brain::UIType>& first = this->_near[owner];
    std::vector<brain::UIType> all;

    // the kept ones, then (rarely) the rest in the same order
    for (const brain::UIType other: first)
        if (visit(other)) return true;
    if (this->_nearAll[owner]) return false;
    all = this->candidates_(owner, neurons);
    for (brain::UIType k = first.size(); k < all.size(); ++k)
        if (visit(all[k])) return true;
    return false;
}

_cold void brain::RawBrainNetwork::weightLimits(const brain::FType min, const brain::FType max)
{
    // check inputs
    if (!std::isfinite(min) || !std::isfinite(max) || min < 0 || max <= min) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the limits of the weights should be finite numbers with 0 <= min < max");
    }
    this->_weightLimits = {min, max};
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
            if (this->_clones.size() == links.size()) { // the slot of the clone is known since the last change of the links
                links[this->_clones[receiver * this->_cap + slot]].weight = link.weight;
                continue;
            }
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
