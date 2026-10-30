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
#include "brain/tools/compute.hpp"
#include "brain/NetworkDefine.hpp"
#include "brain/tools/bytes.hpp"
#include "brain/tools/sum.hpp"
#include <functional>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>
#include <string>
#include <cmath>

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
    brain::tools::put<std::uint64_t>(out, this->_steps);
    brain::tools::put<brain::FType>(out, this->_threshold);
    brain::tools::put<std::uint64_t>(out, neurons.size());
    for (const brain::BrainNeuron& neuron: neurons)
        brain::tools::put<brain::FType>(out, neuron.bias);
    brain::tools::put<std::uint64_t>(out, links.size());
    for (const brain::BrainLink& link: links) {
        brain::tools::put<std::uint64_t>(out, link.id);
        brain::tools::put<brain::FType>(out, link.weight);
    }
}

// The first link that breaks the roles of the inputs and the outputs: an input receives a link or an output sends one (empty: none)
_cold static std::string brokenRole(const std::vector<brain::BrainLink>& links, const brain::UIType neuron, const brain::UIType cap, const std::pair<brain::UIType, brain::UIType>& io)
{
    for (brain::UIType receiver = 0; receiver < neuron && cap > 0; ++receiver) {
        for (brain::UIType slot = 0; slot < cap; ++slot) {
            const brain::BrainLink& link = links[receiver * cap + slot];

            if (link.id == 0 && link.weight == 0) continue; // a free slot
            if (receiver < io.first) return "the input " + std::to_string(receiver) + " receives a link";
            if (link.id >= neuron - io.second) return "the output " + std::to_string(link.id) + " sends a link";
        }
    }
    return "";
}

_cold void brain::BrainNetwork::deserializeBody_(const brain::Bytes& data, brain::UIType offset)
{
    const brain::UIType neuron = brain::tools::get<std::uint64_t>(data, offset);
    const brain::UIType cap = brain::tools::get<std::uint64_t>(data, offset);
    const brain::UIType steps = brain::tools::get<std::uint64_t>(data, offset);
    const brain::FType threshold = brain::tools::get<brain::FType>(data, offset);
    brain::UIType count = brain::tools::get<std::uint64_t>(data, offset);

    // the neurons (none: not allocated yet)
    if (neuron == 0 || steps == 0 || !std::isfinite(threshold) || threshold < 0 || this->_ioSize.first > neuron || this->_ioSize.second > neuron - this->_ioSize.first) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "invalid number of neurons, time steps, signal threshold or inputs and outputs");
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
    const std::string broken = brokenRole(links, neuron, cap, this->_ioSize);
    if (!broken.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, broken + " (an input only sends, an output only receives)");
    }

    // everything is valid: the memories are replaced
    this->_neuron = neuron;
    this->_cap = cap;
    this->_steps = steps;
    this->_threshold = threshold;
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

_cold void brain::BrainNetwork::threshold(const brain::FType threshold)
{
    // check inputs
    if (!std::isfinite(threshold) || threshold < 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the signal threshold should be a finite number above or equal to 0");
    }
    this->_threshold = threshold;
}

_cold void brain::BrainNetwork::fits_(void) const
{
    // the inputs are the first neurons and the outputs the last ones: they can't overlap
    if (this->_ioSize.first > this->_neuron || this->_ioSize.second > this->_neuron - this->_ioSize.first) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the inputs and the outputs (" + std::to_string(this->_ioSize.first) + " + " + std::to_string(this->_ioSize.second) + ") don't fit in the " + std::to_string(this->_neuron) + " neurons");
    }
}

_cold void brain::BrainNetwork::resized_(void)
{
    // the links already there have to keep the roles of the new sizes (a reduced network is not set up again)
    this->fits_();
    const std::string broken = brokenRole(this->_links.toVector(), this->_neuron, this->_cap, this->_ioSize);
    if (!broken.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "these sizes don't match the links: " + broken);
    }
}

_hot brain::UIType brain::BrainNetwork::threadsFor_(void) const
{
    const brain::UIType hardware = std::max<brain::UIType>(1, std::thread::hardware_concurrency());
    const brain::UIType wanted = this->_threads != 0 ? this->_threads : (this->_neuron * this->_cap) / BRAIN_THREADS_MIN_WORK; // by default each thread has at least BRAIN_THREADS_MIN_WORK slots of links in a step

    if (this->_activationFunction.second) return 1; // a function of the user may throw or share a state: never from several threads
    return std::clamp<brain::UIType>(wanted, 1, std::min(hardware, std::max<brain::UIType>(1, (this->_neuron - this->_ioSize.first + BRAIN_THREADS_CHUNK - 1) / BRAIN_THREADS_CHUNK))); // never more threads than chunks of neurons
}

template<typename Activation>
_hot void brain::BrainNetwork::steps_(const Activation& activation, const brain::BrainNeuron* neurons, const brain::BrainLink* links, const brain::Values& inputs, brain::Values& state, brain::Values& next) const
{
    const brain::UIType input = this->_ioSize.first;
    const brain::FType threshold = this->_threshold;
    const brain::UIType cap = this->_cap;
    const bool settles = !this->_activationFunction.second; // a function that depends on its value only: a step equal to the one before (to the threshold) gives the same ones for ever
    const brain::UIType threads = this->threadsFor_();
    brain::Values sums(this->_neuron, 0.0);

    // the neurons [from, to) of a step: the sums, then the activations of the whole range, which the compiler vectorizes
    const std::function<void(const brain::UIType, const brain::UIType)> rows = [&](const brain::UIType from, const brain::UIType to) {
        for (brain::UIType i = from; i < to; ++i)
            sums[i] = neurons[i].bias + (links != nullptr ? brain::tools::rowSum(links + i * cap, cap, state.data()) : 0.0f);
        for (brain::UIType i = from; i < to; ++i) next[i] = activation(sums[i]);
        if (threshold > 0) {
            for (brain::UIType i = from; i < to; ++i)
                if (std::abs(next[i]) < threshold) next[i] = 0.0; // too weak to matter
        }
    };
    const std::function<bool(const brain::UIType)> end = [&](const brain::UIType step) { // the end of a step: nothing moved means the next steps are copies
        const bool same = settles && step > 0 && brain::tools::settled(next.data(), state.data(), this->_neuron, threshold);

        state.swap(next);
        return same;
    };

    // each step, every neuron reads the previous step: the inputs stay at their value, the others are bias + weights times the outputs of their links, then the activation
    if (threads > 1) {
        const brain::UIType first = input;
        const brain::UIType part = (((this->_neuron - first + threads - 1) / threads) + BRAIN_THREADS_CHUNK - 1) / BRAIN_THREADS_CHUNK * BRAIN_THREADS_CHUNK; // the neurons after the inputs, shared between the threads in whole chunks

        brain::tools::stepThreads(threads, this->_steps, [&](const brain::UIType thread, const brain::UIType) {
            if (thread == 0) for (brain::UIType i = 0; i < input; ++i) next[i] = inputs[i];
            rows(std::min(this->_neuron, first + thread * part), std::min(this->_neuron, first + (thread + 1) * part));
        }, end);
        return;
    }
    for (brain::UIType step = 0; step < this->_steps; ++step) {
        for (brain::UIType i = 0; i < input; ++i) next[i] = inputs[i];
        rows(input, this->_neuron);
        if (end(step)) break;
    }
}

template<typename Activation>
_hot void brain::BrainNetwork::stepsAll_(const Activation& activation, const brain::BrainNeuron* neurons, const brain::BrainLink* links, const brain::UIType count, std::vector<brain::FType>& state, std::vector<brain::FType>& next) const
{
    const brain::UIType input = this->_ioSize.first;
    const brain::FType threshold = this->_threshold;
    const brain::UIType cap = this->_cap;
    const bool settles = !this->_activationFunction.second; // a function that depends on its value only
    std::vector<brain::FType> partial(4 * count, 0.0); // the four partial sums of a neuron for every sample (the order of the sums of apply())
    std::vector<brain::FType> sums(this->_neuron * count, 0.0);

    // the inputs stay at their value: the rows of the inputs are the same in both buffers
    std::copy(state.begin(), state.begin() + static_cast<std::ptrdiff_t>(input * count), next.begin());
    for (brain::UIType step = 0; step < this->_steps; ++step) {
        for (brain::UIType i = input; i < this->_neuron; ++i) {
            const brain::BrainLink* row = links != nullptr ? links + i * cap : nullptr;
            brain::UIType used = 0;

            std::fill(partial.begin(), partial.end(), 0.0f);
            while (row != nullptr && used < cap && row[used].weight != 0) { // the inward links are packed first: the rest of the slots is free
                brain::FType* part = partial.data() + (used & 3) * count;
                const brain::FType* source = state.data() + row[used].id * count;
                const brain::FType weight = row[used].weight;

                for (brain::UIType b = 0; b < count; ++b) part[b] += weight * source[b]; // the samples are contiguous: vectorized
                ++used;
            }
            for (brain::UIType b = 0; b < count; ++b)
                sums[i * count + b] = neurons[i].bias + ((partial[b] + partial[count + b]) + (partial[2 * count + b] + partial[3 * count + b]));
        }
        for (brain::UIType k = input * count; k < this->_neuron * count; ++k) next[k] = activation(sums[k]);
        if (threshold > 0) {
            for (brain::UIType k = input * count; k < this->_neuron * count; ++k)
                if (std::abs(next[k]) < threshold) next[k] = 0.0; // too weak to matter
        }
        const bool same = settles && step > 0 && brain::tools::settled(next.data(), state.data(), this->_neuron * count, threshold); // nothing moved for any sample: the next steps are copies
        state.swap(next);
        if (same) break;
    }
}

_hot brain::Dataset brain::BrainNetwork::applyAll(const brain::Dataset& inputs) const
{
    const brain::UIType input = this->_ioSize.first;
    const brain::UIType output = this->_ioSize.second;
    const brain::UIType count = inputs.size();
    brain::Dataset outputs;

    // check inputs (the same as apply(), for every sample)
    if (count == 0) return outputs;
    if (input == 0 || this->_neurons.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "the number of inputs of the network is not set (size() / input())");
    } else if (output == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "the number of outputs of the network is not set (size() / output())");
    }
    for (const brain::Values& sample: inputs) {
        if (sample.empty() || sample.size() != input) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "given " + std::to_string(sample.size()) + ", expected " + std::to_string(input));
        }
    }

    // a gpu memory: one sample after the other on the device (its kernels run one sample at a time)
    if (this->_neurons.isOnGpu() && !this->_activationFunction.second) return this->brain::ANetwork::applyAll(inputs);

    // the memories are read on the host (a gpu memory is fetched once)
    std::vector<brain::BrainNeuron> fetchedNeurons;
    std::vector<brain::BrainLink> fetchedLinks;
    const brain::BrainNeuron* neurons = this->_neurons.data();
    const brain::BrainLink* links = this->_links.data();
    if (neurons == nullptr) {fetchedNeurons = this->_neurons.toVector(); neurons = fetchedNeurons.data();}
    if (links == nullptr && !this->_links.empty()) {fetchedLinks = this->_links.toVector(); links = fetchedLinks.data();}

    // the state of neuron i for the sample b is state[i * count + b]
    std::vector<brain::FType> state(this->_neuron * count, 0.0);
    std::vector<brain::FType> next(this->_neuron * count, 0.0);
    for (brain::UIType b = 0; b < count; ++b)
        for (brain::UIType i = 0; i < input; ++i) state[i * count + b] = inputs[b][i];
    if (this->_activationFunction.second) this->stepsAll_(brain::tools::ExternalActivation{this->_activationFunction.second}, neurons, links, count, state, next);
    else if (this->_activationFunction.first == brain::ActivationFunction::Sigmoid) this->stepsAll_(brain::tools::SigmoidActivation{}, neurons, links, count, state, next);
    else if (this->_activationFunction.first == brain::ActivationFunction::ReLU) this->stepsAll_(brain::tools::ReluActivation{}, neurons, links, count, state, next);
    else this->stepsAll_(brain::tools::LinearActivation{}, neurons, links, count, state, next);
    outputs.assign(count, brain::Values(output));
    for (brain::UIType b = 0; b < count; ++b)
        for (brain::UIType o = 0; o < output; ++o) outputs[b][o] = state[(this->_neuron - output + o) * count + b];
    return outputs;
}

_hot brain::Values brain::BrainNetwork::compute_(const brain::Values& inputs) const
{
    const brain::UIType input = this->_ioSize.first;
    const brain::UIType output = this->_ioSize.second;

    // check inputs (the first neurons are the inputs and the last ones the outputs: both sizes are needed)
    if (input == 0 || this->_neurons.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "the number of inputs of the network is not set (size() / input())");
    } else if (output == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "the number of outputs of the network is not set (size() / output())");
    }

    // the memories stored on the gpu are computed there (an external activation function can only run on the host)
    if (this->_neurons.isOnGpu() && !this->_activationFunction.second) {
        std::vector<brain::Values> gpuStates;

        if (brain::tools::gpu::steps(this->_gpu.get(), this->_neurons.native(), this->_links.native(), false, this->_neuron, this->_cap, input, this->_steps, this->_activationFunction.first, this->_threshold, inputs, gpuStates, false))
            return brain::Values(gpuStates.back().end() - static_cast<std::ptrdiff_t>(output), gpuStates.back().end());
    }

    // the memories are read on the host (a gpu memory is fetched once)
    std::vector<brain::BrainNeuron> fetchedNeurons;
    std::vector<brain::BrainLink> fetchedLinks;
    const brain::BrainNeuron* neurons = this->_neurons.data();
    const brain::BrainLink* links = this->_links.data();
    if (neurons == nullptr) {fetchedNeurons = this->_neurons.toVector(); neurons = fetchedNeurons.data();}
    if (links == nullptr && !this->_links.empty()) {fetchedLinks = this->_links.toVector(); links = fetchedLinks.data();}

    // each step, every neuron reads the previous step: the inputs stay at their value, the others are bias + weights times the outputs of their links, then the activation
    brain::Values state(this->_neuron, 0.0);
    brain::Values next(this->_neuron, 0.0);
    std::copy(inputs.begin(), inputs.end(), state.begin());
    // the activation function is chosen once for the whole execution (a type: it is inlined in the loop over the neurons)
    if (this->_activationFunction.second) this->steps_(brain::tools::ExternalActivation{this->_activationFunction.second}, neurons, links, inputs, state, next);
    else if (this->_activationFunction.first == brain::ActivationFunction::Sigmoid) this->steps_(brain::tools::SigmoidActivation{}, neurons, links, inputs, state, next);
    else if (this->_activationFunction.first == brain::ActivationFunction::ReLU) this->steps_(brain::tools::ReluActivation{}, neurons, links, inputs, state, next);
    else this->steps_(brain::tools::LinearActivation{}, neurons, links, inputs, state, next); // linear, none: the value
    return brain::Values(state.end() - static_cast<std::ptrdiff_t>(output), state.end());
}
