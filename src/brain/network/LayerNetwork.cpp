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
##  @file LayerNetwork.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _Exception
#define _System
#include <utils/utils.hpp>
#include "brain/network/LayerNetwork.hpp"
#include "brain/tools/compute.hpp"
#include "brain/NetworkDefine.hpp"
#include "brain/tools/bytes.hpp"
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <vector>
#include <cmath>

brain::LayerNetwork::LayerNetwork(const brain::Layers& layers, const utils::system::MemoryMode mode)
: ANetwork{mode}, _layers{layers}, _neurons{this->makeMemory_<brain::LayerNeuron>()}, _weights{this->makeMemory_<brain::FType>()}
{
    // check inputs
    if (layers.empty() || std::any_of(layers.begin(), layers.end(), [](const brain::UIType neurons) {return neurons == 0;})) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the network needs at least one layer and every layer needs at least one neuron");
    }
    this->init_();
}

_cold void brain::LayerNetwork::layout_(void)
{
    std::vector<brain::LayerNeuron> neurons;
    brain::UIType previous = this->_ioSize.first; // a neuron has one weight per output of the previous layer (the inputs for the first)
    brain::UIType weights = 0;
    brain::UIType product;

    // where the weights of each neuron start, layer after layer (a count that overflows the size is refused)
    for (const brain::UIType layer: this->_layers) {
        if (__builtin_mul_overflow(layer, previous, &product) || __builtin_add_overflow(weights, product, &product)) _unlikely {
            throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfMemory, "the weights of the layers overflow the size");
        }
        for (brain::UIType neuron = 0; neuron < layer; ++neuron)
            neurons.push_back({weights + neuron * previous, previous, 0.0});
        weights = product;
        previous = layer;
    }
    this->_neurons.allocate(neurons.size());
    this->_neurons.store(neurons);
    this->_weights.allocate(weights);
    this->_weights.store(std::vector<brain::FType>(weights, 1.0)); // 1.0 by default: weightInit gives the random ones
    this->_layoutInput = this->_ioSize.first;
    this->_contiguous = true;
    this->resetDevice_();
}

_cold void brain::LayerNetwork::resetDevice_(void)
{
    this->_device.reset();
    this->_devicePending = 0;
    this->_deviceKept = false;
}

_cold brain::Bytes brain::LayerNetwork::serialize(void) const
{
    const std::vector<brain::LayerNeuron> neurons = this->_neurons.toVector();
    const std::vector<brain::FType> weights = this->_weights.toVector();
    brain::Bytes out;

    // common part, then the layers, the neurons and the weights
    this->serializeHeader_(out, static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER));
    brain::tools::put<std::uint64_t>(out, this->_layers.size());
    for (const brain::UIType layer: this->_layers)
        brain::tools::put<std::uint64_t>(out, layer);
    brain::tools::put<std::uint64_t>(out, neurons.size());
    for (const brain::LayerNeuron& neuron: neurons) {
        brain::tools::put<std::uint64_t>(out, neuron.incoming);
        brain::tools::put<std::uint64_t>(out, neuron.count);
        brain::tools::put<brain::FType>(out, neuron.bias);
    }
    brain::tools::put<std::uint64_t>(out, weights.size());
    for (const brain::FType weight: weights)
        brain::tools::put<brain::FType>(out, weight);
    return out;
}

_cold void brain::LayerNetwork::deserialize(const brain::Bytes& data)
{
    brain::UIType offset = this->deserializeHeader_(data, static_cast<std::uint8_t>(BRAIN_MAGIC_LAYER));
    brain::UIType count = brain::tools::get<std::uint64_t>(data, offset);

    // the layers
    if (!brain::tools::fits(data, offset, count, sizeof(std::uint64_t))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "truncated layers");
    }
    brain::Layers layers(count);
    for (brain::UIType& layer: layers)
        layer = brain::tools::get<std::uint64_t>(data, offset);
    if (layers.empty() || std::any_of(layers.begin(), layers.end(), [](const brain::UIType neurons) {return neurons == 0;})) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the network needs at least one layer and every layer needs at least one neuron");
    }

    // the neurons
    count = brain::tools::get<std::uint64_t>(data, offset);
    if (!brain::tools::fits(data, offset, count, 2 * sizeof(std::uint64_t) + sizeof(brain::FType))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "truncated neurons");
    }
    std::vector<brain::LayerNeuron> neurons(count);
    for (brain::LayerNeuron& neuron: neurons) {
        neuron.incoming = brain::tools::get<std::uint64_t>(data, offset);
        neuron.count = brain::tools::get<std::uint64_t>(data, offset);
        neuron.bias = brain::tools::get<brain::FType>(data, offset);
    }

    // the weights
    count = brain::tools::get<std::uint64_t>(data, offset);
    if (!brain::tools::fits(data, offset, count, sizeof(brain::FType))) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "truncated weights");
    }
    std::vector<brain::FType> weights(count);
    for (brain::FType& weight: weights)
        weight = brain::tools::get<brain::FType>(data, offset);
    if (offset != data.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "unexpected data after the weights");
    }

    // the neurons match the layers and the weights they read (none stored yet: nothing to check)
    bool contiguous = true;
    if (!neurons.empty()) {
        brain::UIType previous = this->_ioSize.first; // the header is read: the number of inputs
        brain::UIType index = 0;

        if (neurons.size() != std::accumulate(layers.begin(), layers.end(), brain::UIType{0})) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the number of neurons does not match the layers");
        }
        for (const brain::UIType layer: layers) {
            for (brain::UIType k = 0; k < layer; ++k, ++index) {
                if (neurons[index].count != previous || neurons[index].incoming > weights.size() || neurons[index].count > weights.size() - neurons[index].incoming) _unlikely {
                    throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidNetworkFile, "the neuron " + std::to_string(index) + " does not match its layer or its weights");
                }
                if (neurons[index].incoming != (index == 0 ? 0 : neurons[index - 1].incoming + neurons[index - 1].count)) contiguous = false;
            }
            previous = layer;
        }
    }

    // everything is valid: the network is replaced
    this->_layers = layers;
    this->_neurons.allocate(neurons.size());
    this->_neurons.store(neurons);
    this->_weights.allocate(weights.size());
    this->_weights.store(weights);
    this->_layoutInput = this->_ioSize.first;
    this->_contiguous = contiguous;
    this->resetDevice_();
}

_hot brain::Values brain::LayerNetwork::forward_(const brain::Values& inputs, std::vector<brain::Values>* sums, std::vector<brain::Values>* outputs) const
{
    // check inputs (the number of inputs gives the weights of the first layer)
    if (this->_ioSize.first == 0 || this->_neurons.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "the number of inputs of the network is not set (size() / input())");
    }

    // the memories stored on the gpu are computed there (an external activation function can only run on the host)
    if (this->_neurons.isOnGpu() && !this->_activationFunction.second) {
        std::vector<brain::Values> gpuSums;
        std::vector<brain::Values> gpuOutputs;

        if (brain::tools::gpu::layers(this->_gpu.get(), this->_neurons.native(), this->_weights.native(), this->_layers, this->_activationFunction.first, inputs, gpuSums, gpuOutputs)) {
            brain::Values result = gpuOutputs.back();

            if (sums) *sums = std::move(gpuSums);
            if (outputs) *outputs = std::move(gpuOutputs);
            return result;
        }
    }

    // the memories are read on the host (a gpu memory is fetched once)
    std::vector<brain::LayerNeuron> fetchedNeurons;
    std::vector<brain::FType> fetchedWeights;
    const brain::LayerNeuron* neurons = this->_neurons.data();
    const brain::FType* weights = this->_weights.data();
    if (neurons == nullptr) {fetchedNeurons = this->_neurons.toVector(); neurons = fetchedNeurons.data();}
    if (weights == nullptr && !this->_weights.empty()) {fetchedWeights = this->_weights.toVector(); weights = fetchedWeights.data();}

    // each neuron: bias + the weights times the outputs of the previous layer, then the activation
    brain::Values previous = inputs;
    brain::UIType index = 0;
    if (sums) sums->clear();
    if (outputs) outputs->clear();
    for (const brain::UIType layer: this->_layers) {
        brain::Values sum(layer);
        brain::Values output(layer);

        for (brain::UIType k = 0; k < layer; ++k, ++index) {
            const brain::LayerNeuron& neuron = neurons[index];

            const brain::FType* row = weights + neuron.incoming;
            brain::FType total = neuron.bias; // a local sum: the compiler keeps it in a register (same order, same result)

            for (brain::UIType j = 0; j < neuron.count; ++j)
                total += row[j] * previous[j];
            sum[k] = total;
            output[k] = this->activate_(total);
        }
        if (sums) sums->push_back(std::move(sum));
        if (outputs) outputs->push_back(output);
        previous = std::move(output);
    }
    return previous;
}

_hot brain::Values brain::LayerNetwork::apply(void)
{
    // check inputs
    if (this->_inputs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "no inputs loaded (call inputs() first)");
    }
    return this->forward_(this->_inputs, &this->_sums, &this->_outputs);
}

_hot brain::Dataset brain::LayerNetwork::applyAll(const brain::Dataset& inputs) const
{
    brain::Dataset outputs;

    // the samples together on the gpu (a size that does not match is refused by the apply() of the host)
    if (!inputs.empty() && this->_ioSize.first != 0 && !this->_neurons.empty() && this->gpuReady_()) {
        const bool fits = std::all_of(inputs.begin(), inputs.end(), [this](const brain::Values& sample) {return sample.size() == this->_ioSize.first;});

        if (fits && brain::tools::gpu::layersAll(this->_gpu.get(), this->_neurons.native(), this->_weights.native(), this->_layers, this->_activationFunction.first, inputs, outputs, this->_profile == brain::GpuProfile::Fast ? BRAIN_GPU_FAST_EXEC_BYTES : BRAIN_GPU_LIGHT_EXEC_BYTES)) return outputs;
    }
    return this->ANetwork::applyAll(inputs);
}

_hot brain::Dataset brain::LayerNetwork::applyTrain(const brain::Dataset& inputs)
{
    brain::Dataset outputs;

    // the samples of the mini-batch together on the gpu, the gradient of the device is where they are kept
    this->_deviceKept = false;
    if (!inputs.empty() && this->_ioSize.first != 0 && !this->_neurons.empty() && this->batchTraining()) {
        const bool fits = std::all_of(inputs.begin(), inputs.end(), [this](const brain::Values& sample) {return sample.size() == this->_ioSize.first;});

        if (!this->_device) this->_device = std::make_unique<brain::tools::gpu::Batch>();
        if (fits && brain::tools::gpu::layersTrain(this->_gpu.get(), this->_neurons.native(), this->_weights.native(), this->_layers, this->_activationFunction.first, *this->_device, inputs, outputs, BRAIN_GPU_FAST_TRAIN_BYTES)) {
            this->_deviceKept = true;
            return outputs;
        }
    }
    return this->ANetwork::applyTrain(inputs);
}

_hot void brain::LayerNetwork::backpropagationAll(const brain::Dataset& coefs)
{
    // kept on the device: their gradient is made there
    if (this->_deviceKept) {
        this->_deviceKept = false;
        for (const brain::Values& sample: coefs) {
            if (sample.size() != this->_layers.back()) _unlikely {
                throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "given " + std::to_string(sample.size()) + ", the last layer has " + std::to_string(this->_layers.back()) + " neurons");
            }
            for (const brain::FType coef: sample) {
                if (!std::isfinite(coef)) _unlikely {
                    throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficients should be finite numbers");
                }
            }
        }
        if (!brain::tools::gpu::layersGradient(this->_gpu.get(), this->_neurons.native(), this->_weights.native(), this->_layers, this->_activationFunction.first, *this->_device, coefs)) _unlikely {
            throw utils::exception::ErrorException(utils::exception::InternalCode::GpuTransfer, "the gradient of the mini-batch failed on the device");
        }
        this->_devicePending += coefs.size();
        return;
    }
    this->ANetwork::backpropagationAll(coefs);
}

_hot void brain::LayerNetwork::backpropagation(const brain::FType coef)
{
    // check inputs
    if (!std::isfinite(coef)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficient should be a finite number");
    } else if (this->_layers.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "the network has no layer");
    }
    this->learn_(brain::Values(this->_layers.back(), coef));
}

_hot void brain::LayerNetwork::backpropagation(const brain::Values& coefs)
{
    // check inputs
    if (coefs.empty()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "no coefficients given");
    } else if (this->_ioSize.second != 0 && coefs.size() != this->_ioSize.second) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "given " + std::to_string(coefs.size()) + ", expected " + std::to_string(this->_ioSize.second));
    } else if (this->_layers.empty() || coefs.size() != this->_layers.back()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidOutputSize, "given " + std::to_string(coefs.size()) + ", the last layer has " + std::to_string(this->_layers.empty() ? 0 : this->_layers.back()) + " neurons");
    }
    for (const brain::FType coef: coefs) {
        if (!std::isfinite(coef)) _unlikely {
            throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidCoefficient, "the coefficients should be finite numbers");
        }
    }
    this->learn_(coefs);
}

_hot void brain::LayerNetwork::learn_(const brain::Values& errors)
{
    // the sums and the outputs of the last apply() are what the gradient is made of
    if (this->_sums.size() != this->_layers.size() || this->_outputs.size() != this->_layers.size()) _unlikely {
        throw utils::exception::ErrorException(utils::exception::ExternalCode::InvalidInputSize, "nothing to learn from, call apply() first");
    }

    // the memories stored on the gpu are changed there (an external activation function can only run on the host, a mini-batch only reads them)
    if (!this->_batch && this->_neurons.isOnGpu() && !this->_activationFunction.second && brain::tools::gpu::layersBack(this->_gpu.get(), this->_neurons.native(), this->_weights.native(), this->_layers, this->_activationFunction.first, this->_inputs, this->_sums, this->_outputs, errors)) return;

    // the memories of the host are changed in place, a gpu memory is fetched then stored back
    std::vector<brain::LayerNeuron> fetchedNeurons;
    std::vector<brain::FType> fetchedWeights;
    const bool host = this->_neurons.data() != nullptr && (this->_weights.empty() || this->_weights.data() != nullptr);
    if (!host) {
        fetchedNeurons = this->_neurons.toVector();
        fetchedWeights = this->_weights.toVector();
    }
    brain::LayerNeuron* neurons = host ? this->_neurons.data() : fetchedNeurons.data();
    brain::FType* weights = host ? this->_weights.data() : fetchedWeights.data();
    if (this->_batch && (this->_pendingWeights.size() != this->_weights.size() || this->_pendingBias.size() != this->_neurons.size())) { // the network was set up again since batch(true)
        this->_pendingWeights.assign(this->_weights.size(), 0.0f);
        this->_pendingBias.assign(this->_neurons.size(), 0.0f);
        this->_pendingCount = 0;
    }
    brain::Values delta = errors; // error of the outputs of the layer, from the last one to the first one
    brain::Values below; // error given to the previous layer (swapped with delta: no new buffer per layer)
    brain::UIType first = this->_neurons.size();

    for (brain::UIType l = this->_layers.size(); l-- > 0;) {
        const brain::UIType size = this->_layers[l];
        const brain::Values& previous = l == 0 ? this->_inputs : this->_outputs[l - 1];

        below.assign(l == 0 ? 0 : this->_layers[l - 1], 0.0);
        first -= size;
        for (brain::UIType k = 0; k < size; ++k) {
            brain::LayerNeuron& neuron = neurons[first + k];
            const brain::FType slope = delta[k] * this->derivative_(this->_sums[l][k], this->_outputs[l][k]);
            brain::FType* row = weights + neuron.incoming;

            if (this->_batch) { // the weights stay as they are: the gradient is added to the pending one
                brain::FType* pending = this->_pendingWeights.data() + neuron.incoming;

                for (brain::UIType j = 0; j < neuron.count; ++j) {
                    if (l > 0) below[j] += slope * row[j];
                    pending[j] += slope * previous[j];
                }
                this->_pendingBias[first + k] += slope;
            } else if (l > 0) {
                for (brain::UIType j = 0; j < neuron.count; ++j) {
                    below[j] += slope * row[j]; // with the weight before the update
                    row[j] -= slope * previous[j];
                }
            } else {
                for (brain::UIType j = 0; j < neuron.count; ++j)
                    row[j] -= slope * previous[j];
            }
            if (!this->_batch) neuron.bias -= slope; // in a mini-batch the bias only changes at applyBatch()
        }
        std::swap(delta, below);
    }
    if (this->_batch) {++this->_pendingCount; return;} // nothing changed
    if (host) return;
    this->_neurons.store(fetchedNeurons);
    this->_weights.store(fetchedWeights);
}

_cold void brain::LayerNetwork::batch(const bool enable)
{
    this->_batch = enable;
    this->_pendingWeights.assign(enable ? this->_weights.size() : 0, 0.0f);
    this->_pendingBias.assign(enable ? this->_neurons.size() : 0, 0.0f);
    this->_pendingCount = 0;
    if (this->_device) brain::tools::gpu::layersClear(*this->_device);
    this->_devicePending = 0;
    this->_deviceKept = false;
}

_hot void brain::LayerNetwork::applyBatch(void)
{
    std::vector<brain::LayerNeuron> neurons;
    std::vector<brain::FType> weights;
    const brain::UIType count = this->_pendingCount + this->_devicePending;

    // nothing pending: nothing done
    if (count == 0) return;

    // the gradients made on the device go down there (their weights never leave it)
    if (this->_devicePending > 0) {
        if (!this->_device || !brain::tools::gpu::layersApply(this->_gpu.get(), this->_neurons.native(), this->_weights.native(), *this->_device, this->_weights.size(), this->_neurons.size(), static_cast<brain::FType>(count))) _unlikely {
            throw utils::exception::ErrorException(utils::exception::InternalCode::GpuTransfer, "the mini-batch could not be applied on the device");
        }
        this->_devicePending = 0;
    }

    // the ones of the host
    if (this->_pendingCount > 0) {
        neurons = this->_neurons.toVector();
        weights = this->_weights.toVector();
        for (brain::UIType k = 0; k < weights.size() && k < this->_pendingWeights.size(); ++k) weights[k] -= this->_pendingWeights[k] / static_cast<brain::FType>(count);
        for (brain::UIType k = 0; k < neurons.size() && k < this->_pendingBias.size(); ++k) neurons[k].bias -= this->_pendingBias[k] / static_cast<brain::FType>(count);
        this->_weights.store(weights);
        this->_neurons.store(neurons);
        std::fill(this->_pendingWeights.begin(), this->_pendingWeights.end(), 0.0f);
        std::fill(this->_pendingBias.begin(), this->_pendingBias.end(), 0.0f);
        this->_pendingCount = 0;
    }
}

_cold void brain::LayerNetwork::weightInit(const brain::FType fallback)
{
    std::vector<brain::FType> weights;

    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }

    // a random value for every weight
    weights = this->_weights.toVector();
    for (brain::FType& weight: weights)
        weight = this->random_(fallback);
    this->_weights.store(weights);
}

_cold void brain::LayerNetwork::biasInit(const brain::FType fallback)
{
    std::vector<brain::LayerNeuron> neurons;

    // check inputs
    if (!std::isfinite(fallback) || fallback == 0) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::InvalidArgument, "the fallback should be a finite number different from 0");
    }

    // a random value for every bias
    neurons = this->_neurons.toVector();
    for (brain::LayerNeuron& neuron: neurons)
        neuron.bias = this->random_(fallback);
    this->_neurons.store(neurons);
}
