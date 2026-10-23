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
##  @file compute.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#define _System
#define _Encapsulation
#define _Exception
#include <utils/utils.hpp>
#include "brain/tools/compute.hpp"
#include <algorithm>
#include <sstream>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>

// The CUDA headers are optional (the cmake adds the toolkit include directory when it finds it): without them nothing is computed on the gpu and the host computes
#if __has_include(<cuda.h>) && __has_include(<nvrtc.h>)
    #include <cuda.h>
    #include <nvrtc.h>
    #define BRAIN_CUDA
#endif

#ifdef BRAIN_CUDA

        // Kernels of the networks (the structs are the ones of brain/NetworkType.hpp, the types come from the macros)
        static const char* const KERNELS = R"CUDA(
        typedef FT ft;
        typedef unsigned long long ul;

        __device__ ft activate(const ft value, const int mode)
        {
            if (mode == 1) return (ft)1 / ((ft)1 + exp(-value));
            if (mode == 3) return value > 0 ? value : (ft)0;
            return value;
        }

        struct LayerNeuron {ul incoming; ul count; ft bias;};

        extern "C" __global__ void layer(const LayerNeuron* neurons, const ft* weights, const ft* previous, ft* sums, ft* outputs, const ul first, const ul size, const int mode)
        {
            const ul k = (ul)blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= size) return;
            const LayerNeuron neuron = neurons[first + k];
            ft sum = neuron.bias;

            for (ul j = 0; j < neuron.count; ++j)
                sum += weights[neuron.incoming + j] * previous[j];
            sums[first + k] = sum;
            outputs[first + k] = activate(sum, mode);
        }

        extern "C" __global__ void step(const char* neurons, const ul nstride, const ul boff, const char* links, const ul lstride, const ul idoff, const ul woff, const long long ioff, const ul cap, const ft* state, ft* next, const ul n, const ul input, const int mode, const ft threshold)
        {
            const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            if (i < input) {next[i] = state[i]; return;}
            ft sum = *(const ft*)(neurons + i * nstride + boff);

            for (ul slot = 0; slot < cap; ++slot) {
                const char* link = links + (i * cap + slot) * lstride;
                if (ioff >= 0 && !*(const unsigned char*)(link + ioff)) continue;
                sum += *(const ft*)(link + woff) * state[*(const ul*)(link + idoff)];
            }
            ft value = activate(sum, mode);
            if (fabs(value) < threshold) value = 0;
            next[i] = value;
        }
    __device__ ft derive(const ft sum, const ft out, const int mode)
    {
        if (mode == 1) return out * ((ft)1 - out);
        if (mode == 3) return sum > 0 ? (ft)1 : (ft)0;
        return (ft)1;
    }

    extern "C" __global__ void layer_slope(const ft* delta, const ft* sums, const ft* outputs, ft* slope, const ul size, const int mode)
    {
        const ul k = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= size) return;
        slope[k] = delta[k] * derive(sums[k], outputs[k], mode);
    }

    extern "C" __global__ void layer_below(const LayerNeuron* neurons, const ft* weights, const ft* slope, ft* below, const ul first, const ul size, const ul belowSize)
    {
        const ul j = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= belowSize) return;
        ft sum = 0;

        for (ul k = 0; k < size; ++k)
            sum += slope[k] * weights[neurons[first + k].incoming + j];
        below[j] = sum;
    }

    extern "C" __global__ void layer_update(LayerNeuron* neurons, ft* weights, const ft* slope, const ft* previous, const ul first, const ul size)
    {
        const ul k = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= size) return;
        LayerNeuron neuron = neurons[first + k];

        for (ul j = 0; j < neuron.count; ++j)
            weights[neuron.incoming + j] -= slope[k] * previous[j];
        neurons[first + k].bias -= slope[k];
    }

    extern "C" __global__ void back_step(const char* neurons, const ul nstride, const ul boff, const char* links, const ul lstride, const ul idoff, const ul woff, const ul ioff, const ul cap, const ft* before, const ft* grad, ft* below, ft* biasGrad, ft* weightGrad, const ul n, const ul input, const int mode)
    {
        const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n || i < input || grad[i] == 0) return;
        ft sum = *(const ft*)(neurons + i * nstride + boff);

        for (ul slot = 0; slot < cap; ++slot) {
            const char* link = links + (i * cap + slot) * lstride;
            if (*(const unsigned char*)(link + ioff)) sum += *(const ft*)(link + woff) * before[*(const ul*)(link + idoff)];
        }
        const ft slope = grad[i] * derive(sum, activate(sum, mode), mode);
        biasGrad[i] += slope;
        for (ul slot = 0; slot < cap; ++slot) {
            const char* link = links + (i * cap + slot) * lstride;
            if (!*(const unsigned char*)(link + ioff)) continue;
            const ul id = *(const ul*)(link + idoff);
            weightGrad[i * cap + slot] += slope * before[id];
            atomicAdd(&below[id], slope * *(const ft*)(link + woff));
        }
    }

    extern "C" __global__ void back_update(char* neurons, const ul nstride, const ul boff, char* links, const ul lstride, const ul idoff, const ul woff, const ul ioff, const ul cap, const ft* biasGrad, const ft* weightGrad, const ul n, const ul input)
    {
        const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n || i < input) return;
        if (isfinite(biasGrad[i])) *(ft*)(neurons + i * nstride + boff) += biasGrad[i];
        for (ul slot = 0; slot < cap; ++slot) {
            char* link = links + (i * cap + slot) * lstride;
            if (!*(const unsigned char*)(link + ioff) || !isfinite(weightGrad[i * cap + slot])) continue;
            ft weight = *(ft*)(link + woff) + weightGrad[i * cap + slot];
            const ul id = *(const ul*)(link + idoff);

            if (weight == 0) weight = (ft)FT_MIN; // a weight at 0 would look like a free slot
            *(ft*)(link + woff) = weight;
            for (ul other = 0; other < cap; ++other) {
                char* clone = links + (id * cap + other) * lstride;
                if (*(const ul*)(clone + idoff) == i && !*(const unsigned char*)(clone + ioff) && *(const ft*)(clone + woff) != 0) {
                    *(ft*)(clone + woff) = weight;
                    break;
                }
            }
        }
    }
    extern "C" __global__ void back_flags(const char* links, const ul lstride, const ul idoff, const ul woff, const ul ioff, const ul cap, const ft low, const ft high, unsigned char* flags, const ul n)
    {
    const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned char flag = 0;

    for (ul slot = 0; slot < cap; ++slot) {
        const char* link = links + (i * cap + slot) * lstride;
        const ft weight = fabs(*(const ft*)(link + woff));
        if (!*(const unsigned char*)(link + ioff) && *(const ul*)(link + idoff) == 0 && weight == 0) continue; // a free slot
        if (weight <= low || weight >= high) flag = 1;
    }
    flags[i] = flag;
    }

    extern "C" __global__ void reduce_copy(const char* neurons, const ul nstride, const ul boff, const char* links, const ul lstride, const ul idoff, const ul woff, const ul ioff, const ul cap, char* reducedNeurons, const ul rnstride, const ul rboff, char* reducedLinks, const ul rlstride, const ul ridoff, const ul rwoff, const ul n)
    {
    const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ul used = 0;

    *(ft*)(reducedNeurons + i * rnstride + rboff) = *(const ft*)(neurons + i * nstride + boff);
    for (ul slot = 0; slot < cap; ++slot) {
        const char* link = links + (i * cap + slot) * lstride;
        if (!*(const unsigned char*)(link + ioff)) continue;
        char* target = reducedLinks + (i * cap + used++) * rlstride;
        *(ul*)(target + ridoff) = *(const ul*)(link + idoff);
        *(ft*)(target + rwoff) = *(const ft*)(link + woff);
    }
    for (; used < cap; ++used) {
        char* target = reducedLinks + (i * cap + used) * rlstride;
        *(ul*)(target + ridoff) = 0;
        *(ft*)(target + rwoff) = 0; // a free slot is all at 0
    }
    }
    )CUDA";

        namespace {

        // Driver (CUDA) and compiler (NVRTC) functions, loaded at runtime from their libraries: the programs start without them
        struct Api {
            std::unique_ptr<utils::encapsulation::SharedObject> driver = nullptr;
            std::unique_ptr<utils::encapsulation::SharedObject> compiler = nullptr;
            decltype(&cuInit) init = nullptr;
            decltype(&cuPointerGetAttribute) pointerAttribute = nullptr;
            decltype(&cuCtxPushCurrent) push = nullptr;
            decltype(&cuCtxPopCurrent) pop = nullptr;
            decltype(&cuCtxGetDevice) device = nullptr;
            decltype(&cuCtxSynchronize) synchronize = nullptr;
            decltype(&cuDeviceGetAttribute) attribute = nullptr;
            decltype(&cuModuleLoadData) load = nullptr;
            decltype(&cuModuleGetFunction) function = nullptr;
            decltype(&cuLaunchKernel) launch = nullptr;
            decltype(&cuMemAlloc) allocate = nullptr;
            decltype(&cuMemFree) release = nullptr;
            decltype(&cuMemcpyHtoD) store = nullptr;
            decltype(&cuMemcpyDtoH) fetch = nullptr;
            decltype(&cuMemsetD8) zero = nullptr;
            decltype(&nvrtcCreateProgram) create = nullptr;
            decltype(&nvrtcCompileProgram) compile = nullptr;
            decltype(&nvrtcGetCUBINSize) imageSize = nullptr;
            decltype(&nvrtcGetCUBIN) image = nullptr;
            decltype(&nvrtcDestroyProgram) destroy = nullptr;
        };

        // The functions of one context: compiled once for it
        struct Module {
            CUfunction layer = nullptr;
            CUfunction step = nullptr;
            CUfunction layerSlope = nullptr;
            CUfunction layerBelow = nullptr;
            CUfunction layerUpdate = nullptr;
            CUfunction backStep = nullptr;
            CUfunction backUpdate = nullptr;
        CUfunction backFlags = nullptr;
        CUfunction reduceCopy = nullptr;
        };

        constexpr unsigned int BLOCK = 256; // threads per block

        // Load a library from the first name found
        _cold std::unique_ptr<utils::encapsulation::SharedObject> loadLibrary(const std::vector<std::string>& names)
        {
            for (const std::string& name: names) {
                try {return std::make_unique<utils::encapsulation::SharedObject>(name);}
                catch (const utils::exception::IException& e) {continue;} // next name
            }
            return nullptr;
        }

        // The driver and the compiler, loaded once (nullptr: not available)
        _cold const Api* api(void)
        {
            static std::unique_ptr<Api> loaded;
            static std::once_flag once;

            std::call_once(once, [&]() {
                std::unique_ptr<Api> candidate = std::make_unique<Api>();

                candidate->driver = loadLibrary({"libcuda.so.1", "libcuda.so", "nvcuda.dll"});
                candidate->compiler = loadLibrary({"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", "nvrtc64_130_0.dll", "nvrtc64_120_0.dll"});
                if (!candidate->driver || !candidate->compiler) return;
                try {
                    utils::encapsulation::SharedObject& d = *candidate->driver;
                    utils::encapsulation::SharedObject& c = *candidate->compiler;

                    candidate->init = d.loadFunction<decltype(&cuInit)>("cuInit");
                    candidate->pointerAttribute = d.loadFunction<decltype(&cuPointerGetAttribute)>("cuPointerGetAttribute");
                    candidate->push = d.loadFunction<decltype(&cuCtxPushCurrent)>("cuCtxPushCurrent_v2");
                    candidate->pop = d.loadFunction<decltype(&cuCtxPopCurrent)>("cuCtxPopCurrent_v2");
                    candidate->device = d.loadFunction<decltype(&cuCtxGetDevice)>("cuCtxGetDevice");
                    candidate->synchronize = d.loadFunction<decltype(&cuCtxSynchronize)>("cuCtxSynchronize");
                    candidate->attribute = d.loadFunction<decltype(&cuDeviceGetAttribute)>("cuDeviceGetAttribute");
                    candidate->load = d.loadFunction<decltype(&cuModuleLoadData)>("cuModuleLoadData");
                    candidate->function = d.loadFunction<decltype(&cuModuleGetFunction)>("cuModuleGetFunction");
                    candidate->launch = d.loadFunction<decltype(&cuLaunchKernel)>("cuLaunchKernel");
                    candidate->allocate = d.loadFunction<decltype(&cuMemAlloc)>("cuMemAlloc_v2");
                    candidate->release = d.loadFunction<decltype(&cuMemFree)>("cuMemFree_v2");
                    candidate->store = d.loadFunction<decltype(&cuMemcpyHtoD)>("cuMemcpyHtoD_v2");
                    candidate->fetch = d.loadFunction<decltype(&cuMemcpyDtoH)>("cuMemcpyDtoH_v2");
                    candidate->zero = d.loadFunction<decltype(&cuMemsetD8)>("cuMemsetD8_v2");
                    candidate->create = c.loadFunction<decltype(&nvrtcCreateProgram)>("nvrtcCreateProgram");
                    candidate->compile = c.loadFunction<decltype(&nvrtcCompileProgram)>("nvrtcCompileProgram");
                    candidate->imageSize = c.loadFunction<decltype(&nvrtcGetCUBINSize)>("nvrtcGetCUBINSize");
                    candidate->image = c.loadFunction<decltype(&nvrtcGetCUBIN)>("nvrtcGetCUBIN");
                    candidate->destroy = c.loadFunction<decltype(&nvrtcDestroyProgram)>("nvrtcDestroyProgram");
                } catch (const utils::exception::IException& e) {return;} // a function is missing: nothing computed on the gpu
                if (candidate->init(0) != CUDA_SUCCESS) return;
                loaded = std::move(candidate);
            });
            return loaded.get();
        }

        // A failed call of the driver: the caller computes on the host instead
        _cold void check(const int result, const char* what)
        {
            if (result != 0) _unlikely {
                throw utils::exception::ErrorException(utils::exception::InternalCode::GpuTransfer, std::string(what) + " failed (" + std::to_string(result) + ")");
            }
        }

        // Make a context current on the calling thread, then restore the previous one
        class Bind {
            private:
                const Api& _api;

            public:
                Bind(const Api& api, const CUcontext context): _api{api} {check(this->_api.push(context), "cuCtxPushCurrent");};
                ~Bind() noexcept {CUcontext previous = nullptr; (void)this->_api.pop(&previous);};
                Bind(const Bind& other) = delete;
                Bind& operator=(const Bind& other) = delete;
        };

        // A buffer of the device, released at the end of the computing
        class Buffer {
            private:
                const Api& _api;
                CUdeviceptr _ptr = 0;

            public:
                _cold _nodiscard inline CUdeviceptr get(void) const {return this->_ptr;};
                _cold _nodiscard inline CUdeviceptr at(const std::size_t bytes) const {return this->_ptr + bytes;}; // address bytes after the start

                Buffer(const Api& api, const std::size_t bytes): _api{api} {check(this->_api.allocate(&this->_ptr, std::max<std::size_t>(bytes, 1)), "cuMemAlloc");};
                ~Buffer() noexcept {if (this->_ptr != 0) (void)this->_api.release(this->_ptr);};
                Buffer(const Buffer& other) = delete;
                Buffer& operator=(const Buffer& other) = delete;
        };

        // The functions compiled for a context (cached, a failure is cached too)
        _cold const Module* compiled(const Api& api, const CUcontext context)
        {
            static std::map<CUcontext, std::unique_ptr<Module>> cache;
            static std::mutex lock;
            std::lock_guard guard(lock);
            std::map<CUcontext, std::unique_ptr<Module>>::iterator it = cache.find(context);

            if (it != cache.end()) return it->second.get();
            std::unique_ptr<Module> built = nullptr;
            try {
                const Bind bind(api, context);
                int device = 0;
                int major = 0;
                int minor = 0;

                check(api.device(&device), "cuCtxGetDevice");
                check(api.attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute");
                check(api.attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute");

                // compile the kernels for this device
                const std::string architecture = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
                const std::string type = std::string("-DFT=") + (sizeof(brain::FType) == sizeof(double) ? "double" : "float");
                std::ostringstream smallest;
                smallest.precision(std::numeric_limits<brain::FType>::max_digits10);
                smallest << "-DFT_MIN=" << std::numeric_limits<brain::FType>::min();
                const std::string minimum = smallest.str();
                const char* options[] = {architecture.c_str(), type.c_str(), minimum.c_str()};
                nvrtcProgram program = nullptr;

                check(api.create(&program, KERNELS, "brain.cu", 0, nullptr, nullptr), "nvrtcCreateProgram");
                try {
                    std::size_t size = 0;

                    check(api.compile(program, 3, options), "nvrtcCompileProgram");
                    check(api.imageSize(program, &size), "nvrtcGetCUBINSize");
                    std::vector<char> image(size);
                    check(api.image(program, image.data()), "nvrtcGetCUBIN");
                    CUmodule loaded = nullptr;
                    built = std::make_unique<Module>();

                    check(api.load(&loaded, image.data()), "cuModuleLoadData");
                    check(api.function(&built->layer, loaded, "layer"), "cuModuleGetFunction");
                    check(api.function(&built->step, loaded, "step"), "cuModuleGetFunction");
                    check(api.function(&built->layerSlope, loaded, "layer_slope"), "cuModuleGetFunction");
                    check(api.function(&built->layerBelow, loaded, "layer_below"), "cuModuleGetFunction");
                    check(api.function(&built->layerUpdate, loaded, "layer_update"), "cuModuleGetFunction");
                    check(api.function(&built->backStep, loaded, "back_step"), "cuModuleGetFunction");
                    check(api.function(&built->backUpdate, loaded, "back_update"), "cuModuleGetFunction");
                check(api.function(&built->backFlags, loaded, "back_flags"), "cuModuleGetFunction");
                check(api.function(&built->reduceCopy, loaded, "reduce_copy"), "cuModuleGetFunction");
                } catch (const utils::exception::IException& e) {
                    (void)api.destroy(&program);
                    throw;
                }
                (void)api.destroy(&program);
            } catch (const utils::exception::IException& e) {built = nullptr;}
            it = cache.emplace(context, std::move(built)).first;
            return it->second.get();
        }

        // The context that owns a buffer of the device
        _cold CUcontext contextOf(const Api& api, const utils::system::GpuBuffer buffer)
        {
            CUcontext context = nullptr;

            check(api.pointerAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, static_cast<CUdeviceptr>(buffer)), "cuPointerGetAttribute");
            return context;
        }

        // Launch a kernel over count threads and wait for it
        _hot void launch(const Api& api, const CUfunction function, const brain::UIType count, void** arguments)
        {
            const unsigned int blocks = static_cast<unsigned int>((count + BLOCK - 1) / BLOCK);

            check(api.launch(function, blocks, 1, 1, BLOCK, 1, 1, 0, nullptr, arguments, nullptr), "cuLaunchKernel");
        }

        } // namespace

        _cold bool brain::tools::gpu::available(const utils::system::IGpu* handler)
        {
            return handler != nullptr && handler->isAvailable() && handler->getMode() == utils::system::MemoryMode::Cuda && api() != nullptr;
        }

        _hot bool brain::tools::gpu::layers(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Values& inputs, std::vector<brain::Values>& sums, std::vector<brain::Values>& outputs)
        {
            // nothing to compute on the gpu
            if (!brain::tools::gpu::available(handler) || neurons == 0 || layers.empty()) return false;
            const Api& driver = *api();

            try {
                const Module* functions = compiled(driver, contextOf(driver, neurons));
                if (functions == nullptr) return false;
                const Bind bind(driver, contextOf(driver, neurons));
                brain::UIType total = 0;

                for (const brain::UIType layer: layers) total += layer;
                const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done

                // every layer reads the outputs of the previous one
                Buffer input(driver, inputs.size() * sizeof(brain::FType));
                Buffer sum(driver, total * sizeof(brain::FType));
                Buffer output(driver, total * sizeof(brain::FType));
                CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
                CUdeviceptr weightsPtr = static_cast<CUdeviceptr>(weights);
                brain::UIType first = 0;
                brain::UIType previous = 0;
                int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu

                check(driver.store(input.get(), inputs.data(), inputs.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
                for (brain::UIType l = 0; l < layers.size(); ++l) {
                    CUdeviceptr previousPtr = l == 0 ? input.get() : output.at(previous * sizeof(brain::FType));
                    CUdeviceptr sumPtr = sum.get();
                    CUdeviceptr outputPtr = output.get();
                    brain::UIType size = layers[l];
                    void* arguments[] = {&neuronsPtr, &weightsPtr, &previousPtr, &sumPtr, &outputPtr, &first, &size, &activationMode};

                    launch(driver, functions->layer, size, arguments);
                    previous = first;
                    first += size;
                }
                check(driver.synchronize(), "cuCtxSynchronize");

                // fetch every sum and output, split by layer
                brain::Values allSums(total);
                brain::Values allOutputs(total);

                check(driver.fetch(allSums.data(), sum.get(), total * sizeof(brain::FType)), "cuMemcpyDtoH");
                check(driver.fetch(allOutputs.data(), output.get(), total * sizeof(brain::FType)), "cuMemcpyDtoH");
                sums.clear();
                outputs.clear();
                brain::UIType offset = 0;
                for (const brain::UIType layer: layers) {
                    sums.emplace_back(allSums.begin() + static_cast<std::ptrdiff_t>(offset), allSums.begin() + static_cast<std::ptrdiff_t>(offset + layer));
                    outputs.emplace_back(allOutputs.begin() + static_cast<std::ptrdiff_t>(offset), allOutputs.begin() + static_cast<std::ptrdiff_t>(offset + layer));
                    offset += layer;
                }
            } catch (const utils::exception::IException& e) {return false;} // computed on the host instead
            return true;
        }

        _hot bool brain::tools::gpu::steps(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const bool raw, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::UIType steps, const brain::ActivationFunction activation, const brain::FType threshold, const brain::Values& inputs, std::vector<brain::Values>& states, const bool keep)
        {
            // nothing to compute on the gpu
            if (!brain::tools::gpu::available(handler) || neurons == 0 || neuron == 0 || steps == 0 || inputs.size() < input) return false;
            if (cap > 0 && links == 0) return false;
            const Api& driver = *api();

            try {
                const Module* functions = compiled(driver, contextOf(driver, neurons));
                if (functions == nullptr) return false;
                const Bind bind(driver, contextOf(driver, neurons));
                const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done

                // layout of the structs of the memories
                brain::UIType neuronStride = sizeof(brain::BrainNeuron);
                brain::UIType biasOffset = 0;
                brain::UIType linkStride = sizeof(brain::BrainLink);
                brain::UIType idOffset = offsetof(brain::BrainLink, id);
                brain::UIType weightOffset = offsetof(brain::BrainLink, weight);
                long long inwardOffset = -1;

                if (raw) {
                    const brain::RawBrainNeuron sample{};

                    neuronStride = sizeof(brain::RawBrainNeuron);
                    biasOffset = static_cast<brain::UIType>(reinterpret_cast<const char*>(&sample.bias) - reinterpret_cast<const char*>(&sample));
                    linkStride = sizeof(brain::RawBrainLink);
                    idOffset = offsetof(brain::RawBrainLink, id);
                    weightOffset = offsetof(brain::RawBrainLink, weight);
                    inwardOffset = static_cast<long long>(offsetof(brain::RawBrainLink, inward));
                }

                // the states: every step kept, or two buffers used in turn
                const brain::UIType count = keep ? steps + 1 : 2;
                Buffer buffer(driver, count * neuron * sizeof(brain::FType));
                brain::Values first(neuron, 0.0);
                CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
                CUdeviceptr linksPtr = static_cast<CUdeviceptr>(links);
                brain::UIType neuronCount = neuron;
                brain::UIType inputCount = input;
                brain::UIType capacity = cap;
                int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu
                brain::FType limit = threshold;

                std::copy_n(inputs.begin(), input, first.begin());
                check(driver.store(buffer.get(), first.data(), neuron * sizeof(brain::FType)), "cuMemcpyHtoD");
                brain::UIType last = 0;
                for (brain::UIType step = 0; step < steps; ++step) {
                    const brain::UIType from = keep ? step : step % 2;
                    const brain::UIType to = keep ? step + 1 : (step + 1) % 2;
                    CUdeviceptr fromPtr = buffer.at(from * neuron * sizeof(brain::FType));
                    CUdeviceptr toPtr = buffer.at(to * neuron * sizeof(brain::FType));
                    void* arguments[] = {&neuronsPtr, &neuronStride, &biasOffset, &linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &fromPtr, &toPtr, &neuronCount, &inputCount, &activationMode, &limit};

                    launch(driver, functions->step, neuron, arguments);
                    last = to;
                }
                check(driver.synchronize(), "cuCtxSynchronize");

                // fetch the states
                states.clear();
                if (keep) {
                    brain::Values all(count * neuron);

                    check(driver.fetch(all.data(), buffer.get(), all.size() * sizeof(brain::FType)), "cuMemcpyDtoH");
                    for (brain::UIType s = 0; s < count; ++s)
                        states.emplace_back(all.begin() + static_cast<std::ptrdiff_t>(s * neuron), all.begin() + static_cast<std::ptrdiff_t>((s + 1) * neuron));
                } else {
                    brain::Values state(neuron);

                    check(driver.fetch(state.data(), buffer.at(last * neuron * sizeof(brain::FType)), state.size() * sizeof(brain::FType)), "cuMemcpyDtoH");
                    states.push_back(std::move(state));
                }
            } catch (const utils::exception::IException& e) {return false;} // computed on the host instead
            return true;
        }

        _hot bool brain::tools::gpu::layersBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Values& inputs, const std::vector<brain::Values>& sums, const std::vector<brain::Values>& outputs, const brain::Values& errors)
        {
            // nothing to compute on the gpu
            if (!brain::tools::gpu::available(handler) || neurons == 0 || layers.empty() || sums.size() != layers.size() || outputs.size() != layers.size() || errors.size() != layers.back()) return false;
            const Api& driver = *api();

            try {
                const Module* functions = compiled(driver, contextOf(driver, neurons));
                if (functions == nullptr) return false;
                const Bind bind(driver, contextOf(driver, neurons));
                brain::UIType total = 0;
                brain::UIType widest = 0;

                for (const brain::UIType layer: layers) {
                    total += layer;
                    widest = std::max(widest, layer);
                }
                const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done

                // what the last forward pass kept goes to the device, the weights and the biases are changed there
                brain::Values allSums;
                brain::Values allOutputs;
                for (brain::UIType l = 0; l < layers.size(); ++l) {
                    allSums.insert(allSums.end(), sums[l].begin(), sums[l].end());
                    allOutputs.insert(allOutputs.end(), outputs[l].begin(), outputs[l].end());
                }
                Buffer input(driver, inputs.size() * sizeof(brain::FType));
                Buffer sum(driver, total * sizeof(brain::FType));
                Buffer output(driver, total * sizeof(brain::FType));
                Buffer slope(driver, widest * sizeof(brain::FType));
                Buffer delta(driver, widest * sizeof(brain::FType));
                Buffer below(driver, widest * sizeof(brain::FType));
                CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
                CUdeviceptr weightsPtr = static_cast<CUdeviceptr>(weights);
                CUdeviceptr deltaPtr = delta.get();
                CUdeviceptr belowPtr = below.get();
                CUdeviceptr slopePtr = slope.get();
                brain::UIType first = total;
                int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu

                check(driver.store(input.get(), inputs.data(), inputs.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
                check(driver.store(sum.get(), allSums.data(), total * sizeof(brain::FType)), "cuMemcpyHtoD");
                check(driver.store(output.get(), allOutputs.data(), total * sizeof(brain::FType)), "cuMemcpyHtoD");
                check(driver.store(delta.get(), errors.data(), errors.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
                for (brain::UIType l = layers.size(); l-- > 0;) {
                    brain::UIType size = layers[l];
                    brain::UIType belowSize = l == 0 ? 0 : layers[l - 1];
                    CUdeviceptr sumsPtr;
                    CUdeviceptr outputsPtr;
                    CUdeviceptr previousPtr;

                    first -= size;
                    sumsPtr = sum.at(first * sizeof(brain::FType));
                    outputsPtr = output.at(first * sizeof(brain::FType));
                    previousPtr = l == 0 ? input.get() : output.at((first - belowSize) * sizeof(brain::FType));
                    void* slopeArguments[] = {&deltaPtr, &sumsPtr, &outputsPtr, &slopePtr, &size, &activationMode};
                    void* belowArguments[] = {&neuronsPtr, &weightsPtr, &slopePtr, &belowPtr, &first, &size, &belowSize};
                    void* updateArguments[] = {&neuronsPtr, &weightsPtr, &slopePtr, &previousPtr, &first, &size};

                    launch(driver, functions->layerSlope, size, slopeArguments);
                    if (l > 0) launch(driver, functions->layerBelow, belowSize, belowArguments); // with the weights before the update
                    launch(driver, functions->layerUpdate, size, updateArguments);
                    std::swap(deltaPtr, belowPtr);
                }
                check(driver.synchronize(), "cuCtxSynchronize");
            } catch (const utils::exception::IException& e) {return false;} // computed on the host instead
            return true;
        }

        _hot bool brain::tools::gpu::stepsBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::ActivationFunction activation, const std::vector<brain::Values>& states, const brain::Values& rewards)
        {
            // nothing to compute on the gpu
            if (!brain::tools::gpu::available(handler) || neurons == 0 || neuron == 0 || cap == 0 || links == 0 || states.size() < 2 || rewards.size() > neuron) return false;
            const Api& driver = *api();

            try {
                const Module* functions = compiled(driver, contextOf(driver, neurons));
                if (functions == nullptr) return false;
                const Bind bind(driver, contextOf(driver, neurons));
                const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done

                // layout of the structs of the raw memories
                const brain::RawBrainNeuron sample{};
                brain::UIType neuronStride = sizeof(brain::RawBrainNeuron);
                brain::UIType biasOffset = static_cast<brain::UIType>(reinterpret_cast<const char*>(&sample.bias) - reinterpret_cast<const char*>(&sample));
                brain::UIType linkStride = sizeof(brain::RawBrainLink);
                brain::UIType idOffset = offsetof(brain::RawBrainLink, id);
                brain::UIType weightOffset = offsetof(brain::RawBrainLink, weight);
                brain::UIType inwardOffset = offsetof(brain::RawBrainLink, inward);
                const brain::UIType steps = states.size() - 1;

                // the states of the forward pass and the scores of the outputs go to the device, the gains are made there
                brain::Values allStates;
                brain::Values start(neuron, 0.0);
                for (const brain::Values& state: states) allStates.insert(allStates.end(), state.begin(), state.end());
                std::copy(rewards.begin(), rewards.end(), start.end() - static_cast<std::ptrdiff_t>(rewards.size()));
                Buffer stateBuffer(driver, allStates.size() * sizeof(brain::FType));
                Buffer gradA(driver, neuron * sizeof(brain::FType));
                Buffer gradB(driver, neuron * sizeof(brain::FType));
                Buffer biasGrad(driver, neuron * sizeof(brain::FType));
                Buffer weightGrad(driver, neuron * cap * sizeof(brain::FType));
                CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
                CUdeviceptr linksPtr = static_cast<CUdeviceptr>(links);
                CUdeviceptr gradPtr = gradA.get();
                CUdeviceptr belowPtr = gradB.get();
                CUdeviceptr biasPtr = biasGrad.get();
                CUdeviceptr weightPtr = weightGrad.get();
                brain::UIType neuronCount = neuron;
                brain::UIType inputCount = input;
                brain::UIType capacity = cap;
                int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu

                check(driver.store(stateBuffer.get(), allStates.data(), allStates.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
                check(driver.store(gradPtr, start.data(), neuron * sizeof(brain::FType)), "cuMemcpyHtoD");
                check(driver.zero(biasPtr, 0, neuron * sizeof(brain::FType)), "cuMemsetD8");
                check(driver.zero(weightPtr, 0, neuron * cap * sizeof(brain::FType)), "cuMemsetD8");
                for (brain::UIType step = steps; step > 0; --step) {
                    CUdeviceptr beforePtr = stateBuffer.at((step - 1) * neuron * sizeof(brain::FType));
                    void* arguments[] = {&neuronsPtr, &neuronStride, &biasOffset, &linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &beforePtr, &gradPtr, &belowPtr, &biasPtr, &weightPtr, &neuronCount, &inputCount, &activationMode};

                    check(driver.zero(belowPtr, 0, neuron * sizeof(brain::FType)), "cuMemsetD8");
                    launch(driver, functions->backStep, neuron, arguments);
                    std::swap(gradPtr, belowPtr);
                }
                void* updateArguments[] = {&neuronsPtr, &neuronStride, &biasOffset, &linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &biasPtr, &weightPtr, &neuronCount, &inputCount};

                launch(driver, functions->backUpdate, neuron, updateArguments);
                check(driver.synchronize(), "cuCtxSynchronize");
            } catch (const utils::exception::IException& e) {return false;} // computed on the host instead
            return true;
        }

    _hot bool brain::tools::gpu::triggered(const utils::system::IGpu* handler, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::FType low, const brain::FType high, bool& any)
    {
        // nothing to compute on the gpu
        if (!brain::tools::gpu::available(handler) || links == 0 || neuron == 0 || cap == 0) return false;
        const Api& driver = *api();

        try {
            const Module* functions = compiled(driver, contextOf(driver, links));
            if (functions == nullptr) return false;
            const Bind bind(driver, contextOf(driver, links));
            brain::UIType linkStride = sizeof(brain::RawBrainLink);
            brain::UIType idOffset = offsetof(brain::RawBrainLink, id);
            brain::UIType weightOffset = offsetof(brain::RawBrainLink, weight);
            brain::UIType inwardOffset = offsetof(brain::RawBrainLink, inward);
            Buffer flagBuffer(driver, neuron);
            CUdeviceptr linksPtr = static_cast<CUdeviceptr>(links);
            CUdeviceptr flagsPtr = flagBuffer.get();
            brain::UIType capacity = cap;
            brain::UIType neuronCount = neuron;
            brain::FType lowest = low;
            brain::FType highest = high;
            std::vector<unsigned char> flags(neuron);
            void* arguments[] = {&linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &lowest, &highest, &flagsPtr, &neuronCount};

            const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done
            launch(driver, functions->backFlags, neuron, arguments);
            check(driver.synchronize(), "cuCtxSynchronize");
            check(driver.fetch(flags.data(), flagBuffer.get(), neuron), "cuMemcpyDtoH");
            any = std::any_of(flags.begin(), flags.end(), [](const unsigned char flag) {return flag != 0;});
        } catch (const utils::exception::IException& e) {return false;} // computed on the host instead
        return true;
    }

    _hot bool brain::tools::gpu::reduce(const utils::system::IGpu* handler, const utils::system::GpuBuffer rawNeurons, const utils::system::GpuBuffer rawLinks, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap)
    {
        // nothing to compute on the gpu
        if (!brain::tools::gpu::available(handler) || rawNeurons == 0 || rawLinks == 0 || neurons == 0 || links == 0 || neuron == 0 || cap == 0) return false;
        const Api& driver = *api();

        try {
            const Module* functions = compiled(driver, contextOf(driver, rawNeurons));
            if (functions == nullptr) return false;
            const Bind bind(driver, contextOf(driver, rawNeurons));
            const brain::RawBrainNeuron sample{};
            brain::UIType neuronStride = sizeof(brain::RawBrainNeuron);
            brain::UIType biasOffset = static_cast<brain::UIType>(reinterpret_cast<const char*>(&sample.bias) - reinterpret_cast<const char*>(&sample));
            brain::UIType linkStride = sizeof(brain::RawBrainLink);
            brain::UIType idOffset = offsetof(brain::RawBrainLink, id);
            brain::UIType weightOffset = offsetof(brain::RawBrainLink, weight);
            brain::UIType inwardOffset = offsetof(brain::RawBrainLink, inward);
            brain::UIType reducedNeuronStride = sizeof(brain::BrainNeuron);
            brain::UIType reducedBiasOffset = offsetof(brain::BrainNeuron, bias);
            brain::UIType reducedLinkStride = sizeof(brain::BrainLink);
            brain::UIType reducedIdOffset = offsetof(brain::BrainLink, id);
            brain::UIType reducedWeightOffset = offsetof(brain::BrainLink, weight);
            CUdeviceptr rawNeuronsPtr = static_cast<CUdeviceptr>(rawNeurons);
            CUdeviceptr rawLinksPtr = static_cast<CUdeviceptr>(rawLinks);
            CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
            CUdeviceptr linksPtr = static_cast<CUdeviceptr>(links);
            brain::UIType capacity = cap;
            brain::UIType neuronCount = neuron;
            void* arguments[] = {&rawNeuronsPtr, &neuronStride, &biasOffset, &rawLinksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &neuronsPtr, &reducedNeuronStride, &reducedBiasOffset, &linksPtr, &reducedLinkStride, &reducedIdOffset, &reducedWeightOffset, &neuronCount};

            const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done
            launch(driver, functions->reduceCopy, neuron, arguments);
            check(driver.synchronize(), "cuCtxSynchronize");
        } catch (const utils::exception::IException& e) {return false;} // computed on the host instead
        return true;
    }

#else

        // No CUDA headers: nothing is computed on the gpu, the networks compute on the host
        _cold bool brain::tools::gpu::available(const utils::system::IGpu* handler)
        {
            (void)handler;
            return false;
        }

        _hot bool brain::tools::gpu::layers(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Values& inputs, std::vector<brain::Values>& sums, std::vector<brain::Values>& outputs)
        {
            (void)handler; (void)neurons; (void)weights; (void)layers; (void)activation; (void)inputs; (void)sums; (void)outputs;
            return false;
        }

        _hot bool brain::tools::gpu::steps(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const bool raw, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::UIType steps, const brain::ActivationFunction activation, const brain::FType threshold, const brain::Values& inputs, std::vector<brain::Values>& states, const bool keep)
        {
            (void)handler; (void)neurons; (void)links; (void)raw; (void)neuron; (void)cap; (void)input; (void)steps; (void)activation; (void)threshold; (void)inputs; (void)states; (void)keep;
            return false;
        }

        _hot bool brain::tools::gpu::layersBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Values& inputs, const std::vector<brain::Values>& sums, const std::vector<brain::Values>& outputs, const brain::Values& errors)
        {
            (void)handler; (void)neurons; (void)weights; (void)layers; (void)activation; (void)inputs; (void)sums; (void)outputs; (void)errors;
            return false;
        }

        _hot bool brain::tools::gpu::stepsBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::ActivationFunction activation, const std::vector<brain::Values>& states, const brain::Values& rewards)
        {
            (void)handler; (void)neurons; (void)links; (void)neuron; (void)cap; (void)input; (void)activation; (void)states; (void)rewards;
            return false;
        }

    _hot bool brain::tools::gpu::triggered(const utils::system::IGpu* handler, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::FType low, const brain::FType high, bool& any)
    {
    (void)handler; (void)links; (void)neuron; (void)cap; (void)low; (void)high; (void)any;
    return false;
    }

    _hot bool brain::tools::gpu::reduce(const utils::system::IGpu* handler, const utils::system::GpuBuffer rawNeurons, const utils::system::GpuBuffer rawLinks, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap)
    {
    (void)handler; (void)rawNeurons; (void)rawLinks; (void)neurons; (void)links; (void)neuron; (void)cap;
    return false;
    }

#endif /* BRAIN_CUDA */
