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
#include <cstdlib>
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
        // All the time steps in one launch of ONE block (a small network, one thread per neuron: a cap under 32): the same values as the kernel of one step, a barrier of the block between two steps (buffer: every step kept, or two states used in turn)
        extern "C" __global__ void steps_all(const char* neurons, const ul nstride, const ul boff, const char* links, const ul lstride, const ul idoff, const ul woff, const long long ioff, const ul cap, ft* buffer, const ul n, const ul input, const int mode, const ft threshold, const ul steps, const int keep)
        {
            for (ul step = 0; step < steps; ++step) {
                const ft* state = buffer + (keep ? step : step % 2) * n;
                ft* next = buffer + (keep ? step + 1 : (step + 1) % 2) * n;

                for (ul i = threadIdx.x; i < n; i += blockDim.x) {
                    if (i < input) {next[i] = state[i]; continue;}
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
                __syncthreads(); // the whole step is written before the next one reads it
            }
        }
        // One warp per neuron (a large cap): the lanes read consecutive slots (grouped reads), then the sum is reduced in the warp
        extern "C" __global__ void step_warp(const char* neurons, const ul nstride, const ul boff, const char* links, const ul lstride, const ul idoff, const ul woff, const long long ioff, const ul cap, const ft* state, ft* next, const ul n, const ul input, const int mode, const ft threshold)
        {
            const ul i = ((ul)blockIdx.x * blockDim.x + threadIdx.x) / 32;
            const unsigned int lane = threadIdx.x % 32;
            if (i >= n) return; // the same for the whole warp
            if (i < input) {if (lane == 0) next[i] = state[i]; return;}
            ft sum = 0;

            for (ul slot = lane; slot < cap; slot += 32) {
                const char* link = links + (i * cap + slot) * lstride;
                if (ioff >= 0 && !*(const unsigned char*)(link + ioff)) continue;
                sum += *(const ft*)(link + woff) * state[*(const ul*)(link + idoff)];
            }
            for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane != 0) return;
            ft value = activate(sum + *(const ft*)(neurons + i * nstride + boff), mode);
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

    // The same as back_step with one warp per neuron (a large cap): the sum reduced in the warp, every lane its slots
    extern "C" __global__ void back_step_warp(const char* neurons, const ul nstride, const ul boff, const char* links, const ul lstride, const ul idoff, const ul woff, const ul ioff, const ul cap, const ft* before, const ft* grad, ft* below, ft* biasGrad, ft* weightGrad, const ul n, const ul input, const int mode)
    {
        const ul i = ((ul)blockIdx.x * blockDim.x + threadIdx.x) / 32;
        const unsigned int lane = threadIdx.x % 32;
        if (i >= n || i < input || grad[i] == 0) return; // the same for the whole warp
        ft sum = 0;

        for (ul slot = lane; slot < cap; slot += 32) {
            const char* link = links + (i * cap + slot) * lstride;
            if (*(const unsigned char*)(link + ioff)) sum += *(const ft*)(link + woff) * before[*(const ul*)(link + idoff)];
        }
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        sum += *(const ft*)(neurons + i * nstride + boff);
        const ft slope = grad[i] * derive(sum, activate(sum, mode), mode);
        if (lane == 0) biasGrad[i] += slope;
        for (ul slot = lane; slot < cap; slot += 32) {
            const char* link = links + (i * cap + slot) * lstride;
            if (!*(const unsigned char*)(link + ioff)) continue;
            const ul id = *(const ul*)(link + idoff);
            weightGrad[i * cap + slot] += slope * before[id];
            atomicAdd(&below[id], slope * *(const ft*)(link + woff));
        }
    }

    // clones[slot of an inward link] = the slot of its clone (found once on the host per change of the links)
    extern "C" __global__ void back_update(char* neurons, const ul nstride, const ul boff, char* links, const ul lstride, const ul idoff, const ul woff, const ul ioff, const ul cap, const ft* biasGrad, const ft* weightGrad, const ul* clones, const ul n, const ul input)
    {
        const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n || i < input) return;
        if (isfinite(biasGrad[i])) *(ft*)(neurons + i * nstride + boff) += biasGrad[i];
        for (ul slot = 0; slot < cap; ++slot) {
            char* link = links + (i * cap + slot) * lstride;
            if (!*(const unsigned char*)(link + ioff) || !isfinite(weightGrad[i * cap + slot])) continue;
            ft weight = *(ft*)(link + woff) + weightGrad[i * cap + slot];

            if (weight == 0) weight = (ft)FT_MIN; // a weight at 0 would look like a free slot
            *(ft*)(link + woff) = weight;
            *(ft*)(links + clones[i * cap + slot] * lstride + woff) = weight;
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

    // ---- Layered networks, a whole mini-batch at once (matrices: the samples are the rows) ----
    #define TILE 64
    #define DEPTH 16
    // C[m][n] = sum over p of A(m, p) * B(p, n), A(m, p) = A[m * sam + p * sap], B(p, n) = B[p * sbp + n * sbn]: tiles of 64 x 64 in the shared memory, 256 threads of 4 x 4 results each
    // epilogue 0: forward (the bias of the neuron is added, the sum and the output of the activation are written), 1: the slope of the layer below (times the derivative of its activation), 2: added to C (the gradient)
    extern "C" __global__ void gemm(const ft* A, const ul sam, const ul sap, const ft* B, const ul sbp, const ul sbn, ft* C, const ul ldc, const ul M, const ul N, const ul K, const int epilogue, const LayerNeuron* neurons, const ul first, ft* sums, const ft* sumsBelow, const ft* outputsBelow, const int mode)
    {
        __shared__ ft tileA[DEPTH][TILE + 1];
        __shared__ ft tileB[DEPTH][TILE + 1];
        const int thread = threadIdx.x;
        const int tx = thread % 16;
        const int ty = thread / 16;
        const ul row0 = (ul)blockIdx.y * TILE;
        const ul col0 = (ul)blockIdx.x * TILE;
        ft acc[4][4];

        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) acc[i][j] = 0;
        for (ul p0 = 0; p0 < K; p0 += DEPTH) {
            for (int e = 0; e < 4; ++e) {
                const int idx = thread + e * 256;
                int a1 = sap == 1 ? idx / DEPTH : idx % TILE; // m of the element of A
                int a2 = sap == 1 ? idx % DEPTH : idx / TILE; // p of the element of A
                const ul gm = row0 + a1;
                const ul gp = p0 + a2;
                tileA[a2][a1] = (gm < M && gp < K) ? A[gm * sam + gp * sap] : (ft)0;
                int b1 = sbp == 1 ? idx / DEPTH : idx % TILE; // n of the element of B
                int b2 = sbp == 1 ? idx % DEPTH : idx / TILE; // p of the element of B
                const ul gn = col0 + b1;
                const ul gq = p0 + b2;
                tileB[b2][b1] = (gn < N && gq < K) ? B[gq * sbp + gn * sbn] : (ft)0;
            }
            __syncthreads();
            #pragma unroll
            for (int p = 0; p < DEPTH; ++p) {
                ft a[4];
                ft b[4];

                for (int i = 0; i < 4; ++i) a[i] = tileA[p][ty + i * 16];
                for (int j = 0; j < 4; ++j) b[j] = tileB[p][tx + j * 16];
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
            }
            __syncthreads();
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const ul m = row0 + ty + i * 16;
                const ul n = col0 + tx + j * 16;
                if (m >= M || n >= N) continue;
                const ul at = m * ldc + n;

                if (epilogue == 0) {
                    const ft sum = acc[i][j] + neurons[first + n].bias;
                    if (sums != 0) sums[at] = sum;
                    C[at] = activate(sum, mode);
                } else if (epilogue == 1) {
                    C[at] = acc[i][j] * derive(sumsBelow[at], outputsBelow[at], mode);
                } else {
                    C[at] += acc[i][j];
                }
            }
        }
    }

    // The slope of the last layer of every sample: error * derivative
    extern "C" __global__ void slope_all(const ft* errors, const ft* sums, const ft* outputs, ft* slope, const ul count, const int mode)
    {
        const ul k = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= count) return;
        slope[k] = errors[k] * derive(sums[k], outputs[k], mode);
    }

    // The gradient of the biases of a layer: the sum of the slopes of the samples
    extern "C" __global__ void bias_grad(const ft* slope, ft* gradient, const ul size, const ul samples)
    {
        const ul k = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= size) return;
        ft sum = 0;

        for (ul b = 0; b < samples; ++b) sum += slope[b * size + k];
        gradient[k] += sum;
    }

    // The weights go down the mean of the gradients (the gradient goes back to 0)
    extern "C" __global__ void apply_weights(ft* weights, ft* gradient, const ul count, const ft inverse)
    {
        const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        weights[i] -= gradient[i] * inverse;
        gradient[i] = 0;
    }

    extern "C" __global__ void apply_bias(LayerNeuron* neurons, ft* gradient, const ul count, const ft inverse)
    {
        const ul i = (ul)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        neurons[i].bias -= gradient[i] * inverse;
        gradient[i] = 0;
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
        CUfunction stepWarp = nullptr;
        CUfunction stepsAll = nullptr; // every time step in one launch of one block (a small network)
        CUfunction backStepWarp = nullptr;
    CUfunction backFlags = nullptr;
    CUfunction reduceCopy = nullptr;
        CUfunction gemm = nullptr; // a mini-batch of a layered network: the products of the matrices
        CUfunction slopeAll = nullptr;
        CUfunction biasGrad = nullptr;
        CUfunction applyWeights = nullptr;
        CUfunction applyBias = nullptr;
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
            (void)std::atexit([]() {loaded.reset();}); // the libraries are closed at the exit, before the statics of libutils they use are destroyed
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

    // Device buffers kept by the calls of one thread for one context (grown when too small): no allocation and no release per call
    class Scratch {
        private:
            const Api& _api;
            const CUcontext _context;
            std::vector<std::pair<CUdeviceptr, std::size_t>> _slots; // <buffer, bytes> of each slot

        public:
            // The buffer of a slot with at least bytes (a buffer of another context, freed by its owner, is dropped)
            _hot CUdeviceptr get(const std::size_t slot, const std::size_t bytes)
            {
                std::pair<CUdeviceptr, std::size_t>& entry = (slot < this->_slots.size()) ? this->_slots[slot] : (this->_slots.resize(slot + 1), this->_slots[slot]);
                CUcontext owner = nullptr;

                if (entry.first != 0 && (this->_api.pointerAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, entry.first) != CUDA_SUCCESS || owner != this->_context)) entry = {0, 0};
                if (entry.first != 0 && entry.second >= bytes) return entry.first;
                if (entry.first != 0) (void)this->_api.release(entry.first);
                entry = {0, 0};
                check(this->_api.allocate(&entry.first, std::max<std::size_t>(bytes, 1)), "cuMemAlloc");
                entry.second = std::max<std::size_t>(bytes, 1);
                return entry.first;
            };

            Scratch(const Api& api, const CUcontext context): _api{api}, _context{context} { /* Nothing */ };
            ~Scratch() noexcept
            {
                CUcontext previous = nullptr;

                if (this->_api.push(this->_context) != CUDA_SUCCESS) return; // the context is gone with its memory
                for (const std::pair<CUdeviceptr, std::size_t>& entry: this->_slots)
                    if (entry.first != 0) (void)this->_api.release(entry.first);
                (void)this->_api.pop(&previous);
            };
            Scratch(const Scratch& other) = delete;
            Scratch& operator=(const Scratch& other) = delete;
    };

    // The scratch buffers of the calling thread for a context (a thread never shares them: the execution stays thread safe)
    _hot Scratch& scratch(const Api& api, const CUcontext context)
    {
        thread_local std::map<CUcontext, std::unique_ptr<Scratch>> pools;
        std::unique_ptr<Scratch>& pool = pools[context];

        if (!pool) pool = std::make_unique<Scratch>(api, context);
        return *pool;
    }

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
                check(api.function(&built->stepWarp, loaded, "step_warp"), "cuModuleGetFunction");
                check(api.function(&built->stepsAll, loaded, "steps_all"), "cuModuleGetFunction");
                check(api.function(&built->backStepWarp, loaded, "back_step_warp"), "cuModuleGetFunction");
            check(api.function(&built->backFlags, loaded, "back_flags"), "cuModuleGetFunction");
            check(api.function(&built->reduceCopy, loaded, "reduce_copy"), "cuModuleGetFunction");
                check(api.function(&built->gemm, loaded, "gemm"), "cuModuleGetFunction");
                check(api.function(&built->slopeAll, loaded, "slope_all"), "cuModuleGetFunction");
                check(api.function(&built->biasGrad, loaded, "bias_grad"), "cuModuleGetFunction");
                check(api.function(&built->applyWeights, loaded, "apply_weights"), "cuModuleGetFunction");
                check(api.function(&built->applyBias, loaded, "apply_bias"), "cuModuleGetFunction");
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
            const CUcontext context = contextOf(driver, neurons);
            const Bind bind(driver, context);
            Scratch& pool = scratch(driver, context);
            brain::UIType total = 0;

            for (const brain::UIType layer: layers) total += layer;
            const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done

            // every layer reads the outputs of the previous one
            const CUdeviceptr input = pool.get(0, inputs.size() * sizeof(brain::FType));
            const CUdeviceptr sum = pool.get(1, total * sizeof(brain::FType));
            const CUdeviceptr output = pool.get(2, total * sizeof(brain::FType));
            CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
            CUdeviceptr weightsPtr = static_cast<CUdeviceptr>(weights);
            brain::UIType first = 0;
            brain::UIType previous = 0;
            int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu

            check(driver.store(input, inputs.data(), inputs.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
            for (brain::UIType l = 0; l < layers.size(); ++l) {
                CUdeviceptr previousPtr = l == 0 ? input : output + previous * sizeof(brain::FType);
                CUdeviceptr sumPtr = sum;
                CUdeviceptr outputPtr = output;
                brain::UIType size = layers[l];
                void* arguments[] = {&neuronsPtr, &weightsPtr, &previousPtr, &sumPtr, &outputPtr, &first, &size, &activationMode};

                launch(driver, functions->layer, size, arguments);
                previous = first;
                first += size;
            }

            // fetch every sum and output, split by layer (a copy to the host waits for the kernels)
            brain::Values allSums(total);
            brain::Values allOutputs(total);

            check(driver.fetch(allSums.data(), sum, total * sizeof(brain::FType)), "cuMemcpyDtoH");
            check(driver.fetch(allOutputs.data(), output, total * sizeof(brain::FType)), "cuMemcpyDtoH");
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
            const CUcontext context = contextOf(driver, neurons);
            const Bind bind(driver, context);
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
            const CUdeviceptr buffer = scratch(driver, context).get(0, count * neuron * sizeof(brain::FType));
            brain::Values first(neuron, 0.0);
            CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
            CUdeviceptr linksPtr = static_cast<CUdeviceptr>(links);
            brain::UIType neuronCount = neuron;
            brain::UIType inputCount = input;
            brain::UIType capacity = cap;
            int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu
            brain::FType limit = threshold;

            std::copy_n(inputs.begin(), input, first.begin());
            check(driver.store(buffer, first.data(), neuron * sizeof(brain::FType)), "cuMemcpyHtoD");
            brain::UIType last = 0;
            const bool warp = cap >= 32; // a warp per neuron: grouped reads of its slots (a small cap keeps one thread per neuron)

            // a small network with one thread per neuron: every step in one launch of one block with a barrier between two steps (no launch per step; with a warp per neuron, a bigger network or more blocks it was slower than the launches)
            const bool single = !warp && neuron <= BRAIN_GPU_SINGLE_KERNEL_NEURONS && functions->stepsAll != nullptr;
            if (single) {
                CUdeviceptr bufferPtr = buffer;
                brain::UIType stepCount = steps;
                int keepFlag = keep ? 1 : 0;
                void* arguments[] = {&neuronsPtr, &neuronStride, &biasOffset, &linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &bufferPtr, &neuronCount, &inputCount, &activationMode, &limit, &stepCount, &keepFlag};

                check(driver.launch(functions->stepsAll, 1, 1, 1, BLOCK, 1, 1, 0, nullptr, arguments, nullptr), "cuLaunchKernel");
                last = keep ? steps : steps % 2;
            }
            for (brain::UIType step = 0; !single && step < steps; ++step) {
                const brain::UIType from = keep ? step : step % 2;
                const brain::UIType to = keep ? step + 1 : (step + 1) % 2;
                CUdeviceptr fromPtr = buffer + from * neuron * sizeof(brain::FType);
                CUdeviceptr toPtr = buffer + to * neuron * sizeof(brain::FType);
                void* arguments[] = {&neuronsPtr, &neuronStride, &biasOffset, &linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &fromPtr, &toPtr, &neuronCount, &inputCount, &activationMode, &limit};

                launch(driver, warp ? functions->stepWarp : functions->step, warp ? neuron * 32 : neuron, arguments);
                last = to;
            }

            // fetch the states (a copy to the host waits for the kernels)
            states.clear();
            if (keep) {
                brain::Values all(count * neuron);

                check(driver.fetch(all.data(), buffer, all.size() * sizeof(brain::FType)), "cuMemcpyDtoH");
                for (brain::UIType s = 0; s < count; ++s)
                    states.emplace_back(all.begin() + static_cast<std::ptrdiff_t>(s * neuron), all.begin() + static_cast<std::ptrdiff_t>((s + 1) * neuron));
            } else {
                brain::Values state(neuron);

                check(driver.fetch(state.data(), buffer + last * neuron * sizeof(brain::FType), state.size() * sizeof(brain::FType)), "cuMemcpyDtoH");
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
            const CUcontext context = contextOf(driver, neurons);
            const Bind bind(driver, context);
            Scratch& pool = scratch(driver, context);
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
            const CUdeviceptr input = pool.get(0, inputs.size() * sizeof(brain::FType));
            const CUdeviceptr sum = pool.get(1, total * sizeof(brain::FType));
            const CUdeviceptr output = pool.get(2, total * sizeof(brain::FType));
            CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
            CUdeviceptr weightsPtr = static_cast<CUdeviceptr>(weights);
            CUdeviceptr deltaPtr = pool.get(3, widest * sizeof(brain::FType));
            CUdeviceptr belowPtr = pool.get(4, widest * sizeof(brain::FType));
            CUdeviceptr slopePtr = pool.get(5, widest * sizeof(brain::FType));
            brain::UIType first = total;
            int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu

            check(driver.store(input, inputs.data(), inputs.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
            check(driver.store(sum, allSums.data(), total * sizeof(brain::FType)), "cuMemcpyHtoD");
            check(driver.store(output, allOutputs.data(), total * sizeof(brain::FType)), "cuMemcpyHtoD");
            check(driver.store(deltaPtr, errors.data(), errors.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
            for (brain::UIType l = layers.size(); l-- > 0;) {
                brain::UIType size = layers[l];
                brain::UIType belowSize = l == 0 ? 0 : layers[l - 1];
                CUdeviceptr sumsPtr;
                CUdeviceptr outputsPtr;
                CUdeviceptr previousPtr;

                first -= size;
                sumsPtr = sum + first * sizeof(brain::FType);
                outputsPtr = output + first * sizeof(brain::FType);
                previousPtr = l == 0 ? input : output + (first - belowSize) * sizeof(brain::FType);
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

    namespace {

    // Where the weights of a layer start in the weights, its inputs and neurons, and its first neuron
    struct Shape {
        brain::UIType weights = 0;
        brain::UIType in = 0;
        brain::UIType out = 0;
        brain::UIType first = 0;
    };

    // The geometry of a layered network (the weights follow each other, layer after layer)
    struct Plan {
        std::vector<Shape> shapes;
        brain::UIType input = 0; // inputs of the network
        brain::UIType total = 0; // neurons
        brain::UIType widest = 0; // neurons of the biggest layer
        brain::UIType weights = 0; // weights
    };

    _cold Plan planOf(const brain::Layers& layers, const brain::UIType input)
    {
        Plan plan;
        brain::UIType previous = input;

        plan.input = input;
        for (const brain::UIType layer: layers) {
            plan.shapes.push_back({plan.weights, previous, layer, plan.total});
            plan.weights += layer * previous;
            plan.total += layer;
            plan.widest = std::max(plan.widest, layer);
            previous = layer;
        }
        return plan;
    }

    // One product of matrices (see the kernel gemm): C[m][n] = sum over p of A(m, p) * B(p, n)
    struct Product {
        CUdeviceptr a = 0;
        brain::UIType sam = 0;
        brain::UIType sap = 0;
        CUdeviceptr b = 0;
        brain::UIType sbp = 0;
        brain::UIType sbn = 0;
        CUdeviceptr c = 0;
        brain::UIType ldc = 0;
        brain::UIType m = 0;
        brain::UIType n = 0;
        brain::UIType k = 0;
        int epilogue = 0; // 0: forward, 1: the slope of the layer below, 2: added to C
        CUdeviceptr neurons = 0;
        brain::UIType first = 0;
        CUdeviceptr sums = 0;
        CUdeviceptr sumsBelow = 0;
        CUdeviceptr outputsBelow = 0;
        int mode = 0;
    };

    _hot void multiply(const Api& api, const Module& functions, Product p)
    {
        void* arguments[] = {&p.a, &p.sam, &p.sap, &p.b, &p.sbp, &p.sbn, &p.c, &p.ldc, &p.m, &p.n, &p.k, &p.epilogue, &p.neurons, &p.first, &p.sums, &p.sumsBelow, &p.outputsBelow, &p.mode};
        const unsigned int columns = static_cast<unsigned int>((p.n + 63) / 64);
        const unsigned int rows = static_cast<unsigned int>((p.m + 63) / 64);

        check(api.launch(functions.gemm, columns, rows, 1, 256, 1, 1, 0, nullptr, arguments, nullptr), "cuLaunchKernel");
    }

    // The forward pass of count samples (the matrix input, count rows): every layer is a product of matrices
    // keep: the sums and the outputs of every layer stay in sums and outputs (count rows each, layer after layer), else two buffers are used in turn; returns the outputs of the last layer
    _hot CUdeviceptr forwardPass(const Api& api, const Module& functions, const Plan& plan, const CUdeviceptr neurons, const CUdeviceptr weights, const int mode, const CUdeviceptr input, const brain::UIType count, const CUdeviceptr sums, const CUdeviceptr outputs, const CUdeviceptr turnA, const CUdeviceptr turnB)
    {
        const bool keep = sums != 0;
        CUdeviceptr previous = input;
        CUdeviceptr last = input;

        for (std::size_t l = 0; l < plan.shapes.size(); ++l) {
            const Shape& shape = plan.shapes[l];
            Product product;
            const CUdeviceptr target = keep ? outputs + count * shape.first * sizeof(brain::FType) : (l % 2 == 0 ? turnA : turnB);

            product.a = previous;
            product.sam = shape.in;
            product.sap = 1;
            product.b = weights + shape.weights * sizeof(brain::FType);
            product.sbp = 1;
            product.sbn = shape.in;
            product.c = target;
            product.ldc = shape.out;
            product.m = count;
            product.n = shape.out;
            product.k = shape.in;
            product.neurons = neurons;
            product.first = shape.first;
            product.sums = keep ? sums + count * shape.first * sizeof(brain::FType) : 0;
            product.mode = mode;
            multiply(api, functions, product);
            previous = target;
            last = target;
        }
        return last;
    }

    } // namespace

    // The buffers of the device kept for the mini-batches of a network
    struct brain::tools::gpu::Batch::State {
        enum Slot {Input, Sums, Outputs, SlopeA, SlopeB, Errors, GradWeights, GradBias, SlotCount};

        const Api* driver = nullptr;
        CUcontext context = nullptr;
        std::pair<CUdeviceptr, std::size_t> buffers[SlotCount] = {}; // <buffer, bytes>
        brain::UIType count = 0; // samples of the last forward pass kept in the buffers
        Plan plan; // the geometry of the network they were made for

        // The buffer of a slot with at least bytes (an older one is released, not copied; zero: filled with 0)
        _hot CUdeviceptr get(const Slot slot, const std::size_t bytes, const bool zero = false)
        {
            std::pair<CUdeviceptr, std::size_t>& entry = this->buffers[slot];

            if (entry.first != 0 && entry.second >= bytes) return entry.first;
            if (entry.first != 0) (void)this->driver->release(entry.first);
            entry = {0, 0};
            check(this->driver->allocate(&entry.first, std::max<std::size_t>(bytes, 1)), "cuMemAlloc");
            entry.second = std::max<std::size_t>(bytes, 1);
            if (zero) check(this->driver->zero(entry.first, 0, entry.second), "cuMemsetD8");
            return entry.first;
        };

        State(const Api& api, const CUcontext owner): driver{&api}, context{owner} { /* Nothing */ };
        ~State() noexcept
        {
            CUcontext previous = nullptr;

            if (this->driver == nullptr || this->driver->push(this->context) != CUDA_SUCCESS) return; // the context is gone with its memory
            for (const std::pair<CUdeviceptr, std::size_t>& entry: this->buffers)
                if (entry.first != 0) (void)this->driver->release(entry.first);
            (void)this->driver->pop(&previous);
        };
        State(const State& other) = delete;
        State& operator=(const State& other) = delete;
    };

    brain::tools::gpu::Batch::Batch() = default;
    brain::tools::gpu::Batch::~Batch() = default;

    _cold std::size_t brain::tools::gpu::Batch::bytes(void) const
    {
        std::size_t total = 0;

        if (this->state) for (const std::pair<CUdeviceptr, std::size_t>& entry: this->state->buffers) total += entry.first != 0 ? entry.second : 0;
        return total;
    }

    _hot bool brain::tools::gpu::layersAll(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Dataset& inputs, brain::Dataset& outputs, const std::size_t budget)
    {
        // nothing to compute on the gpu
        if (!brain::tools::gpu::available(handler) || neurons == 0 || weights == 0 || layers.empty() || inputs.empty() || inputs.front().empty()) return false;
        const Api& driver = *api();
        const Plan plan = planOf(layers, inputs.front().size());

        for (const brain::Values& sample: inputs)
            if (sample.size() != plan.input) return false;
        try {
            const CUcontext context = contextOf(driver, neurons);
            const Module* functions = compiled(driver, context);
            if (functions == nullptr) return false;
            const Bind bind(driver, context);
            Scratch& pool = scratch(driver, context);
            const brain::UIType perSample = (plan.input + 2 * plan.widest + layers.back()) * sizeof(brain::FType);
            const brain::UIType group = std::clamp<brain::UIType>(budget / perSample, 1, std::min<brain::UIType>(inputs.size(), 65536));
            brain::Values flat;
            brain::Values result;

            const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done
            outputs.clear();
            outputs.reserve(inputs.size());
            for (brain::UIType begin = 0; begin < inputs.size(); begin += group) {
                const brain::UIType count = std::min(group, inputs.size() - begin);
                const CUdeviceptr input = pool.get(0, count * plan.input * sizeof(brain::FType));
                const CUdeviceptr turnA = pool.get(1, count * plan.widest * sizeof(brain::FType));
                const CUdeviceptr turnB = pool.get(2, count * plan.widest * sizeof(brain::FType));

                flat.resize(count * plan.input);
                for (brain::UIType i = 0; i < count; ++i) std::copy(inputs[begin + i].begin(), inputs[begin + i].end(), flat.begin() + static_cast<std::ptrdiff_t>(i * plan.input));
                check(driver.store(input, flat.data(), flat.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
                const CUdeviceptr last = forwardPass(driver, *functions, plan, static_cast<CUdeviceptr>(neurons), static_cast<CUdeviceptr>(weights), static_cast<int>(activation), input, count, 0, 0, turnA, turnB);

                result.resize(count * layers.back());
                check(driver.fetch(result.data(), last, result.size() * sizeof(brain::FType)), "cuMemcpyDtoH"); // waits for the kernels
                for (brain::UIType i = 0; i < count; ++i)
                    outputs.emplace_back(result.begin() + static_cast<std::ptrdiff_t>(i * layers.back()), result.begin() + static_cast<std::ptrdiff_t>((i + 1) * layers.back()));
            }
        } catch (const utils::exception::IException& e) {outputs.clear(); return false;} // computed on the host instead
        return true;
    }

    _hot bool brain::tools::gpu::layersTrain(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, brain::tools::gpu::Batch& batch, const brain::Dataset& inputs, brain::Dataset& outputs, const std::size_t budget)
    {
        // nothing to compute on the gpu
        if (!brain::tools::gpu::available(handler) || neurons == 0 || weights == 0 || layers.empty() || inputs.empty() || inputs.front().empty()) return false;
        const Api& driver = *api();
        const Plan plan = planOf(layers, inputs.front().size());
        const brain::UIType count = inputs.size();

        for (const brain::Values& sample: inputs)
            if (sample.size() != plan.input) return false;
        if (count * (plan.input + 2 * plan.total + 2 * plan.widest + layers.back()) * sizeof(brain::FType) + (plan.weights + plan.total) * sizeof(brain::FType) > budget) return false;
        try {
            const CUcontext context = contextOf(driver, neurons);
            const Module* functions = compiled(driver, context);
            if (functions == nullptr) return false;
            const Bind bind(driver, context);

            if (batch.state && (batch.state->context != context || batch.state->plan.weights != plan.weights || batch.state->plan.total != plan.total)) batch.state.reset(); // another network: its buffers are not these ones
            if (!batch.state) {
                batch.state = std::make_unique<brain::tools::gpu::Batch::State>(driver, context);
                batch.state->plan = plan;
            }
            brain::tools::gpu::Batch::State& state = *batch.state;
            using Slot = brain::tools::gpu::Batch::State;
            brain::Values flat(count * plan.input);
            brain::Values result(count * layers.back());

            const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done
            state.get(Slot::GradWeights, plan.weights * sizeof(brain::FType), true);
            state.get(Slot::GradBias, plan.total * sizeof(brain::FType), true);
            const CUdeviceptr input = state.get(Slot::Input, flat.size() * sizeof(brain::FType));
            const CUdeviceptr sums = state.get(Slot::Sums, count * plan.total * sizeof(brain::FType));
            const CUdeviceptr kept = state.get(Slot::Outputs, count * plan.total * sizeof(brain::FType));
            state.get(Slot::SlopeA, count * plan.widest * sizeof(brain::FType));
            state.get(Slot::SlopeB, count * plan.widest * sizeof(brain::FType));
            state.get(Slot::Errors, result.size() * sizeof(brain::FType));
            state.count = 0;

            for (brain::UIType i = 0; i < count; ++i) std::copy(inputs[i].begin(), inputs[i].end(), flat.begin() + static_cast<std::ptrdiff_t>(i * plan.input));
            check(driver.store(input, flat.data(), flat.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
            const CUdeviceptr last = forwardPass(driver, *functions, plan, static_cast<CUdeviceptr>(neurons), static_cast<CUdeviceptr>(weights), static_cast<int>(activation), input, count, sums, kept, 0, 0);

            check(driver.fetch(result.data(), last, result.size() * sizeof(brain::FType)), "cuMemcpyDtoH"); // waits for the kernels
            outputs.clear();
            outputs.reserve(count);
            for (brain::UIType i = 0; i < count; ++i)
                outputs.emplace_back(result.begin() + static_cast<std::ptrdiff_t>(i * layers.back()), result.begin() + static_cast<std::ptrdiff_t>((i + 1) * layers.back()));
            state.count = count;
        } catch (const utils::exception::IException& e) {outputs.clear(); return false;} // computed on the host instead
        return true;
    }

    _hot bool brain::tools::gpu::layersGradient(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, brain::tools::gpu::Batch& batch, const brain::Dataset& errors)
    {
        // the errors of the samples of the last layersTrain()
        if (!brain::tools::gpu::available(handler) || neurons == 0 || weights == 0 || layers.empty() || !batch.state || batch.state->count == 0 || errors.size() != batch.state->count) return false;
        const Api& driver = *api();

        for (const brain::Values& sample: errors)
            if (sample.size() != layers.back()) return false;
        try {
            brain::tools::gpu::Batch::State& state = *batch.state;
            using Slot = brain::tools::gpu::Batch::State;
            const Module* functions = compiled(driver, state.context);
            if (functions == nullptr) return false;
            const Bind bind(driver, state.context);
            const Plan& plan = state.plan;
            const brain::UIType count = state.count;
            const int mode = static_cast<int>(activation);
            brain::Values flat(count * layers.back());

            for (brain::UIType i = 0; i < count; ++i) std::copy(errors[i].begin(), errors[i].end(), flat.begin() + static_cast<std::ptrdiff_t>(i * layers.back()));
            const CUdeviceptr errorsPtr = state.buffers[Slot::Errors].first;
            const CUdeviceptr sums = state.buffers[Slot::Sums].first;
            const CUdeviceptr outputs = state.buffers[Slot::Outputs].first;
            const CUdeviceptr gradWeights = state.buffers[Slot::GradWeights].first;
            const CUdeviceptr gradBias = state.buffers[Slot::GradBias].first;
            CUdeviceptr slope = state.buffers[Slot::SlopeA].first;
            CUdeviceptr below = state.buffers[Slot::SlopeB].first;
            const CUdeviceptr input = state.buffers[Slot::Input].first;
            const CUdeviceptr weightsPtr = static_cast<CUdeviceptr>(weights);

            check(driver.store(errorsPtr, flat.data(), flat.size() * sizeof(brain::FType)), "cuMemcpyHtoD");

            // the slope of the last layer, then layer after layer: the gradient of the weights and of the biases, the slope of the layer below
            {
                const Shape& shape = plan.shapes.back();
                CUdeviceptr errorsArg = errorsPtr;
                CUdeviceptr sumsArg = sums + count * shape.first * sizeof(brain::FType);
                CUdeviceptr outputsArg = outputs + count * shape.first * sizeof(brain::FType);
                brain::UIType size = count * shape.out;
                int modeArg = mode;
                void* arguments[] = {&errorsArg, &sumsArg, &outputsArg, &slope, &size, &modeArg};

                launch(driver, functions->slopeAll, size, arguments);
            }
            for (std::size_t l = plan.shapes.size(); l-- > 0;) {
                const Shape& shape = plan.shapes[l];
                Product product;

                // weights: gradient[k][j] += sum over the samples of slope[b][k] * previous[b][j]
                product.a = slope;
                product.sam = 1;
                product.sap = shape.out;
                product.b = l == 0 ? input : outputs + count * plan.shapes[l - 1].first * sizeof(brain::FType);
                product.sbp = shape.in;
                product.sbn = 1;
                product.c = gradWeights + shape.weights * sizeof(brain::FType);
                product.ldc = shape.in;
                product.m = shape.out;
                product.n = shape.in;
                product.k = count;
                product.epilogue = 2;
                multiply(driver, *functions, product);

                // biases
                CUdeviceptr slopeArg = slope;
                CUdeviceptr gradientArg = gradBias + shape.first * sizeof(brain::FType);
                brain::UIType size = shape.out;
                brain::UIType samples = count;
                void* arguments[] = {&slopeArg, &gradientArg, &size, &samples};

                launch(driver, functions->biasGrad, size, arguments);

                // the slope of the layer below: slope * weights, times the derivative of the activation of that layer (with the weights before they change)
                if (l == 0) continue;
                Product down;

                down.a = slope;
                down.sam = shape.out;
                down.sap = 1;
                down.b = weightsPtr + shape.weights * sizeof(brain::FType);
                down.sbp = shape.in;
                down.sbn = 1;
                down.c = below;
                down.ldc = shape.in;
                down.m = count;
                down.n = shape.in;
                down.k = shape.out;
                down.epilogue = 1;
                down.sumsBelow = sums + count * plan.shapes[l - 1].first * sizeof(brain::FType);
                down.outputsBelow = outputs + count * plan.shapes[l - 1].first * sizeof(brain::FType);
                down.mode = mode;
                multiply(driver, *functions, down);
                std::swap(slope, below);
            }
        } catch (const utils::exception::IException& e) {return false;}
        return true;
    }

    _hot bool brain::tools::gpu::layersApply(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, brain::tools::gpu::Batch& batch, const brain::UIType weightCount, const brain::UIType neuronCount, const brain::FType divisor)
    {
        // nothing kept: nothing to apply
        if (!batch.state) return true;
        if (!brain::tools::gpu::available(handler) || neurons == 0 || weights == 0 || !(divisor > 0)) return false;
        const Api& driver = *api();

        try {
            brain::tools::gpu::Batch::State& state = *batch.state;
            using Slot = brain::tools::gpu::Batch::State;
            if (state.buffers[Slot::GradWeights].second < weightCount * sizeof(brain::FType) || state.buffers[Slot::GradBias].second < neuronCount * sizeof(brain::FType)) return false;
            const Module* functions = compiled(driver, state.context);
            if (functions == nullptr) return false;
            const Bind bind(driver, state.context);
            CUdeviceptr weightsPtr = static_cast<CUdeviceptr>(weights);
            CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
            CUdeviceptr gradWeights = state.buffers[Slot::GradWeights].first;
            CUdeviceptr gradBias = state.buffers[Slot::GradBias].first;
            brain::UIType weightSize = weightCount;
            brain::UIType neuronSize = neuronCount;
            brain::FType inverse = static_cast<brain::FType>(1.0) / divisor;
            void* weightArguments[] = {&weightsPtr, &gradWeights, &weightSize, &inverse};
            void* biasArguments[] = {&neuronsPtr, &gradBias, &neuronSize, &inverse};

            const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done
            launch(driver, functions->applyWeights, weightCount, weightArguments);
            launch(driver, functions->applyBias, neuronCount, biasArguments);
        } catch (const utils::exception::IException& e) {return false;}
        return true;
    }

    _hot void brain::tools::gpu::layersClear(brain::tools::gpu::Batch& batch)
    {
        if (!batch.state) return;
        try {
            brain::tools::gpu::Batch::State& state = *batch.state;
            using Slot = brain::tools::gpu::Batch::State;
            const Bind bind(*state.driver, state.context);

            for (const Slot::Slot slot: {Slot::GradWeights, Slot::GradBias}) {
                const std::pair<CUdeviceptr, std::size_t>& entry = state.buffers[slot];

                if (entry.first != 0) check(state.driver->zero(entry.first, 0, entry.second), "cuMemsetD8");
            }
        } catch (const utils::exception::IException& e) {batch.state.reset();} // the gradient is lost with the buffers
    }

    _hot bool brain::tools::gpu::stepsBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::ActivationFunction activation, const std::vector<brain::Values>& states, const brain::Values& rewards, const utils::system::GpuBuffer clones)
    {
        // nothing to compute on the gpu
        if (!brain::tools::gpu::available(handler) || neurons == 0 || neuron == 0 || cap == 0 || links == 0 || clones == 0 || states.size() < 2 || rewards.size() > neuron) return false;
        const Api& driver = *api();

        try {
            const Module* functions = compiled(driver, contextOf(driver, neurons));
            if (functions == nullptr) return false;
            const CUcontext context = contextOf(driver, neurons);
            const Bind bind(driver, context);
            Scratch& pool = scratch(driver, context);
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
            const CUdeviceptr stateBuffer = pool.get(0, allStates.size() * sizeof(brain::FType));
            CUdeviceptr neuronsPtr = static_cast<CUdeviceptr>(neurons);
            CUdeviceptr linksPtr = static_cast<CUdeviceptr>(links);
            CUdeviceptr gradPtr = pool.get(1, neuron * sizeof(brain::FType));
            CUdeviceptr belowPtr = pool.get(2, neuron * sizeof(brain::FType));
            CUdeviceptr biasPtr = pool.get(3, neuron * sizeof(brain::FType));
            CUdeviceptr weightPtr = pool.get(4, neuron * cap * sizeof(brain::FType));
            brain::UIType neuronCount = neuron;
            brain::UIType inputCount = input;
            brain::UIType capacity = cap;
            int activationMode = static_cast<int>(activation); // none | sigmoid | linear | relu

            check(driver.store(stateBuffer, allStates.data(), allStates.size() * sizeof(brain::FType)), "cuMemcpyHtoD");
            check(driver.store(gradPtr, start.data(), neuron * sizeof(brain::FType)), "cuMemcpyHtoD");
            check(driver.zero(biasPtr, 0, neuron * sizeof(brain::FType)), "cuMemsetD8");
            check(driver.zero(weightPtr, 0, neuron * cap * sizeof(brain::FType)), "cuMemsetD8");
            for (brain::UIType step = steps; step > 0; --step) {
                CUdeviceptr beforePtr = stateBuffer + (step - 1) * neuron * sizeof(brain::FType);
                void* arguments[] = {&neuronsPtr, &neuronStride, &biasOffset, &linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &beforePtr, &gradPtr, &belowPtr, &biasPtr, &weightPtr, &neuronCount, &inputCount, &activationMode};

                check(driver.zero(belowPtr, 0, neuron * sizeof(brain::FType)), "cuMemsetD8");
                launch(driver, cap >= 32 ? functions->backStepWarp : functions->backStep, cap >= 32 ? neuron * 32 : neuron, arguments);
                std::swap(gradPtr, belowPtr);
            }
            CUdeviceptr clonesPtr = static_cast<CUdeviceptr>(clones);
            void* updateArguments[] = {&neuronsPtr, &neuronStride, &biasOffset, &linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &biasPtr, &weightPtr, &clonesPtr, &neuronCount, &inputCount};

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
            const CUcontext context = contextOf(driver, links);
            const Bind bind(driver, context);
            brain::UIType linkStride = sizeof(brain::RawBrainLink);
            brain::UIType idOffset = offsetof(brain::RawBrainLink, id);
            brain::UIType weightOffset = offsetof(brain::RawBrainLink, weight);
            brain::UIType inwardOffset = offsetof(brain::RawBrainLink, inward);
            CUdeviceptr linksPtr = static_cast<CUdeviceptr>(links);
            CUdeviceptr flagsPtr = scratch(driver, context).get(0, neuron);
            brain::UIType capacity = cap;
            brain::UIType neuronCount = neuron;
            brain::FType lowest = low;
            brain::FType highest = high;
            std::vector<unsigned char> flags(neuron);
            void* arguments[] = {&linksPtr, &linkStride, &idOffset, &weightOffset, &inwardOffset, &capacity, &lowest, &highest, &flagsPtr, &neuronCount};

            const_cast<utils::system::IGpu*>(handler)->join(); // the transfers of the memories are done
            launch(driver, functions->backFlags, neuron, arguments);
            check(driver.fetch(flags.data(), flagsPtr, neuron), "cuMemcpyDtoH"); // waits for the kernel
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

    _hot bool brain::tools::gpu::layersAll(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, const brain::Dataset& inputs, brain::Dataset& outputs, const std::size_t budget)
    {
        (void)handler; (void)neurons; (void)weights; (void)layers; (void)activation; (void)inputs; (void)outputs; (void)budget;
        return false;
    }

    _hot bool brain::tools::gpu::layersTrain(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, brain::tools::gpu::Batch& batch, const brain::Dataset& inputs, brain::Dataset& outputs, const std::size_t budget)
    {
        (void)handler; (void)neurons; (void)weights; (void)layers; (void)activation; (void)batch; (void)inputs; (void)outputs; (void)budget;
        return false;
    }

    _hot bool brain::tools::gpu::layersGradient(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, const brain::Layers& layers, const brain::ActivationFunction activation, brain::tools::gpu::Batch& batch, const brain::Dataset& errors)
    {
        (void)handler; (void)neurons; (void)weights; (void)layers; (void)activation; (void)batch; (void)errors;
        return false;
    }

    _hot bool brain::tools::gpu::layersApply(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer weights, brain::tools::gpu::Batch& batch, const brain::UIType weightCount, const brain::UIType neuronCount, const brain::FType divisor)
    {
        (void)handler; (void)neurons; (void)weights; (void)batch; (void)weightCount; (void)neuronCount; (void)divisor;
        return false;
    }

    _hot void brain::tools::gpu::layersClear(brain::tools::gpu::Batch& batch)
    {
        (void)batch;
        return;
    }

    struct brain::tools::gpu::Batch::State { /* Nothing */ };
    brain::tools::gpu::Batch::Batch() = default;
    brain::tools::gpu::Batch::~Batch() = default;
    std::size_t brain::tools::gpu::Batch::bytes(void) const {return 0;}

    _hot bool brain::tools::gpu::stepsBack(const utils::system::IGpu* handler, const utils::system::GpuBuffer neurons, const utils::system::GpuBuffer links, const brain::UIType neuron, const brain::UIType cap, const brain::UIType input, const brain::ActivationFunction activation, const std::vector<brain::Values>& states, const brain::Values& rewards, const utils::system::GpuBuffer clones)
    {
        (void)handler; (void)neurons; (void)links; (void)neuron; (void)cap; (void)input; (void)activation; (void)states; (void)rewards; (void)clones;
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
