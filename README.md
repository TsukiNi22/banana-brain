# banana-brain

> [!TIP]
> Technical documentation, project graph and explanations of the [layered](https://tsukini22.github.io/banana-brain/docs/layer-network.html) and [brain](https://tsukini22.github.io/banana-brain/docs/brain-network.html) networks: [tsukini22.github.io/banana-brain/docs](https://tsukini22.github.io/banana-brain/docs/) (v1.0.0).

C++20 neural network library (`brain::`): one `brain::NeuralNetwork` handling several kinds of networks (layered, brain), their configuration, training and usage, plus the dataset, randomness and serialization tools.
Built as a static library on top of [libutils](https://github.com/TsukiNi22/libutils) and distributed as RPM/DEB/Arch (pacman) packages (optimized, debug and asan variants, plus the `mingw64-libbrain*` packages for Windows), or installed from the sources with CMake.

> [!NOTE]
> The library is complete in its first release (`v1.0.0`; `v1.1.0-pre` adds the mini-batches on the gpu): the layered network and the brain network (execution, backpropagation, links that change with the weights, the rules as flags, sequences), the one training entry `NeuralNetwork::train` (epochs, mini-batch, shuffle, validation and stop rules), the execution on several samples and threads, the GPU (CUDA) for the execution and the plain gradient, the serialization (`save` / `load`) and the dataset tools.

### Table of Contents
 - [Dependencies](#dependencies)
 - [Packages](#packages)
 - [Quick Setup 1 (All)](#quick-setup---1-all)
 - [Quick Setup 2 (Limited)](#quick-setup---2-limited)
 - [Windows (cross compilation)](#windows-cross-compilation)
 - [Usage](#usage)
 - [Unit tests](#unit-tests)
 - [Workflows/Release](#workflowsrelease)
 - [License](#license)

## Dependencies

> [!CAUTION]
> The CC BY-NC-SA 4.0 license of this project ([LICENSE.md](LICENSE.md)) only applies to the code of `banana-brain` itself.
> It does **not** apply to the dependencies, included or linked: each one keeps its own license.

| Name + Link | License | Status | Last Update |
| ----------- | ------- | ------ | ----------- |
| [libutils](https://github.com/TsukiNi22/libutils) | MIT | ![Dispatch (CI/CD)](https://github.com/TsukiNi22/libutils/actions/workflows/dispatch.yml/badge.svg) | ![](https://img.shields.io/github/last-commit/TsukiNi22/libutils) |

> [!NOTE]
> The content of the packages is chosen at the top of the `CMakeLists.txt` (layout of [cpp_project_template](https://github.com/TsukiNi22/cpp_project_template)): `BRAIN_KIND` (`library`) sets the defaults of `BRAIN_PACKAGE_BIN` (executable, `OFF`), `BRAIN_PACKAGE_LIB` (library, `BRAIN_SHARED_LIB` for a `.so`) and `BRAIN_PACKAGE_DEV` (headers + CMake config), each one can be forced with `-D...=ON/OFF`.
> After a change, update `PACKAGE_BIN` / `PACKAGE_LIB` / `PACKAGE_DEV` in `.github/workflows/build.yml` and `build-packages.yml`.

| Name | Version | Fedora (`dnf`) | Debian/Ubuntu (`apt`) | Arch (`pacman`) |
| ---- | ------- | -------------- | --------------------- | --------------- |
| `clang++` | C++20 | `clang` | `clang` | `clang` |
| `cmake` | >= 3.20 | `cmake` | `cmake` | `cmake` |
| `python3` (exception header generation) | >= 3.10 | `python3` | `python3` | `python` |
| libutils | >= 4.3.0, stable or `-pre` (the first with `HybridMemory`; Arch: the stable package only) | `libutils` (+ `libutils-db` / `libutils-as` for the `Debug` / `Asan` modes) | `libutils` (+ `libutils-db` / `libutils-as`) | `libutils` (+ `libutils-db` / `libutils-as`) |
| GoogleTest (tests only) | — | `gtest-devel` | `libgtest-dev` | `gtest` |
| MinGW-w64 (Windows only) | GCC >= 13 | `mingw64-gcc-c++ mingw64-libutils` | `g++-mingw-w64-x86-64` + libutils built for MinGW | — |
| Wine (Windows tests only) | — | `wine` | `wine64` | — |
| `rpm-build`, `dpkg`, `bsdtar` (packages only, `bsdtar` for the Arch ones) | — | `rpm-build dpkg bsdtar` | `rpm dpkg-dev libarchive-tools` | `rpm-tools dpkg libarchive` |

## Packages

> [!NOTE]
> Most packages have a pre-release/unstable version named `<package_name>-pre`.
> The `-pre` packages are marked as obsolete by any release/stable package (without `-pre`) of the same version or higher.

| File Name | Content |
| --------- | ------- |
| `libbrain` | Install the default packages (`libbrain-dev`, `libbrain-op`) |
| `libbrain-dev` | All the includes (exception codes included) and the global CMake configuration (requires `libutils-dev`) |
| `libbrain-dev-op` | CMake configuration for `libbrain-op` |
| `libbrain-dev-db` | CMake configuration for `libbrain-db` |
| `libbrain-dev-as` | CMake configuration for `libbrain-as` |
| `libbrain-op` | Library compiled with the optimization options (requires `libutils-op`) |
| `libbrain-db` | Library compiled with the debug options (requires `libutils-db`, intended for debugging only) |
| `libbrain-as` | Library compiled with the asan options (requires `libutils-as`, intended for debugging only) |

**Windows** (MinGW-w64 cross compilation, installed in the MinGW sysroot: `/usr/x86_64-w64-mingw32/sys-root/mingw` on Fedora, `/usr/x86_64-w64-mingw32` on Debian/Ubuntu)

| File Name | Content |
| --------- | ------- |
| `mingw64-libbrain` | Install the default Windows packages (`mingw64-libbrain-dev`, `mingw64-libbrain-op`) |
| `mingw64-libbrain-dev` | All the includes and the global CMake configuration (requires `mingw64-libutils-dev`) |
| `mingw64-libbrain-dev-op` | CMake configuration for `mingw64-libbrain-op` |
| `mingw64-libbrain-dev-db` | CMake configuration for `mingw64-libbrain-db` |
| `mingw64-libbrain-op` | Windows library compiled with the optimization options |
| `mingw64-libbrain-db` | Windows library compiled with the debug options (no asan version with MinGW) |

## Quick Setup - 1 (all)
> Setup into `/usr/local`

### Clone the repository
```bash
git clone https://github.com/TsukiNi22/banana-brain.git
cd banana-brain
```

### Install the lib
```bash
export BUILD_DIR=build
cmake -S . -B $BUILD_DIR
sudo cmake --build $BUILD_DIR --target install --parallel $(nproc)          # Optimized only
sudo cmake --build $BUILD_DIR --target install_release --parallel $(nproc)  # Debug, Asan & Optimized
```

## Quick Setup - 2 (Limited)
> Setup into `/usr`

> [!WARNING]
> If during the installation using tools like `dnf`, `apt`... you see an `invalid hash` error, reset the cache of the tool or relaunch the `setup.sh` script.
>
> If the error still persists, it might be due to an error in the repository.

> [!WARNING]
> Restriction: `fedora-based (rpm)`, `debian-based (deb)`, `arch-based (pacman)` (no `--windows` packages on Arch)

> [!NOTE]
> The usage of `sudo` in the script can be removed with the `--no-sudo` argument.

Run the setup script directly, without cloning the repository. It registers the banana-brain package repository and its GPG key for the system package manager (by default nothing is installed, `--install` also installs the package):
1. Detect the distribution from `/etc/os-release` (`rpm`, `deb` or `pacman` based).
2. `rpm`: add `/etc/yum.repos.d/brain.repo` and import the GPG key.
3. `deb`: add the GPG key in `/etc/apt/keyrings/brain.gpg`, the source in `/etc/apt/sources.list.d/brain.list`, then `apt-get update`.
4. `pacman`: trust the GPG key in the pacman keyring (`pacman-key --add` / `--lsign-key`), add the `[brain]` repository (`Server = https://tsukini22.github.io/banana-brain/arch/$arch`, `SigLevel = Required`) to `/etc/pacman.conf`, then `pacman -Sy`.
5. With `--install` only: install the package `libbrain` (`--pre` for the unstable one, `--windows` for `mingw64-libbrain`, rpm / deb only); else print the install command of the system (`sudo dnf install libbrain`, `sudo apt install libbrain` or `sudo pacman -S libbrain`).

```bash
wget -qO- https://raw.githubusercontent.com/TsukiNi22/banana-brain/main/setup.sh | bash -s
```

or with `curl`:

```bash
curl -fsSL https://raw.githubusercontent.com/TsukiNi22/banana-brain/main/setup.sh | bash -s
```

| Argument | Effect |
| -------- | ------ |
| `--no-sudo` | Run without `sudo` (the script must already run as root) |
| `--install` | Also install the package (by default only the mirror & the GPG key are setup) |
| `--pre` | Select the pre-release/unstable package (`libbrain-pre`) |
| `--windows` | Select the Windows packages (`mingw64-libbrain`, MinGW sysroot) instead of the Linux ones |
| `-h`, `--help` | Display the usage |

## Windows (cross compilation)
> Build `libbrain.a` for Windows (x86_64) from Linux with the MinGW-w64 toolchain

From the packages (after the `setup.sh` of [Quick Setup - 2](#quick-setup---2-limited)), installed in the MinGW sysroot:

```bash
curl -fsSL https://raw.githubusercontent.com/TsukiNi22/banana-brain/main/setup.sh | bash -s -- --install --windows   # mingw64-libbrain (--pre for the unstable one)
sudo dnf install mingw64-libbrain-db        # Fedora: Debug variant (requires mingw64-libutils-db)
```

From the sources:

```bash
export BUILD_DIR=build-windows
cmake -S . -B $BUILD_DIR -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/mingw-w64.cmake -DCMAKE_BUILD_TYPE=Optimized
cmake --build $BUILD_DIR --parallel $(nproc)
cmake --install $BUILD_DIR --prefix <windows_prefix>          # include/, lib/libbrain.a, lib/cmake/brain/
cmake --build $BUILD_DIR --target package_release               # mingw64-libbrain-* RPM/DEB (MinGW sysroot, no Arch packages)
```

> [!NOTE]
> No AddressSanitizer with MinGW: the `Asan` mode is built like `Debug`. The tests (`-DBUILD_TESTS=ON`) download GoogleTest and run with `wine` through `ctest`.

## Usage

```cmake
find_package(brain 1.1.0 REQUIRED)                            # installed CMake configuration (finds libutils too)
target_link_libraries(${TARGET} PRIVATE brain::brain)  # includes, utils::utils & library
```

The same CMake code works for Linux and Windows (MinGW toolchain, see [Windows](#windows-cross-compilation)). The library follows the `CMAKE_BUILD_TYPE` of the project, and needs the libutils of the same mode:

| Build type | Library | Package (Linux) | Package (Windows) |
| ---------- | ------- | --------------- | ----------------- |
| `Debug` | `libbrain_debug.a` | `libbrain-db` | `mingw64-libbrain-db` |
| `Asan` | `libbrain_asan.a` | `libbrain-as` | none (no AddressSanitizer with MinGW, use `Debug`) |
| anything else | `libbrain.a` | `libbrain-op` (in `libbrain`) | `mingw64-libbrain-op` (in `mingw64-libbrain`) |

A missing package stops the configuration with the name of the package to install.

```cpp
#include <brain/brain.hpp>    // every type, tool and network (or one header below)
```

> [!WARNING]
> Everything is defined within the namespace `brain::` (tools in `brain::tools::`).

| Include | Content |
| ------- | ------- |
| `brain/brain.hpp` | Global include of the library (+ `brain::version`), without the network classes: they are internal, created by `NeuralNetwork::init<...>` |
| `brain/NeuralNetwork.hpp` | `brain::NeuralNetwork` (`utils::type::Freezable`): `init<NetworkType>(...)` (`Layered`: `(layer, neuron_per_layer)` or `(Layers)`, `Brain`: `(neurons, size)`), `weightInit`, `biasInit`, the settings of the network (`size`, `rand`, `activationFunction`... applied to it: `init` first), `learningRate`, `lossFunction`, `train(inputs, expected, options)` (the one entry of every training: `brain::TrainOptions` is optional), manual training (`inputs`, `apply`, `backpropagation`, `loss<mean>`), `exec`, `save(path)` / `load(path)`, `clear` |
| `brain/NetworkDefine.hpp` | `brain::NetworkType` (`Layered`, `Brain`), `ActivationFunction`, `LossFunction`, `TextMode` (`Normal`, `Compact`), the defines of the files (`BRAIN_EXTENSION_*`, `BRAIN_MAGIC*`, `BRAIN_ACTIVATION_*`) and of the brain network limits (`BRAIN_LINKS_CAP_*`, `BRAIN_STEPS_CAP_*`, `BRAIN_SIGNAL_THRESHOLD`) |
| `brain/NetworkType.hpp` | `brain::IType` / `UIType` / `FType`, `Values` / `Dataset` / `Layers` / `Bytes`, the hooks `ActivationFn` / `LossFn` / `RandFn` / `LinksCapFn`, the trivially copyable neurons and links stored in a `HybridMemory`: `LayerNeuron`, `BrainNeuron` / `BrainLink` (reduced), `RawBrainNeuron` (`distance2()` / `operator\|` squared distance) / `RawBrainLink` |
| `brain/network/Network.hpp` | `brain::INetwork` -> `brain::ANetwork` -> `brain::LayerNetwork(Layers)` / `brain::BrainNetwork(neuron, cap)` (reduced, execution only) -> `brain::RawBrainNetwork(neuron, size)` (training, `reduce()`) (not in `brain.hpp`: include it only when the network classes are needed) |
| `brain/tools/random.hpp` | `brain::tools::generator` (`std::mt19937_64`, scope chosen by `BRAIN_SEED_GLOBAL` / `BRAIN_SEED_LOCAL`), `rand(seed)`, `rand_unsigned_integer()`, `rand_floating()` |
| `brain/tools/dataset.hpp` | `brain::tools::shuffle`, `rand_sets`, `text_to_input<mode>`, `input_to_text<mode, round>` |
| `brain/tools/neuron.hpp` | `brain::tools::sort(index, neurons, cap)`: the ids of the `cap` nearest neurons of a neuron, closest first (squared distances computed again, nothing stored) |
| `brain/tools/loss.hpp` / `activation.hpp` | The internal losses (`mse`, `cross_entropy`) and activations (`sigmoid`, `linear`, `relu`) of one value |
| `brain/tools/bytes.hpp` | `brain::tools::put<T>` / `get<T>` / `fits`: numbers as little endian bytes (the serialization) |

The random generator of the library (`brain::tools::generator`, used by every randomness call) has a scope chosen at compile time, with a define set **before** the include (`-DBRAIN_SEED_LOCAL` in the CMake, or `#define` on the line before):

| Define | Generator |
| ------ | --------- |
| `BRAIN_SEED_GLOBAL` (default when nothing is defined) | One single generator shared by the whole program (`extern`, defined once in the library) |
| `BRAIN_SEED_LOCAL` | One generator per translation unit including `brain/tools/random.hpp` (`static`): each file has its own independent sequence |

> [!WARNING]
> Defining both at once is a compile error (`#error`). The helpers `rand(seed)`, `rand_unsigned_integer()` and `rand_floating()` stay the same and use whichever generator is active.

> [!IMPORTANT]
> `train(inputs, expected, options = {})` is the one entry of the training and returns a `brain::TrainResult` (`epochs`, `updates`, `loss`, `reason`). Without options: one epoch, an update after each sample, as before. `brain::TrainOptions`: `epochs` (at most, 0 = no limit), `batchSize` (1 = after each sample, n = mini-batch of n with the mean of their gradients, 0 = the whole set; a layered network and a brain one both do it), `shuffle` and `seed`, `mean` (one coefficient), `targetLoss`, `patience`, `validationIn` / `validationOut` (measured after each epoch), `stop` (the built-in rule: `StopCondition::Epochs`, `TargetLoss`, `Patience`) and `onEpoch` (your rule, `bool(options, state)`: true = go on, it wins over `stop`; `TrainState` has `epoch`, `updates`, `loss`, `validationLoss`, `bestValidation`, `sinceBest`, `seconds`). `nn.train(x, y, {.epochs = 200, .batchSize = 32, .shuffle = true})`.

> [!NOTE]
> The size, random and activation settings belong to the network: they throw `NoNetwork` before `init<...>`. `exec` executes without storing anything in the network (thread-safe) and is the only call allowed once the network is frozen (`freeze()`).

### Execution
- `LayerNetwork`: layer after layer, each neuron is `activation(bias + Σ weight × output of the previous layer)`; the inputs feed the first layer and the outputs are the last layer.
- `BrainNetwork` / `RawBrainNetwork`: **time steps**. The first `input` neurons (ids `0..input-1`) hold the inputs, the last `output` neurons (ids `n-output..n-1`) are the outputs (`size(input, output)` is needed), and at each step every other neuron computes `activation(bias + Σ weight × output of the neuron it is linked to, at the previous step)`; the outputs are read after the last step. On a `RawBrainNetwork` the input neurons are placed on the left edge of the space and the output ones on the right edge, evenly spread; changing the sizes places the neurons again.
- Speed of the execution on the host (what to know to use it): `applyAll(Dataset)` (`NeuralNetwork::execAll`) executes several samples together, the links are read once per step for all of them (about 1.4 to 2.8 times faster per sample, the same outputs as `apply()` to 1e-6, a gpu network executes them one by one); the time steps stop when nothing moved any more (to the bit, or by less than `threshold()` when it is set: the signal threshold is also the tolerance of the stop, 0 by default = exact), the next steps being copies; the steps of a big network are shared between threads (`threads(count)`: 0 = automatic, a thread for each `BRAIN_THREADS_MIN_WORK` slots of links of a step, 1 = one thread; the states are the same to the bit whatever the number of threads, a function of the user or a sequence stay on one thread); the sums of the links use four partial sums and the sigmoid of a brain network `brain::tools::fastExp` (about 1 ulp), so its values differ from a network of before by about 1e-7; on a gpu a network of at most `BRAIN_GPU_SINGLE_KERNEL_NEURONS` neurons with a cap under 32 computes all its steps in one launch.
- A `RawBrainNetwork` can also take a **sequence**: `sequence(steps)` gives the inputs of each time step (step t reads `steps[t]`, the inputs are 0 after the last one), so the outputs depend on what the network kept of the earlier steps; `applySequence(steps)` executes one without storing anything, `inputs()` goes back to the same inputs at every step. A sequence runs on the host.
- The number of steps is `min(percentage × neurons, hook(neurons))`, at least 1 (`stepsCap(percentage | fn)`, `resetStepsCap()`, getter `stepsCap()`, same as `linksCap`; default: 100 % of the neurons and `log2(n)`, `BRAIN_STEPS_CAP_*`, never under `BRAIN_STEPS_PATH_MARGIN` (1.1) times the farthest path in links from an input to an output, computed at the init (the signal has to arrive); the links cap defaults to 100 % and `3 sqrt(n)`, `BRAIN_LINKS_CAP_*`: 12 links at 16 neurons, 94 at 1000). `threshold(value)` / `threshold()` cancel (set to 0) a signal whose absolute value is under it at each step, to stop signals too weak to matter (`BRAIN_SIGNAL_THRESHOLD`, a placeholder at 0 = nothing cancelled).

### Backpropagation
- The score: `internalScoreFunction(ScoreFunction)` (`Difference` by default = the signed error, `Sign`, `Loss` = its square) or `scoreFunction(fn)` (a `ScoreFn(output, expected)` that wins over the internal one, with the convention of the network: positive = good for a brain network, too high for a layered one), and `scoreRate(rate)` (> 0, 1 by default), on `NeuralNetwork` and on every network (`scores(outputs, expected)` gives the coefficients of one sample: score function, then score rate). `train` sends `score * scoreRate * learningRate`. Not saved in the files, like the learning rate.
- `backpropagation(coef | coefs)` follows an `apply()` (the sums, outputs and states of it are kept). The learning rate is already in the coefficients: `train` sends the signed error times it (`output - expected` for a layered network, `expected - output` as the score of a brain network); the loss function only measures and stops `train(loss)`.
- `LayerNetwork`: the classic one. A coefficient is the error of an output (positive = too high): the weights and biases go down the gradient, layer by layer (derivative of the activation; estimated around the sum for an external function).
- `RawBrainNetwork`: a coefficient is a **score** (positive = good): going back through the time steps, the links and biases that made the output are strengthened, the ones of a negative score weakened; a link and its clone keep the same weight.
- The links change too: a link is removed (with its clone) when `|weight| <= weightMin()` and, when `|weight| >= weightMax()`, the nearest neuron that is still free is linked (within the links cap) with a random weight between half and all of the mean of the two nearest links of the list that did not cause it (fewer, or the middle of the limits, without them), always inside the limits (`BRAIN_WEIGHT_MIN` 0.01, `BRAIN_WEIGHT_MAX` 2.0, setters `weightMin`, `weightMax`, `weightLimits`). It never removes the last link of a neuron, nor the last inward link of an output or hidden neuron, nor the last outward link of an input, unless `allowOrphans()` (`disallowOrphans()` by default). With `leftToRight(true)` a link only goes from a neuron to a neuron on its right (the list of the neurons on the left of each one is computed once, at the init or the load, and used for every new link). The training has options, all off by default and saved in the `.rbn`: `momentum(m)` (step = m * last step + gradient, `0 <= m < 1`), `adam(rate)` (Adam: each weight moves by about the rate in the direction of the mean of its gradients, whatever the size of the score; replaces the momentum, `0` = off), `stepLimit(l)` (each change of a weight or a bias kept in `[-l, l]`), `clampWeights(true)` (`|weight|` and `|bias|` never above `weightMax()`), `directOutput(true)` (the score reaches the sum of an output without the slope of its activation: a saturated output still learns), `batch(true)` (`backpropagation()` only adds the gradient, `applyBatch()` applies the mean of the pending ones), `lowNewLinks(true)` (a link added at `weightMax()` starts at 5 % of the limits instead of the size of the nearest links), `keepPaths(true)` (a link is never removed when it takes the last path of an input to an output away). At the init, `farLinks(k)` gives `k` slots of the cap to random far neurons and `pathInit(h)` (`h >= 2`) adds the missing links so that every input has a path of at most `h` links to every output. These training options run on the host (a network on the gpu reads its memories back). `revive()`, called at the end of a round, gives a bias of 0 and halves the inward weights of every neuron stuck at 0 or 1 on all the executions since the last call (a saturated neuron has no gradient), and returns how many were revived. The limits and the mode are saved in the `.rbn`. On the gpu a kernel checks whether a link is at a limit: if none is, nothing is read back and the execution copy is rebuilt on the device.
- Options of the links and of the steps of the weights on a `RawBrainNetwork`, saved in the `.rbn`: `rateDecay(decay)` divides the steps by `1 + decay * updates so far` (with `batch()` an update is a round; 0 = constant by default), `invertedLinks(bool)` swaps the limits (a link is removed at `weightMax`, a new one grows at `weightMin`; off by default), `randomLinks(bool)` gives each neuron a random number of links at the init, between `minLinks()` and the cap (**on by default**, `false` = as many as the cap allows), `minLinks(count)` never removes a link that leaves one of its two neurons with `count` links or fewer (**1 by default**: a neuron keeps its last link, so `allowOrphans()` needs `minLinks(0)`).
- The on / off rules of a `RawBrainNetwork` are flags, as the flags of the cli of libutils: `brain::Rule` (`ORPHANS`, `LEFT_TO_RIGHT`, `CLAMP_WEIGHTS`, `LOW_NEW_LINKS`, `DIRECT_OUTPUT`, `KEEP_PATHS`, `BATCH`, `RANDOM_LINKS`, `INVERTED_LINKS`, `INPUT_PATHS`, `MOVE_LAST_LINK`) with `addRules(flags)`, `subRules(flags)`, `setRules(flags)`, `resetRules()` (the defaults: `brain::rules::DEFAULT` = `RANDOM_LINKS`), `clearRules()` (every rule off, `brain::rules::NONE`) and `getRules()`; `brain::rules::LAB` is the set of the lab (`ORPHANS | LOW_NEW_LINKS | KEEP_PATHS | DIRECT_OUTPUT | BATCH`) and `brain::rules::ALL` every rule. The single setters (`allowOrphans()`, `keepPaths(bool)`...) are the same bits. Two rules are off by default: `INPUT_PATHS` (every input keeps a path to at least one output: built at the init, a removal that would cut the last one is put back; the weights stay free to go to 0) and `MOVE_LAST_LINK` (the last link of a neuron that would be removed under `weightMin` is moved to the nearest free neuron instead of being kept, its new weight follows the rule of the new links, 5 % of the limits or random).

The weights and biases are updated on the gpu when the memories are stored there (CUDA, activation not external); the links that change with the weights are then read back and rebuilt on the host.

**Mini-batches on the gpu (`LayerNetwork`).** With `batchSize` above 1, a CUDA gpu and an internal activation function, `train` computes the samples of a mini-batch together: the forward pass is one product of matrices per layer, the gradient of the weights, of the biases and the slope of each layer are made on the device and added to the gradient kept there, and `applyBatch()` applies their mean without the weights leaving the device. By hand: `applyTrain(inputs)` (the outputs of the mini-batch, what the gradient needs stays on the device) then `backpropagationAll(coefs)` (the coefficients of each output of each sample) in the mini-batch mode. `applyAll` / `execAll` also compute their samples together. What it takes on the device is the `brain::GpuProfile` of the network (`NeuralNetwork::gpuProfile`):

| Profile | Training | Execution (`applyAll`) | Device memory (network `2048,1024,256,64`, 3.95 M weights) |
| ------- | -------- | ---------------------- | ----------------------------------------------------------- |
| `Fast` (default) | the whole mini-batch on the gpu | groups up to 256 MiB (`BRAIN_GPU_FAST_EXEC_BYTES`) | the weights (15 MiB) + the gradient (15 MiB) + the sums and outputs of the batch (11 MiB at 256 samples), 1 GiB at most per mini-batch (`BRAIN_GPU_FAST_TRAIN_BYTES`, over it the host trains) + the groups of the execution |
| `Light` | the gradient is made by the host, one sample at a time | groups up to 8 MiB (`BRAIN_GPU_LIGHT_EXEC_BYTES`) | the weights only + 8 MiB for the execution |

Anything the gpu cannot do (no CUDA, an external activation function, a `.ln` file whose weights do not follow each other) is computed by the host as before, with the same results up to the float rounding of the sums.

### Save / load
`save(path)` / `load(path)` write and read the network (`NeuralNetwork`), `serialize()` / `deserialize(bytes)` do it in memory (each network).

| Extension | Network | Magic bytes |
| --------- | ------- | ----------- |
| `.ln` | `LayerNetwork` | `0x22` `'l'` |
| `.bn` | `BrainNetwork` (reduced, execution only) | `0x22` `'b'` |
| `.rbn` | `RawBrainNetwork` (training) | `0x22` `'r'` |
| `.net` | any of them (the magic bytes tell which) | the one of the network |

A brain file (`.bn`, `.rbn`) also holds the number of time steps and the signal threshold, a `.rbn` the percentage of its links and steps limits too. An other extension only warns. The third byte tells the activation (`0x00` none, `0xFF` an external function, otherwise the internal `ActivationFunction`). An external function (activation, random, links cap) can't be saved: the load warns (`CustomActivationRequired`, `CustomRandomRequired`, `CustomLinksCapRequired`, `CustomStepsCapRequired`) and it has to be given again. A bad, truncated or inconsistent file throws `InvalidNetworkFile` and the current network is kept.

```cpp
brain::NeuralNetwork network;
network.init<brain::NetworkType::Layered>(brain::Layers{8, 4, 1});   // or init<Layered>(3, 8): 3 layers of 8 neurons
network.size(2, 1);                                                  // 2 inputs, 1 output
network.internalActivationFunction(brain::ActivationFunction::Sigmoid);
network.internalLossFunction(brain::LossFunction::MSE);
network.weightInit();
network.biasInit();
network.save("network.ln");                                          // .ln, .bn, .rbn or .net
brain::Values outputs = network.exec({0.0, 1.0});
```

The errors are libutils exceptions with the codes of the library (`utils::exception::ExternalCode::NoNetwork`, `InvalidInputSize`, `InvalidOutputSize`, `InvalidCoefficient`, `InvalidLearningRate`, `InvalidNetworkFile`, `ReadOnlyNetwork`...), generated from `cmake/config/exceptions/brain.json` and installed with the headers:

```cpp
#define _Exception
#include <utils/utils.hpp>
#include <brain/brain.hpp>

try {
    brain::NeuralNetwork network;
    network.exec({0.0, 1.0});   // no init<...>() before
} catch (const utils::exception::IException& e) {
    std::cerr << e.formated() << std::endl;   // [Error] ... -> No network initialized, call init<NetworkType>() first
}
```

> [!NOTE]
> Every new `.cpp` goes in the `SRC_SHARED` list of the `CMakeLists.txt` (the library; `SRC` is the executable only, empty here): `cmake --build build --target get_unregistered_files` lists the missing ones.

## Unit tests

```bash
export BUILD_DIR=build
cmake -S . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build $BUILD_DIR --parallel $(nproc)
ctest --test-dir $BUILD_DIR --output-on-failure --timeout 30   # or ./unit_tests (written at the root)
```

> [!NOTE]
> The tests use GoogleTest (`tests/`, one file per module) and run in the `Unit Tests - Libraries (CI)` workflow. The `Gpu.*` tests are skipped without a CUDA gpu (and under AddressSanitizer, where the driver can't be loaded).

> [!TIP]
> The lab (a page that runs the layered and the brain networks side by side on logic, curve, 2D and 3D tasks, with a C++ server engine on this library) is a local tool: it is not part of the repository.

> [!WARNING]
> The `Optimized` mode builds with `-ffast-math`: the compiler removes the finite checks (`std::isfinite`), so a NaN coefficient, range or limit is not refused and 3 tests fail in this mode. Run the tests in `Debug` or `Asan`.

## Workflows/Release
### Workflows
- `Dispatch (CI/CD)`: runs on every push (branch `main` or tag `v*`, not for `*.md` / `*.txt` / `docs/` only changes) and decides what to trigger from the ref / commit message
- `Unit Tests (CI)`: builds and runs the unit tests, always first (Linux in the `unit-tests` image, Windows in the `mingw64-unit-tests` image under `wine`)
- `Build (CI)`: only builds the libraries (Linux in the `ci` image, Windows in the `mingw64-ci` image) and checks the outputs, without producing packages
- `Build - Packages (CI/CD)`: builds the packages (RPM/DEB/Arch: Linux in the `ci` image, Windows `mingw64-*` in the `mingw64-ci` image), checks them against the `PACKAGE_*` content, signs them and publishes them on the `gh-pages` repository (`fedora/44/x86_64`, `debian/`, `arch/x86_64`, 5 last `-pre` versions kept)
- `Release - GitHub (CD)`: creates the GitHub release of a tag from its section of the `CHANGELOG.md`, once the packages are published
- `Docs - Pages (CD)`: copies `docs/` on the `gh-pages` branch on every push changing it (https://tsukini22.github.io/banana-brain/docs/)
- `Sync Packages (CD)` (on `gh-pages`): rebuilds the metadata and signatures of the package repository after a manual push on the branch
- `Style - xstyle (CI)`: checks the coding style with `xstyle` (whole project on a push, changed lines on a pull request, fails on a major issue)

> [!NOTE]
> `Build (CI)` only runs when `Dispatch (CI/CD)` decides **not** to build the packages, to check that the code still compiles.

> [!IMPORTANT]
> The workflows need the repository variable `RUNNER` (`gh variable set RUNNER --body <runner>`), the containers `ghcr.io/tsukini22/ci`, `unit-tests`, `mingw64-ci`, `mingw64-unit-tests`, `package`, `git`, the secrets `GPG_PRIVATE_KEY` (base64), `GPG_KEY_ID`, `GPG_PASSPHRASE` to sign and publish the packages.

> [!TIP]
> A commit containing `[ignore]` (preferably in the description) skips the whole CI.

### Pre-Release (unstable)
The pre-release of the packages (`<package_name>-pre`) is triggered by:
- a tag with a suffix, matching `vx.x.x-<suffix>` (ex: `v1.0.1-pre`, `x` = `major`, `minor`, `fix`)
- or a commit containing `[build]`, preferably in the description

> [!NOTE]
> Only the 5 last versions of each `-pre` package are kept in the package repository, every release is kept.

### Release (stable)
The release of the packages (channel `stable`) and the GitHub release are only triggered by a tag matching exactly `vx.x.x` (ex: `v1.0.0`), whose section must exist in the `CHANGELOG.md`.

## License
![CC-BY-NC-SA-4.0](https://img.shields.io/badge/build-CC--BY--NC--SA--4.0-brightgreen?style=for-the-badge&logo=github&label=Licence&link=https%3A%2F%2Fgithub.com%2FTsukiNi22%2Fbanana-brain%2Fblob%2Fmain%2FLICENSE.md)

This project is licensed under [CC BY-NC-SA 4.0](LICENSE.md) (`CC-BY-NC-SA-4.0`):
- free for personal and educational use, modification and sharing, with credit to **Tsukini**
- any adapted work must be shared under the same license and keep the credit
- no commercial use, of this project or of any adapted work, without the author's agreement

> [!NOTE]
> For commercial use, contact [Tsukini on GitHub](https://github.com/TsukiNi22) to discuss a separate license.
