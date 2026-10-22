# banana-brain

> [!TIP]
> Technical documentation and project graph: [tsukini22.github.io/banana-brain/docs](https://tsukini22.github.io/banana-brain/docs/) (v0.0.0).

C++20 neural network library (`brain::`): one `brain::NeuralNetwork` handling several kinds of networks (layered, brain), their configuration, training and usage, plus the dataset, randomness and serialization tools.
Built as a static library on top of [libutils](https://github.com/TsukiNi22/libutils) and distributed as RPM/DEB packages (optimized, debug and asan variants, plus the `mingw64-libbrain*` packages for Windows), or installed from the sources with CMake.

> [!NOTE]
> The library is in progress (the tag `v0.0.0` was the architecture only). Written: the checks and errors of every function, the memories of the neurons (`utils::system::HybridMemory`, GPU when available), the default setup of the three networks, the serialization (`save` / `load`) and the tools. The forward pass of the three networks is written (`apply`, `exec`); **not written yet**: the backpropagation of the networks.

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

| Name | Version | Fedora (`dnf`) | Debian/Ubuntu (`apt`) |
| ---- | ------- | -------------- | --------------------- |
| `clang++` | C++20 | `clang` | `clang` |
| `cmake` | >= 3.20 | `cmake` | `cmake` |
| `python3` (exception header generation) | >= 3.10 | `python3` | `python3` |
| libutils | >= 4.1.0 | `libutils` (+ `libutils-db` / `libutils-as` for the `Debug` / `Asan` modes) | `libutils` (+ `libutils-db` / `libutils-as`) |
| GoogleTest (tests only) | — | `gtest-devel` | `libgtest-dev` |
| MinGW-w64 (Windows only) | GCC >= 13 | `mingw64-gcc-c++ mingw64-libutils` | `g++-mingw-w64-x86-64` + libutils built for MinGW |
| Wine (Windows tests only) | — | `wine` | `wine64` |

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
> Restriction: `fedora-based (rpm)`, `debian-based (deb)`

> [!NOTE]
> The usage of `sudo` in the script can be removed with the `--no-sudo` argument.

Run the setup script directly, without cloning the repository. It registers the banana-brain package repository so the system package manager can install the packages, then installs `libbrain`:
1. Detect the distribution from `/etc/os-release` (`rpm` or `deb` based).
2. `rpm`: add `/etc/yum.repos.d/brain.repo` and import the GPG key.
3. `deb`: add the GPG key in `/etc/apt/keyrings/brain.gpg`, the source in `/etc/apt/sources.list.d/brain.list`, then `apt-get update`.
4. Install the package `libbrain` (`--pre` for the unstable one, `--windows` for `mingw64-libbrain`), unless `--no-install`.

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
| `--pre` | Install the pre-release/unstable package (`libbrain-pre`) |
| `--windows` | Install the Windows packages (`mingw64-libbrain`, MinGW sysroot) instead of the Linux ones |
| `--no-install` | Only setup the mirror & the GPG key, don't install the package |
| `-h`, `--help` | Display the usage |

## Windows (cross compilation)
> Build `libbrain.a` for Windows (x86_64) from Linux with the MinGW-w64 toolchain

From the packages (after the `setup.sh` of [Quick Setup - 2](#quick-setup---2-limited)), installed in the MinGW sysroot:

```bash
curl -fsSL https://raw.githubusercontent.com/TsukiNi22/banana-brain/main/setup.sh | bash -s -- --windows   # mingw64-libbrain (--pre for the unstable one)
sudo dnf install mingw64-libbrain-db        # Fedora: Debug variant (requires mingw64-libutils-db)
```

From the sources:

```bash
export BUILD_DIR=build-windows
cmake -S . -B $BUILD_DIR -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/mingw-w64.cmake -DCMAKE_BUILD_TYPE=Optimized
cmake --build $BUILD_DIR --parallel $(nproc)
cmake --install $BUILD_DIR --prefix <windows_prefix>          # include/, lib/libbrain.a, lib/cmake/brain/
cmake --build $BUILD_DIR --target package_release               # mingw64-libbrain-* RPM/DEB (MinGW sysroot)
```

> [!NOTE]
> No AddressSanitizer with MinGW: the `Asan` mode is built like `Debug`. The tests (`-DBUILD_TESTS=ON`) download GoogleTest and run with `wine` through `ctest`.

## Usage

```cmake
find_package(brain 0.0.0 REQUIRED)                            # installed CMake configuration (finds libutils too)
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
| `brain/NeuralNetwork.hpp` | `brain::NeuralNetwork` (`utils::type::Freezable`): `init<NetworkType>(...)` (`Layered`: `(layer, neuron_per_layer)` or `(Layers)`, `Brain`: `(neurons, size)`), `weightInit`, `biasInit`, the settings of the network (`size`, `rand`, `activationFunction`... applied to it: `init` first), `learningRate`, `lossFunction`, `train<mean>(round \| loss, inputs, expected)`, manual training (`inputs`, `apply`, `backpropagation`, `loss<mean>`), `exec`, `save(path)` / `load(path)`, `clear` |
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
> The first argument of `train` must be exactly a `brain::UIType` (number of rounds) or a `brain::FType` (loss to reach): `train(10, ...)` with an `int` is ambiguous, write `brain::UIType{10}` or `0.1f`. `train<false>` (default) gives the backpropagation one coefficient per output, `train<true>` a single one.

> [!NOTE]
> The size, random and activation settings belong to the network: they throw `NoNetwork` before `init<...>`. `exec` executes without storing anything in the network (thread-safe) and is the only call allowed once the network is frozen (`freeze()`).

### Execution
- `LayerNetwork`: layer after layer, each neuron is `activation(bias + Σ weight × output of the previous layer)`; the inputs feed the first layer and the outputs are the last layer.
- `BrainNetwork` / `RawBrainNetwork`: **time steps**. The first `input` neurons (ids `0..input-1`) hold the inputs, the last `output` neurons (ids `n-output..n-1`) are the outputs (`size(input, output)` is needed), and at each step every other neuron computes `activation(bias + Σ weight × output of the neuron it is linked to, at the previous step)`; the outputs are read after the last step. On a `RawBrainNetwork` the input neurons are placed on the left edge of the space and the output ones on the right edge, evenly spread; changing the sizes places the neurons again.
- The number of steps is `min(percentage × neurons, hook(neurons))`, at least 1 (`stepsCap(percentage | fn)`, `resetStepsCap()`, getter `stepsCap()`, same as `linksCap`; default: 25 % of the neurons and `2 log2(n)`, `BRAIN_STEPS_CAP_*`). `threshold(value)` / `threshold()` cancel (set to 0) a signal whose absolute value is under it at each step, to stop signals too weak to matter (`BRAIN_SIGNAL_THRESHOLD`, a placeholder at 0 = nothing cancelled).

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
> The tests use GoogleTest (`tests/`, one file per module) and run in the `Unit Tests - Libraries (CI)` workflow.

## Workflows/Release
### Workflows
- `Dispatch (CI/CD)`: runs on every push (branch `main` or tag `v*`, not for `*.md` / `*.txt` / `docs/` only changes) and decides what to trigger from the ref / commit message
- `Unit Tests (CI)`: builds and runs the unit tests, always first (Linux in the `unit-tests` image, Windows in the `mingw64-unit-tests` image under `wine`)
- `Build (CI)`: only builds the libraries (Linux in the `ci` image, Windows in the `mingw64-ci` image) and checks the outputs, without producing packages
- `Build - Packages (CI/CD)`: builds the packages (RPM/DEB: Linux in the `ci` image, Windows `mingw64-*` in the `mingw64-ci` image), checks them against the `PACKAGE_*` content, signs them and publishes them on the `gh-pages` repository
- `Release - GitHub (CD)`: creates the GitHub release of a tag from its section of the `CHANGELOG.md`, once the packages are published
- `Docs - Pages (CD)`: copies `docs/` on the `gh-pages` branch on every push changing it (https://tsukini22.github.io/banana-brain/docs/)
- `Sync Packages (CD)` (on `gh-pages`): rebuilds the metadata and signatures of the package repository after a manual push on the branch
- `Style - xstyle (CI)`: checks the coding style with `xstyle` (whole project on a push, changed lines on a pull request, fails on a major issue)
- `Merge - libutils (CD)`: reserved for the future automatic merge of the library into libutils, does nothing yet (manual trigger only)

> [!NOTE]
> `Build (CI)` only runs when `Dispatch (CI/CD)` decides **not** to build the packages, to check that the code still compiles.

> [!IMPORTANT]
> The workflows need the repository variable `RUNNER` (`gh variable set RUNNER --body <runner>`), the containers `ghcr.io/tsukini22/ci`, `unit-tests`, `mingw64-ci`, `mingw64-unit-tests`, `package`, `git`, the secrets `GPG_PRIVATE_KEY` (base64), `GPG_KEY_ID`, `GPG_PASSPHRASE` to sign and publish the packages, and `DOWNSTREAM_REPO_TOKEN` for the downstream notification.

> [!TIP]
> A commit containing `[ignore]` (preferably in the description) skips the whole CI.

### Pre-Release (unstable)
The pre-release of the packages (`<package_name>-pre`) is triggered by:
- a tag with a suffix, matching `vx.x.x-<suffix>` (ex: `v1.0.1-pre`, `x` = `major`, `minor`, `fix`)
- or a commit containing `[build]`, preferably in the description

> [!NOTE]
> Only the 5 last versions of each `-pre` package are kept in the package repository, every release is kept.

### Release (stable)
The release of the packages (channel `stable`) and the GitHub release are only triggered by a tag matching exactly `vx.x.x` (ex: `v0.0.0`), whose section must exist in the `CHANGELOG.md`.

### Downstream notification
`Dispatch (CI/CD)` notifies the `docker-image` repository (`notify-downstream` job: `repository_dispatch`, event `banana-brain-released`, secret `DOWNSTREAM_REPO_TOKEN`) after the packages are published:
- on every package build (release or pre-release)
- or on a commit containing `[release]`, preferably in the description

## License
![CC-BY-NC-SA-4.0](https://img.shields.io/badge/build-CC--BY--NC--SA--4.0-brightgreen?style=for-the-badge&logo=github&label=Licence&link=https%3A%2F%2Fgithub.com%2FTsukiNi22%2Fbanana-brain%2Fblob%2Fmain%2FLICENSE.md)

This project is licensed under [CC BY-NC-SA 4.0](LICENSE.md) (`CC-BY-NC-SA-4.0`):
- free for personal and educational use, modification and sharing, with credit to **Tsukini**
- any adapted work must be shared under the same license and keep the credit
- no commercial use, of this project or of any adapted work, without the author's agreement

> [!NOTE]
> For commercial use, contact [Tsukini on GitHub](https://github.com/TsukiNi22) to discuss a separate license.
