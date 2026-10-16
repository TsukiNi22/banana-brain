# banana-brain

> [!TIP]
> Technical documentation and project graph: [tsukini22.github.io/banana-brain/docs](https://tsukini22.github.io/banana-brain/docs/) (v1.0.0).

C++20 neural network library (`brain::`): one `brain::NeuralNetwork` handling several kinds of networks (layered, brain, banana), their configuration, training and usage, plus the dataset and randomness tools.
Built as a static library on top of [libutils](https://github.com/TsukiNi22/libutils) and distributed as RPM/DEB packages (optimized, debug and asan variants, plus the `mingw64-libbanana-brain*` packages for Windows), or installed from the sources with CMake.

> [!NOTE]
> The library is an architecture for now: every class, function and exception code exists with its final signature, the logic comes with the next versions.

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
| `libbanana-brain` | Install the default packages (`libbanana-brain-dev`, `libbanana-brain-op`) |
| `libbanana-brain-dev` | All the includes (exception codes included) and the global CMake configuration (requires `libutils-dev`) |
| `libbanana-brain-dev-op` | CMake configuration for `libbanana-brain-op` |
| `libbanana-brain-dev-db` | CMake configuration for `libbanana-brain-db` |
| `libbanana-brain-dev-as` | CMake configuration for `libbanana-brain-as` |
| `libbanana-brain-op` | Library compiled with the optimization options (requires `libutils-op`) |
| `libbanana-brain-db` | Library compiled with the debug options (requires `libutils-db`, intended for debugging only) |
| `libbanana-brain-as` | Library compiled with the asan options (requires `libutils-as`, intended for debugging only) |

**Windows** (MinGW-w64 cross compilation, installed in the MinGW sysroot: `/usr/x86_64-w64-mingw32/sys-root/mingw` on Fedora, `/usr/x86_64-w64-mingw32` on Debian/Ubuntu)

| File Name | Content |
| --------- | ------- |
| `mingw64-libbanana-brain` | Install the default Windows packages (`mingw64-libbanana-brain-dev`, `mingw64-libbanana-brain-op`) |
| `mingw64-libbanana-brain-dev` | All the includes and the global CMake configuration (requires `mingw64-libutils-dev`) |
| `mingw64-libbanana-brain-dev-op` | CMake configuration for `mingw64-libbanana-brain-op` |
| `mingw64-libbanana-brain-dev-db` | CMake configuration for `mingw64-libbanana-brain-db` |
| `mingw64-libbanana-brain-op` | Windows library compiled with the optimization options |
| `mingw64-libbanana-brain-db` | Windows library compiled with the debug options (no asan version with MinGW) |

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

Run the setup script directly, without cloning the repository. It registers the banana-brain package repository so the system package manager can install the packages, then installs `libbanana-brain`:
1. Detect the distribution from `/etc/os-release` (`rpm` or `deb` based).
2. `rpm`: add `/etc/yum.repos.d/banana-brain.repo` and import the GPG key.
3. `deb`: add the GPG key in `/etc/apt/keyrings/banana-brain.gpg`, the source in `/etc/apt/sources.list.d/banana-brain.list`, then `apt-get update`.
4. Install the package (`libbanana-brain`, or `libbanana-brain-pre` with `--pre`), unless `--no-install`.

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
| `--pre` | Install the pre-release/unstable package (`libbanana-brain-pre`) |
| `--no-install` | Only setup the mirror & the GPG key, don't install the package |
| `-h`, `--help` | Display the usage |

## Windows (cross compilation)
> Build `libbanana-brain.a` for Windows (x86_64) from Linux with the MinGW-w64 toolchain

From the packages (after the `setup.sh` of [Quick Setup - 2](#quick-setup---2-limited)), installed in the MinGW sysroot:

```bash
sudo dnf install mingw64-libbanana-brain-pre       # Fedora: + mingw64-libbanana-brain-db-pre for Debug (requires mingw64-libutils)
sudo apt install mingw64-libbanana-brain-pre       # Debian/Ubuntu: libutils for MinGW must be built by hand
```

From the sources:

```bash
export BUILD_DIR=build-windows
cmake -S . -B $BUILD_DIR -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/mingw-w64.cmake -DCMAKE_BUILD_TYPE=Optimized
cmake --build $BUILD_DIR --parallel $(nproc)
cmake --install $BUILD_DIR --prefix <windows_prefix>          # include/, lib/libbanana-brain.a, lib/cmake/banana-brain/
cmake --build $BUILD_DIR --target package                       # mingw64-libbanana-brain-* RPM/DEB (MinGW sysroot)
```

> [!NOTE]
> No AddressSanitizer with MinGW: the `Asan` mode is built like `Debug`. The tests (`-DBUILD_TESTS=ON`) download GoogleTest and run with `wine` through `ctest`.

## Usage

```cmake
find_package(banana-brain 1.0.0 REQUIRED)                            # installed CMake configuration (finds libutils too)
target_link_libraries(${TARGET} PRIVATE banana-brain::banana-brain)  # includes, utils::utils & library
```

The same CMake code works for Linux and Windows (MinGW toolchain, see [Windows](#windows-cross-compilation)). The library follows the `CMAKE_BUILD_TYPE` of the project, and needs the libutils of the same mode:

| Build type | Library | Package (Linux) | Package (Windows) |
| ---------- | ------- | --------------- | ----------------- |
| `Debug` | `libbanana-brain_debug.a` | `libbanana-brain-db` | `mingw64-libbanana-brain-db` |
| `Asan` | `libbanana-brain_asan.a` | `libbanana-brain-as` | none (no AddressSanitizer with MinGW, use `Debug`) |
| anything else | `libbanana-brain.a` | `libbanana-brain-op` (in `libbanana-brain`) | `mingw64-libbanana-brain-op` (in `mingw64-libbanana-brain`) |

A missing package stops the configuration with the name of the package to install.

```cpp
#include <brain/brain.hpp>    // every type, tool and network (or one header below)
```

> [!WARNING]
> Everything is defined within the namespace `brain::` (tools in `brain::tools::`).

| Include | Content |
| ------- | ------- |
| `brain/brain.hpp` | Global include of the library (+ `brain::version`), without the network classes: they are internal, created by `NeuralNetwork::init<...>` |
| `brain/NeuralNetwork.hpp` | `brain::NeuralNetwork` (`utils::type::Freezable`): `init<NetworkType>(...)`, `weightInit`, `biasInit`, setters / getters (`seed`, `size`, `rand`, `learningRate`, `activationFunction`, `lossFunction`...), `train(round | loss, inputs, expected, backpropagation)`, manual training (`load`, `inputs`, `apply`, `backpropagation`, `loss`), `exec`, `load(path)` / `save(path)`, `clear` |
| `brain/NetworkDefine.hpp` | `brain::NetworkType` (`Layered`, `Brain`, `Banana`), `brain::ActivationFunction`, `brain::LossFunction` |
| `brain/NetworkType.hpp` | `brain::IType` / `UIType` / `FType`, `brain::Values` / `Dataset`, the hooks `ActivationFn` / `LossFn` / `RandFn` / `BackpropagationFn` / `BackpropagationsFn`, `brain::LayerNeuron` / `BrainNeuron` (cache line aligned; `BrainNeuron`: `id`, `ordered` <id, squared distance> list, `operator==` on the ids, `distance2()` / `operator|` squared distance) |
| `brain/network/Network.hpp` | `brain::INetwork` -> `brain::ANetwork` -> `brain::LayerNetwork(layer, neuron_per_layer)` / `brain::BrainNetwork(neuron, size)` (not in `brain.hpp`: include it only when the network classes are needed) |
| `brain/tools/random.hpp` | `brain::tools::generator` (`std::mt19937_64`, scope chosen by `BRAIN_SEED_GLOBAL` / `BRAIN_SEED_LOCAL`), `rand(seed)`, `rand_unsigned_integer()`, `rand_floating()` |
| `brain/tools/dataset.hpp` | `brain::tools::shuffle`, `rand_sets`, `text_to_input`, `input_to_text` |
| `brain/tools/neuron.hpp` | `brain::tools::sort(neuron, neurons)`: fills `neuron.ordered` with every other neuron sorted from the closest (squared distances, stored in both neurons) |

The random generator of the library (`brain::tools::generator`, used by every randomness call) has a scope chosen at compile time, with a define set **before** the include (`-DBRAIN_SEED_LOCAL` in the CMake, or `#define` on the line before):

| Define | Generator |
| ------ | --------- |
| `BRAIN_SEED_GLOBAL` (default when nothing is defined) | One single generator shared by the whole program (`extern`, defined once in the library) |
| `BRAIN_SEED_LOCAL` | One generator per translation unit including `brain/tools/random.hpp` (`static`): each file has its own independent sequence |

> [!WARNING]
> Defining both at once is a compile error (`#error`). The helpers `rand(seed)`, `rand_unsigned_integer()` and `rand_floating()` stay the same and use whichever generator is active.

> [!IMPORTANT]
> The `train` overloads only accept a `brain::UIType` (number of rounds) or a `brain::FType` (loss to reach): any other type (`int`, `double`...) is refused at compile time, no implicit conversion.

The errors are libutils exceptions with the codes of the library (`utils::exception::ExternalCode::NoNetwork`, `NetworkNotLoaded`, `FrozenNetwork`, `InvalidInputSize`, `InvalidCoefficient`...), generated from `cmake/config/exceptions/brain.json` and installed with the headers:

```cpp
#define _Exception
#include <utils/utils.hpp>
#include <brain/brain.hpp>

try {
    brain::NeuralNetwork network;
    network.init<brain::NetworkType::Layered>(2, 3);
    network.exec({0.0, 1.0});
} catch (const utils::exception::IException& e) {
    std::cerr << e.formated() << std::endl;   // [Error] ... -> No network initialized, call init<NetworkType>() first
}
```

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
- `Dispatch (CI/CD)`: runs on every push (branch `main` or tag `v*`, not for `*.md` / `*.txt` only changes) and decides what to trigger from the ref / commit message
- `Unit Tests - Libraries (CI)`: builds and runs the unit tests, always first (Linux in the `unit-tests` image, Windows in the `mingw64-unit-tests` image under `wine`)
- `Build - Packages (CI/CD)`: builds the packages (RPM/DEB: Linux in the `ci` image, Windows `mingw64-libbanana-brain*` in the `build-packages-windows` job on `mingw64-ci`), signs them and publishes them on the `gh-pages` repository
- `Build - Libraries (CI)`: only builds the libraries and checks the compilation, without producing or publishing packages
- `Style - xstyle (CI)`: checks the coding style with `xstyle` (whole project on a push, changed lines on a pull request, fails on a major issue)
- `Merge - libutils (CD)`: reserved for the future automatic merge of the library into libutils, does nothing yet (manual trigger only)

> [!NOTE]
> `Build - Libraries (CI)` only runs when `Dispatch (CI/CD)` decides **not** to build the packages, to check that the code still compiles.

> [!TIP]
> A commit containing `[ignore]` (preferably in the description) skips the whole CI.

### Pre-Release (unstable)
The pre-release of the packages (`<package_name>-pre`) is triggered by:
- a tag with a suffix, matching `vx.x.x-<suffix>` (ex: `v1.0.1-pre`, `x` = `major`, `minor`, `fix`)
- or a commit containing `[build]`, preferably in the description

> [!NOTE]
> Only the 5 last versions of each `-pre` package are kept in the package repository, every release is kept.

### Release (stable)
The release of the packages (channel `stable`) is only triggered by a tag matching exactly `vx.x.x` (ex: `v1.0.0`).

### Downstream notification
`Dispatch (CI/CD)` notifies the `docker-image` repository (`repository_dispatch`, event `banana-brain-released`) after the packages are published:
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
