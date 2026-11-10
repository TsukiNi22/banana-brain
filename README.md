# banana-brain

> [!TIP]
> Documentation [TsukiNi22/banana-brain](https://tsukini22.github.io/banana-brain/docs/) (v1.0.0): technical documentation (coming soon) and the interactive project graph.

C++20 neural network library (`libbrain.a`, `find_package(brain)`, namespace `brain::`) built with CMake, `clang++`, ccache and [libutils](https://github.com/TsukiNi22/libutils); the exception codes are generated from JSON.
Generated from [cpp_project_template](https://github.com/TsukiNi22/cpp_project_template): Linux and Windows (MinGW-w64 cross compilation) builds, RPM / DEB / Arch (pacman) packages and the libutils workflows.

> [!NOTE]
> Restarted from the start for a clean global rework: the classes (`NeuralNetwork`, `LayerNetwork`, `BrainNetwork`, `RawBrainNetwork`) only have their base methods (init, serialization, execution, backpropagation, loss / activation functions) with empty bodies, the tools (`brain::tools::`) are kept.

### Table of Contents
 - [Dependencies](#dependencies)
 - [Packages](#packages)
 - [Quick Setup - 1 (all)](#quick-setup---1-all)
 - [Quick Setup - 2 (Limited)](#quick-setup---2-limited)
 - [Usage](#usage)
 - [Windows (cross compilation)](#windows-cross-compilation)
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

| Name | Version | Fedora (`dnf`) | Debian/Ubuntu (`apt`) | Arch (`pacman`) |
| ---- | ------- | -------------- | --------------------- | --------------- |
| `clang++` / `cmake` | C++20 / `>= 3.20` | `clang cmake` | `clang cmake` | `clang cmake` |
| `python3` | `>= 3.8` | `python3` | `python3` | `python` |
| `libutils` | `>= 4.3.0` (stable or `-pre`, Arch: the stable package only) | `libutils` | `libutils` | `libutils` |
| GoogleTest (tests only) | any | `gtest-devel` | `libgtest-dev` | `gtest` |
| MinGW-w64 (Windows only) | GCC `>= 13` | `mingw64-gcc-c++ mingw64-libutils` | `g++-mingw-w64-x86-64` + libutils built for MinGW | — |
| `rpm-build`, `dpkg`, `bsdtar` (packages only, `bsdtar` for the Arch ones) | any | `rpm-build dpkg bsdtar` | `rpm dpkg-dev libarchive-tools` | `rpm-tools dpkg libarchive` |
| `ccache` **[OPTIONAL]** | any | `ccache` | `ccache` | `ccache` |

> [!TIP]
> libutils packages and source install: see the [libutils README](https://github.com/TsukiNi22/libutils#packages).

## Packages

> [!NOTE]
> Every package has a pre-release/unstable version named `<package_name>-pre`.
> The `-pre` packages are marked as obsolete by any release/stable package (without `-pre`) of the same version or higher.
>
> The same packages exist as RPM (Fedora), DEB (Debian/Ubuntu) and Arch Linux (pacman, `.pkg.tar.zst`): same names, content and dependencies (on Arch the libraries are in `/usr/lib`).

| File Name | Content |
| --------- | ------- |
| `libbrain` | Install the default packages (`libbrain-dev`, `libbrain-op`) |
| `libbrain-dev` | All the includes and the global CMake configuration (requires `libutils-dev`) |
| `libbrain-dev-op` | CMake configuration for `libbrain-op` |
| `libbrain-dev-db` | CMake configuration for `libbrain-db` |
| `libbrain-dev-as` | CMake configuration for `libbrain-as` |
| `libbrain-op` | Library compiled with the optimization options (`libbrain.a`) |
| `libbrain-db` | Library compiled with the debug options (`libbrain_debug.a`, intended for debugging only) |
| `libbrain-as` | Library compiled with the asan options (`libbrain_asan.a`, intended for debugging only) |

**Windows** (MinGW-w64 cross compilation, installed in the MinGW sysroot: `/usr/x86_64-w64-mingw32/sys-root/mingw` on Fedora, `/usr/x86_64-w64-mingw32` on Debian/Ubuntu): the same packages prefixed with `mingw64-` (ex: `mingw64-libbrain`), without the `-as` variants (no AddressSanitizer with MinGW).

> [!NOTE]
> The content of the packages is chosen at the top of the `CMakeLists.txt`: `BRAIN_KIND` (`library`) sets the defaults of `BRAIN_PACKAGE_BIN` (executable), `BRAIN_PACKAGE_LIB` (library, `BRAIN_SHARED_LIB` for a `.so`) and `BRAIN_PACKAGE_DEV` (headers + CMake config), each one can be forced with `-D...=ON/OFF`.
> After a change, update `PACKAGE_BIN` / `PACKAGE_LIB` / `PACKAGE_DEV` in `.github/workflows/build.yml` and `build-packages.yml`.

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
sudo cmake --build $BUILD_DIR --target install_release --parallel $(nproc) # Debug, Asan & Optimized variants
```

### Include
```cmake
find_package(brain 1.0.0 CONFIG REQUIRED)
target_link_libraries(<target> PRIVATE brain::brain)
```

> [!WARNING]
> Everything is defined within the namespace `brain::` (tools in `brain::tools::`).

| Include | Content |
| ------- | ------- |
| `brain/brain.hpp` | Global include of the library (types, tools, `brain::NeuralNetwork`, `brain::version`) |
| `brain/NeuralNetwork.hpp` | `brain::NeuralNetwork`: `init<NetworkType>(...)`, `weightInit` / `biasInit`, the settings applied to the network, `train`, the manual training (`inputs`, `apply`, `backpropagation`, `loss`), `exec`, `save` / `load`, `clear`, `get()` (pointer to the internal network, kept by the `NeuralNetwork`, `nullptr` without one) |
| `brain/network/Network.hpp` | `brain::INetwork` -> `brain::ANetwork` -> `brain::LayerNetwork` / `brain::BrainNetwork` -> `brain::RawBrainNetwork` |
| `brain/tools/*.hpp` | `activation` (`sigmoid`, `linear`, `relu`), `bytes` (serialization), `dataset`, `loss` (`mse`, `cross_entropy`), `neuron` (`sort`), `random` |

## Quick Setup - 2 (Limited)
> Setup into `/usr`

The packages are distributed by [my-universe](https://tsukini22.github.io/my-universe/), the package repository of every project of TsukiNi22 (libutils, brain, xstyle, context-forge...):

```bash
curl -fsSL https://tsukini22.github.io/my-universe/install.sh | bash -s
sudo dnf install libbrain       # or: sudo apt install libbrain / sudo pacman -S libbrain (libbrain-pre for the unstable one)
```

## Usage

```bash
make                                                      # Configure & build into build/
make re                                                   # Full rebuild
make fclean                                               # Remove the build directory

export BUILD_DIR=build
cmake -S . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Debug         # Debug build (-g -ggdb3)
cmake -S . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Asan          # AddressSanitizer build
cmake --build $BUILD_DIR --target release                 # Debug, Asan & Optimized builds
cmake --build $BUILD_DIR --target install_release         # Install the three variants
cmake --build $BUILD_DIR --target package_release         # RPM / DEB / Arch packages of every variant (build/*.rpm, *.deb, *.pkg.tar.zst)
cmake --build $BUILD_DIR --target get_unregistered_files  # .cpp files missing from the SRC / SRC_SHARED lists
```

| Option | Default | Content |
| ------ | ------- | ------- |
| `BRAIN_KIND` | `library` | Project kind (`binary`, `library`, `header-only`): defaults of the three options below |
| `BRAIN_PACKAGE_BIN` | `OFF` | Build and package the executable (`brain-op` / `-db` / `-as`) |
| `BRAIN_PACKAGE_LIB` | `ON` | Build and package the library (`libbrain-op` / `-db` / `-as`) |
| `BRAIN_PACKAGE_DEV` | `ON` | Package the public headers and the CMake config (`libbrain-dev`) |
| `BRAIN_SHARED_LIB` | `OFF` | Shared library (`.so` / `.dll`) instead of a static one |
| `BRAIN_STABLE_RELEASE` | `OFF` | Package channel: stable (`ON`) or pre-release `-pre` (`OFF`) |
| `BUILD_TESTS` | `OFF` | Build the unit tests (`unit_tests`, GoogleTest) |

| Path | Content |
| ---- | ------- |
| `src/brain/` | Sources: `NeuralNetwork.cpp`, `network/` (the networks), `tools/` (the tools) |
| `include/brain/` | Public headers |
| `tests/` | Unit tests (one file per module) |
| `cmake/config/exceptions/*.json` | Project exception codes (`utils::exception::ExternalCode`) |
| `cmake/scripts/` | Exception header generator (`generate_exception_header.py`, `const.py`), `check_files.sh` |
| `cmake/toolchain/mingw-w64.cmake` | Windows cross compilation toolchain |
| `cmake/package/` | CMake package config (`find_package(brain)`) and the CPack prefix of the Windows packages |
| `docs/` | HTML documentation (technical page, project graph) |
| `include/brain/exception/generated_external_exception_header.hpp` | Generated at build time from the JSON files (not committed) |

> [!NOTE]
> Every new `.cpp` must be added to the `SRC` (executable only) or `SRC_SHARED` (executable and library) list of the `CMakeLists.txt` (`get_unregistered_files` lists the missing ones).

## Windows (cross compilation)
> Build for Windows (x86_64) from Linux with the MinGW-w64 toolchain (libutils: `mingw64-libutils` packages in the MinGW sysroot)

```bash
export BUILD_DIR=build-windows
cmake -S . -B $BUILD_DIR -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/mingw-w64.cmake -DCMAKE_BUILD_TYPE=Optimized
cmake --build $BUILD_DIR --parallel $(nproc)                 # build-windows/libbrain.a
cmake --build $BUILD_DIR --target package_release           # mingw64-libbrain* RPM/DEB (MinGW sysroot)
```

> [!NOTE]
> No AddressSanitizer with MinGW: the `Asan` mode is built like `Debug`. Portable code only: system calls through `utils::platform::`, `<windows.h>` before `<utils/utils.hpp>`.

## Unit tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build --parallel $(nproc)                       # ./unit_tests at the root
(cd build && ctest --output-on-failure)
```

## Workflows/Release
### Workflows
- `Dispatch (CI/CD)`: runs on every push (branch `main` or tag `v*`, not for `*.md` / `*.txt` / `docs/` only changes) and decides what to trigger from the ref / commit message
- `Unit Tests (CI)`: builds and runs the unit tests, always first (Linux in the `unit-tests` image, Windows in the `mingw64-unit-tests` image under `wine`)
- `Build (CI)`: only builds the project (Linux in the `ci` image, Windows in the `mingw64-ci` image) and checks the outputs, without producing packages
- `Build - Packages (CI/CD)`: builds the packages (RPM/DEB/Arch: Linux in the `ci` image, Windows `mingw64-*` in the `mingw64-ci` image), signs them and publishes them on the `gh-pages` repository (listed by my-universe)
- `Release - GitHub (CD)`: creates the GitHub release of a tag from its section of the `CHANGELOG.md`, once the packages are published
- `Docs - Pages (CD)`: copies `docs/` on the `gh-pages` branch on every push changing it (https://tsukini22.github.io/banana-brain/docs/)
- `Sync Packages (CD)` (on `gh-pages`): rebuilds the metadata and signatures of the package repository after a manual push on the branch
- `Style - xstyle (CI)`: checks the coding style with `xstyle` (whole project on a push, changed lines on a pull request, fails on a major issue)

> [!NOTE]
> `Build (CI)` only runs when `Dispatch (CI/CD)` decides **not** to build the packages, to check that the code still compiles.

> [!IMPORTANT]
> The workflows need the repository variable `RUNNER` (`gh variable set RUNNER --body <runner>`), the containers `ghcr.io/tsukini22/ci`, `unit-tests`, `mingw64-ci`, `mingw64-unit-tests`, `package`, `git`, and the secrets `GPG_PRIVATE_KEY` (base64), `GPG_KEY_ID`, `GPG_PASSPHRASE` to sign and publish the packages.

> [!TIP]
> A commit containing `[ignore]` skips the whole CI.

### Pre-Release (unstable)
The pre-release of the packages (`<package_name>-pre`) is triggered by:
- a tag with a suffix, matching `vx.x.x-<suffix>` (ex: `v1.0.1-pre`, `x` = `major`, `minor`, `fix`)
- or a commit containing `[build]`

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
