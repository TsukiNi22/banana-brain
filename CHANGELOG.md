# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

| Scope | Content |
| ----- | ------- |
| `In Progress` | Work that hasn't been completed yet (typically listed under [pre-release]) |
| `Added`       | New features, files, or sections introduced in this release |
| `Removed`     | Existing features, files, or sections that have been deleted |
| `Changed`     | Existing features or sections that were modified (behavior, structure, or content) |
| `Fixed`       | Bugs or issues that have been resolved |

> [!NOTE]
> The `(unofficial)` version are sub-versions (typically fixes) that don't have associated tag.
> A package built at this version is not guaranteed!

---

## [pre-release] (empty)

## v1.0.0 - 2026-10-17
### Added
- Architecture of the library (every signature, no logic yet): `brain::NeuralNetwork` (`utils::type::Freezable`, `init<NetworkType>`, configuration setters / getters, `train`, manual training, `exec`, `load` / `save`), the network family `brain::INetwork` -> `brain::ANetwork` -> `brain::LayerNetwork` / `brain::BrainNetwork`, the enums `NetworkType` / `ActivationFunction` / `LossFunction`, the value types `IType` / `UIType` / `FType`, the neuron structs `LayerNeuron` / `BrainNeuron` (cache line aligned) and the tools `brain::tools::` (`generator`, `rand`, `rand_unsigned_integer`, `rand_floating`, `shuffle`, `rand_sets`, `text_to_input`, `input_to_text`)
- Exception codes of the library (`utils::exception::ExternalCode::NoNetwork`, `NetworkNotLoaded`, `FrozenNetwork`, `InvalidInputSize`...) generated from `cmake/config/exceptions/brain.json` and installed with the headers
- Static library `libbanana-brain.a` (`_debug`, `_asan`) with `find_package(banana-brain)` / `banana-brain::banana-brain`, RPM/DEB packages like libutils (`libbanana-brain`, `-dev`, `-dev-op` / `-dev-db` / `-dev-as`, `-op` / `-db` / `-as`, `-pre` channel) and the Windows ones (`mingw64-libbanana-brain*`, MinGW-w64 cross compilation)
- Unit tests (GoogleTest, `tests/`) of the default states, the types and the compile time checks of `train`
- Workflows of [cpp_project_template](https://github.com/TsukiNi22/cpp_project_template): `Dispatch (CI/CD)`, `Unit Tests (CI)`, `Build (CI)`, `Build - Packages (CI/CD)`, `Release - GitHub (CD)`, `Docs - Pages (CD)`, plus `Style - xstyle (CI)` and `Merge - libutils (CD)` (reserved, does nothing yet)
- HTML documentation (`docs/`: technical page + interactive project graph, EN / FR, published on GitHub Pages)
- `setup.sh` (package repository + GPG key + installation), README, CHANGELOG, license CC BY-NC-SA 4.0

[v1.0.0]: https://github.com/TsukiNi22/banana-brain/releases/tag/v1.0.0
