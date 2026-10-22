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

## [pre-release]
### In Progress
- The forward pass (`compute_`, `apply()`) and the backpropagation of the networks, still empty (`/* Nothing */`)

### Changed
- **[MAJOR]** The brain network is split in `brain::BrainNetwork` (reduced, execution only: the training calls throw `ReadOnlyNetwork`) and `brain::RawBrainNetwork` (execution and training, derives from it and gives a reduced copy with `reduce()`); `init<Brain>` creates the raw one
- **[MAJOR]** The neurons and the links are trivially copyable structs stored in `utils::system::HybridMemory` (GPU when available, else the RAM): `LayerNeuron` (`incoming`, `count`, `bias`) with the weights in their own memory, `BrainNeuron` / `BrainLink` (reduced), `RawBrainNeuron` / `RawBrainLink` (raw, a link is found twice with the same weight: the receiver for the execution, the clone on the sender for the backpropagation); no cache line alignment any more
- **[MAJOR]** The input / output sizes, the random range / function and the activation function belong to the network (`INetwork` / `ANetwork`) and `NeuralNetwork` applies its setters to it: they throw `NoNetwork` before `init<...>`
- **[MAJOR]** `train` is `train<mean>(round | loss, inputs, expected_outputs)` without hook: the loss of the outputs (times the learning rate) is the coefficient of the backpropagation, one per output (`mean = false`) or the mean (`mean = true`)
- `exec` executes without storing anything in the network (`apply(inputs) const`, thread-safe); `inputs()` / `apply()` are the non const forward of the training
- `init<Layered>` also takes the number of neurons of each layer (`brain::Layers`), the networks check their parameters and the current network is kept when they are invalid
- `loss` has the versions `loss<true>` (mean), `loss<false>` (each output) and the pair of both, with an optional coefficient
- `text_to_input` / `input_to_text` have the modes `Normal` (one value per character, byte / 255) and `Compact` (the bytes copied in the values)
- `brain::tools::sort(index, neurons, cap)` returns the ids of the `cap` nearest neurons, closest first, instead of filling a list in the neurons
- The networks share one GPU handler per mode for the whole program (finding a GPU takes a few hundred ms)

### Fixed
- `loss.cpp` and `activation.cpp` were only in the sources of the executable: they are in the library
- `BRAIN_LINKS_CAP_COMPUTING` overflowed on a big network (`pow`): written as `2 log2(n) ln(n) / ln(1.1)`
- `brain::tools::get<bool>` read an undefined `bool` from a byte other than 0 or 1

### Added
- Every function checks the frozen state, the network, the sizes, the datasets and its parameters (`NoNetwork`, `InvalidInputSize`, `InvalidOutputSize`, `InvalidCoefficient`, `InvalidLearningRate`, `InvalidArgument`...)
- Default setup of the networks (`init_`): the layers and the weights, the reduced neurons and slots, the neurons of the raw network placed randomly and linked to their nearest ones (limit `linksCap`: the smallest of a percentage of the neurons and a function of their number), and the random `weightInit` / `biasInit` (range or function, `fallback` on 0)
- Serialization and files: `serialize()` / `deserialize()` of every network, `NeuralNetwork::save(path)` / `load(path)` (`.ln`, `.bn`, `.rbn`, `.net`, magic bytes `0x22` + `'l'` / `'b'` / `'r'`, activation byte), the refusal of a bad, truncated or inconsistent file, and the warning for the functions that can't be saved
- Exception codes `ReadOnlyNetwork`, `CustomActivationRequired`, `CustomRandomRequired`, `CustomLinksCapRequired`
- Tools: the losses (`mse`, `cross_entropy`), the activations (`sigmoid`, `linear`, `relu`), the little endian bytes (`put`, `get`, `fits`), `shuffle`, `rand_sets`, the generator used by the positions and the random weights
- Unit tests of the types, the checks, the three networks, the files, `NeuralNetwork` and the tools

### Removed
- `NetworkType::Banana`, the hooks `BackpropagationFn` / `BackpropagationsFn`, `NeuralNetwork::seed` and the memory `load()` / `unload()` of the networks (the codes `NetworkNotLoaded`, `NetworkAlreadyLoaded`, `FrozenNetwork`, `DeadNetwork` and `InvalidPercentage` stay reserved, not thrown yet)

## v0.0.0 - 2026-10-17
### Added
- Nothing executes yet: the project, its packages and its documentation exist, every class and function has its final signature with an empty body (the logic comes with the next versions)
- Architecture of the library (every signature, no logic yet): `brain::NeuralNetwork` (`utils::type::Freezable`, `init<NetworkType>`, configuration setters / getters, `train`, manual training, `exec`, `load` / `save`), the network family `brain::INetwork` -> `brain::ANetwork` -> `brain::LayerNetwork` / `brain::BrainNetwork`, the enums `NetworkType` / `ActivationFunction` / `LossFunction`, the value types `IType` / `UIType` / `FType`, the neuron structs `LayerNeuron` / `BrainNeuron` (cache line aligned) and the tools `brain::tools::` (`generator`, `rand`, `rand_unsigned_integer`, `rand_floating`, `shuffle`, `rand_sets`, `text_to_input`, `input_to_text`)
- Exception codes of the library (`utils::exception::ExternalCode::NoNetwork`, `NetworkNotLoaded`, `FrozenNetwork`, `InvalidInputSize`...) generated from `cmake/config/exceptions/brain.json` and installed with the headers
- Static library `libbrain.a` (`_debug`, `_asan`) with `find_package(brain)` / `brain::brain`, RPM/DEB packages like libutils (`libbrain`, `-dev`, `-dev-op` / `-dev-db` / `-dev-as`, `-op` / `-db` / `-as`, `-pre` channel) and the Windows ones (`mingw64-libbrain*`, MinGW-w64 cross compilation)
- Unit tests (GoogleTest, `tests/`) of the default states, the types and the compile time checks of `train`
- Workflows of [cpp_project_template](https://github.com/TsukiNi22/cpp_project_template): `Dispatch (CI/CD)`, `Unit Tests (CI)`, `Build (CI)`, `Build - Packages (CI/CD)`, `Release - GitHub (CD)`, `Docs - Pages (CD)`, plus `Style - xstyle (CI)` and `Merge - libutils (CD)` (reserved, does nothing yet)
- HTML documentation (`docs/`: technical page + interactive project graph, EN / FR, published on GitHub Pages)
- `setup.sh` (package repository + GPG key + installation), README, CHANGELOG, license CC BY-NC-SA 4.0

[v0.0.0]: https://github.com/TsukiNi22/banana-brain/releases/tag/v0.0.0
