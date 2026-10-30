/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 25/10/2026 by @author Tsukini

File Name:
##  @file BrainSession.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#include "brain/tools/random.hpp"
#include "BrainSession.hpp"
#include <functional>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>

lab::BrainSession::BrainSession(const nlohmann::json& options, lab::Task task)
: _options{options}, _task{std::move(task)}
{
    const brain::IType filter = options.value("filter", 0);
    const bool every = options.value("allInputs", true);

    // the seed of the page, then the next ones until the output is reached in at most <filter> links (the same rule as Regenerate)
    this->_seed = options.value("seed", 1);
    for (int tries = 0; tries < 500; ++tries) {
        this->_unsupported.clear();
        this->_network = build_(options, this->_seed, this->_unsupported);
        const std::pair<brain::IType, brain::IType> path = hops(*this->_network);
        const brain::IType reach = every ? path.second : path.first;

        if (filter == 0 || (reach >= 1 && reach <= filter)) break;
        this->_seed = (this->_seed * 48271 + 11) % 2147483647;
    }
}

std::unique_ptr<brain::RawBrainNetwork> lab::BrainSession::build_(const nlohmann::json& options, const brain::IType seed, std::vector<std::string>& unsupported)
{
    const brain::UIType neurons = options.value("neurons", 16);
    const std::string activation = options.value("activation", "sigmoid");
    const utils::system::MemoryMode mode = options.value("gpu", false) ? utils::system::MemoryMode::Auto : utils::system::MemoryMode::Ram;
    std::unique_ptr<brain::RawBrainNetwork> network = std::make_unique<brain::RawBrainNetwork>(neurons, std::pair<brain::FType, brain::FType>{options.value("width", 8.0f), options.value("height", 5.0f)}, mode);

    // the shape (every setter sets the links up again: the seed is given just before the last one, size())
    network->linksCap(options.value("linksPct", 1.0f));
    if (options.contains("linksHook") && options["linksHook"].is_number()) {
        const brain::UIType hook = options["linksHook"].get<brain::UIType>();
        network->linksCap([hook](const brain::UIType) {return hook;});
    }
    network->stepsCap(options.value("stepsPct", 1.0f));
    if (options.contains("stepsHook") && options["stepsHook"].is_number()) {
        const brain::UIType hook = options["stepsHook"].get<brain::UIType>();
        network->stepsCap([hook](const brain::UIType) {return hook;});
    }
    if (options.value("relink", "normal") == "off") network->weightLimits(0.0, std::numeric_limits<brain::FType>::max()); // never at a limit: the links never change
    else network->weightLimits(options.value("weightMin", 0.01f), options.value("weightMax", 5.0f));
    network->invertedLinks(options.value("relink", "normal") == "inverted");
    network->threshold(options.value("threshold", 0.0f));
    network->internalActivationFunction(activation == "relu" ? brain::ActivationFunction::ReLU : (activation == "linear" ? brain::ActivationFunction::Linear : brain::ActivationFunction::Sigmoid));
    network->rand(-options.value("range", 3.0f), options.value("range", 3.0f));
    if (options.value("direction", "random") == "forward") network->leftToRight(true);
    if (options.value("orphans", false)) network->allowOrphans();
    network->keepPaths(options.value("keepPaths", false));
    network->lowNewLinks(options.value("newLink", "mean") == "low");
    network->farLinks(options.value("farLinks", 0));
    if (options.value("pathInit", false)) network->pathInit(std::max<brain::UIType>(2, options.value("pathHops", 3)));
    network->minLinks(options.value("minLinks", 0));
    network->randomLinks(options.value("linkInit", "max") != "max");

    // the training
    if (options.value("adam", false)) network->adam(options.value("adamRate", 0.03f));
    else network->momentum(options.value("momentum", 0.0f));
    network->stepLimit(options.value("stepLimit", 0.0f));
    network->clampWeights(options.value("clamp", false));
    network->directOutput(options.value("direct", false));
    network->batch(options.value("batch", false));

    // the seed, then the neurons placed and linked
    brain::tools::rand(seed);
    network->size(options.value("input", 2), options.value("output", 1));
    if (options.value("weightInit", true)) network->weightInit();
    if (options.value("biasInit", true)) network->biasInit();
    return network;
}

void lab::BrainSession::round_(const nlohmann::json& train)
{
    const brain::FType rate = train.value("rate", 1.0f);
    const std::string coef = train.value("coef", "diff");

    // the decay is per round on the page: an update is a round with the mini-batch, a sample without it
    this->_network->rateDecay(train.value("decay", 0.0f) / static_cast<brain::FType>(this->_network->batch() ? 1 : std::max<brain::UIType>(1, this->_task.in.size())));

    // one score per output, by the score function and the score rate of the network: rate * (target - output), its sign, or the squared error
    this->_network->scoreRate(rate);
    this->_network->internalScoreFunction(coef == "sign" ? brain::ScoreFunction::Sign : (coef == "loss" ? brain::ScoreFunction::Loss : brain::ScoreFunction::Difference));
    for (brain::UIType i = 0; i < this->_task.in.size(); ++i) {
        if (this->_task.sequence) this->_network->sequence(this->_task.in[i]);
        else this->_network->inputs(this->_task.in[i].front());
        const brain::Values outputs = this->_network->apply();

        this->_network->backpropagation(this->_network->scores(outputs, this->_task.out[i]));
    }
    if (this->_network->batch()) this->_network->applyBatch();
    if (train.value("revive", false)) (void)this->_network->revive();
    ++this->_round;
}

nlohmann::json lab::BrainSession::train(const nlohmann::json& request)
{
    const brain::UIType rounds = request.value("rounds", 1);
    const double budget = request.value("budget", 0.0); // ms: the page gets a frame at least every <budget> ms whatever the size of the network
    const nlohmann::json train = request.value("train", nlohmann::json::object());
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    nlohmann::json errors = nlohmann::json::array();
    nlohmann::json tests = nlohmann::json::array();
    nlohmann::json out;

    for (brain::UIType r = 0; r < rounds; ++r) {
        if (r > 0 && budget > 0 && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() >= budget) break;
        this->round_(train);
        errors.push_back(error(*this->_network, this->_task.in, this->_task.clean, this->_task.sequence));
        tests.push_back(this->_task.testIn.empty() ? std::numeric_limits<brain::FType>::quiet_NaN() : error(*this->_network, this->_task.testIn, this->_task.testOut, this->_task.sequence));
    }
    out["round"] = this->_round;
    out["errors"] = errors;
    out["testErrors"] = tests;
    out["ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    out["solved"] = solved(*this->_network, this->_task.in, this->_task.clean, this->_task.regression, this->_task.sequence);
    out["snapshot"] = this->snapshot(request);
    return out;
}

nlohmann::json lab::BrainSession::snapshot(const nlohmann::json& request) const
{
    const std::vector<brain::RawBrainNeuron> neurons = this->_network->rawNeurons();
    const std::vector<brain::RawBrainLink> links = this->_network->rawLinks();
    const brain::UIType cap = this->_network->links();
    const std::pair<brain::IType, brain::IType> path = hops(*this->_network);
    nlohmann::json out;

    // the drawing: positions, biases, inward links <receiver, sender, weight>
    out["n"] = neurons.size();
    out["cap"] = cap;
    out["steps"] = this->_network->steps();
    out["input"] = this->_network->input();
    out["output"] = this->_network->output();
    out["seed"] = this->_seed;
    out["hops"] = path.first;
    out["hopsAll"] = path.second;
    out["gpu"] = this->_network->onGpu() ? this->_network->gpuName() : "";
    out["unsupported"] = this->_unsupported;
    out["pos"] = nlohmann::json::array();
    out["bias"] = nlohmann::json::array();
    for (const brain::RawBrainNeuron& neuron: neurons) {
        out["pos"].push_back({neuron.pos.x, neuron.pos.y});
        out["bias"].push_back(neuron.bias);
    }
    out["links"] = nlohmann::json::array();
    for (brain::UIType owner = 0; owner < neurons.size() && cap > 0; ++owner) {
        for (brain::UIType slot = 0; slot < cap; ++slot) {
            const brain::RawBrainLink& link = links[owner * cap + slot];
            if (link.inward) out["links"].push_back({owner, link.id, link.weight});
        }
    }

    // the outputs on the points of the page (samples, curve, map), the states of the sample shown at each step
    out["outputs"] = nlohmann::json::array();
    for (const brain::Dataset& point: this->_points)
        out["outputs"].push_back(run(*this->_network, point, this->_pointsSequence));
    if (request.contains("sample") && request["sample"].is_array() && !request["sample"].empty()) {
        brain::RawBrainNetwork& network = *this->_network;
        bool sequence = false;
        const std::vector<brain::Dataset> sample = samples(nlohmann::json::array({request["sample"]}), sequence);

        if (sequence) network.sequence(sample.front());
        else network.inputs(sample.front().front());
        (void)network.apply();
        out["states"] = network.states();
    }
    return out;
}

nlohmann::json lab::BrainSession::batch(const nlohmann::json& request)
{
    const lab::Task data = task(request["task"]);
    const brain::UIType rounds = request.value("rounds", 100);
    nlohmann::json results = nlohmann::json::array();

    // one network per seed, the same options and training as the page
    for (const nlohmann::json& seed: request["seeds"]) {
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        nlohmann::json options = request["options"];
        brain::IType solvedAt = -1;
        nlohmann::json result;

        options["seed"] = seed;
        lab::BrainSession session(options, data);
        for (brain::UIType r = 0; r < rounds; ++r) {
            session.round_(request.value("train", nlohmann::json::object()));
            if (solvedAt < 0 && solved(*session._network, data.in, data.clean, data.regression, data.sequence)) solvedAt = static_cast<brain::IType>(r) + 1;
        }
        result["seed"] = session._seed;
        result["error"] = error(*session._network, data.in, data.clean, data.sequence);
        result["solved"] = solved(*session._network, data.in, data.clean, data.regression, data.sequence);
        result["solvedAt"] = solvedAt;
        result["testError"] = data.testIn.empty() ? std::numeric_limits<brain::FType>::quiet_NaN() : error(*session._network, data.testIn, data.testOut, data.sequence);
        result["testSolved"] = !data.testIn.empty() && solved(*session._network, data.testIn, data.testOut, data.regression, data.sequence);
        result["hops"] = hops(*session._network).first;
        result["ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        results.push_back(result);
    }
    return results;
}

lab::Task lab::BrainSession::task(const nlohmann::json& data)
{
    lab::Task out;

    bool testSequence = false;

    out.in = samples(data.value("in", nlohmann::json::array()), out.sequence);
    out.out = data.value("out", brain::Dataset{});
    out.clean = data.value("clean", out.out);
    out.testIn = samples(data.value("testIn", nlohmann::json::array()), testSequence);
    out.testOut = data.value("testOut", brain::Dataset{});
    out.regression = data.value("kind", "class") == "reg";
    out.sequence = out.sequence || testSequence;
    return out;
}

std::vector<brain::Dataset> lab::BrainSession::samples(const nlohmann::json& list, bool& sequence)
{
    std::vector<brain::Dataset> out;

    // [[x, y], ...] = one step each, [[[x], [y]], ...] = the steps of each sample
    sequence = false;
    for (const nlohmann::json& sample: list) {
        if (!sample.empty() && sample.front().is_array()) {
            out.push_back(sample.get<brain::Dataset>());
            sequence = true;
        } else {
            out.push_back({sample.get<brain::Values>()});
        }
    }
    return out;
}

brain::FType lab::BrainSession::error(const brain::RawBrainNetwork& network, const std::vector<brain::Dataset>& inputs, const brain::Dataset& expected, const bool sequence)
{
    brain::FType total = 0;

    if (inputs.empty()) return std::numeric_limits<brain::FType>::quiet_NaN();
    for (brain::UIType i = 0; i < inputs.size(); ++i) {
        const brain::Values outputs = run(network, inputs[i], sequence);
        brain::FType sample = 0;

        for (brain::UIType k = 0; k < outputs.size(); ++k)
            sample += std::abs(outputs[k] - expected[i][k]);
        total += sample / static_cast<brain::FType>(outputs.size());
    }
    return total / static_cast<brain::FType>(inputs.size());
}

bool lab::BrainSession::solved(const brain::RawBrainNetwork& network, const std::vector<brain::Dataset>& inputs, const brain::Dataset& expected, const bool regression, const bool sequence)
{
    for (brain::UIType i = 0; i < inputs.size(); ++i) {
        const brain::Values outputs = run(network, inputs[i], sequence);

        for (brain::UIType k = 0; k < outputs.size(); ++k) {
            if (regression ? std::abs(outputs[k] - expected[i][k]) >= 0.1f : (outputs[k] >= 0.5f) != (expected[i][k] >= 0.5f)) return false;
        }
    }
    return !inputs.empty();
}

std::pair<brain::IType, brain::IType> lab::BrainSession::hops(const brain::RawBrainNetwork& network)
{
    const std::vector<brain::RawBrainLink> links = network.rawLinks();
    const brain::UIType count = network.neurons().second;
    const brain::UIType cap = network.links();
    const brain::UIType input = network.input();
    const brain::UIType output = network.output();
    std::pair<brain::IType, brain::IType> out = {-1, 0};

    // shortest numbers of links along the outward links (the senders): from all the inputs at once, then from each one
    const std::function<std::vector<brain::IType>(const std::vector<brain::UIType>&)> from = [&](const std::vector<brain::UIType>& sources) {
        std::vector<brain::IType> distance(count, -1);
        std::vector<brain::UIType> queue = sources;

        for (const brain::UIType s: sources) distance[s] = 0;
        for (brain::UIType head = 0; head < queue.size(); ++head) {
            for (brain::UIType slot = 0; slot < cap; ++slot) {
                const brain::RawBrainLink& link = links[queue[head] * cap + slot];

                if (link.inward || (link.id == 0 && link.weight == 0) || distance[link.id] >= 0) continue;
                distance[link.id] = distance[queue[head]] + 1;
                queue.push_back(link.id);
            }
        }
        return distance;
    };
    std::vector<brain::UIType> inputs;
    for (brain::UIType i = 0; i < input; ++i) inputs.push_back(i);
    const std::vector<brain::IType> all = from(inputs);
    brain::IType farthest = 0;
    for (brain::UIType o = count - output; o < count; ++o) {
        if (all[o] < 0) {farthest = -1; break;}
        farthest = std::max(farthest, all[o]);
    }
    out.first = farthest;
    for (brain::UIType i = 0; i < input && out.second >= 0; ++i) {
        const std::vector<brain::IType> distance = from({i});

        for (brain::UIType o = count - output; o < count; ++o) {
            if (distance[o] < 0) {out.second = -1; break;}
            out.second = std::max(out.second, distance[o]);
        }
    }
    return out;
}
