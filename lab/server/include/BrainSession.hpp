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
##  @file BrainSession.hpp

File Description:
##  A raw brain network of the lab, built from the options of the page,
##  trained on its task and drawn from its snapshot
\**************************************************************/

#ifndef BRAINSESSION_H
    #define BRAINSESSION_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "brain/network/RawBrainNetwork.hpp"    // brain::RawBrainNetwork
    #include "brain/NetworkType.hpp"                // brain::UIType, brain::IType, brain::FType, brain::Values, brain::Dataset
    #include <nlohmann/json.hpp>                    // nlohmann::json
    #include <memory>                               // std::unique_ptr
    #include <string>                               // std::string
    #include <vector>                               // std::vector

namespace lab { // namespace start
//----------------------------------------------------------------//
/* STRUCT */

// The samples of a task, as the page made them: a sample is the inputs of each of its time steps (one step unless the task is a sequence)
struct Task {
    std::vector<brain::Dataset> in; // trained inputs
    brain::Dataset out; // trained targets (with the noise)
    brain::Dataset clean; // trained targets without the noise (measured)
    std::vector<brain::Dataset> testIn; // inputs kept for the test
    brain::Dataset testOut; // targets of the test
    bool regression = false; // true: good within 0.1 of the target | false: on the right side of 0.5
    bool sequence = false; // true: the steps of a sample come one per time step (RawBrainNetwork::sequence) | false: its one step at every time step
};

//----------------------------------------------------------------//
/* CLASS */

// One brain network of the page: options, task, training and snapshot
class BrainSession {
    private:
        nlohmann::json _options; // the options of the page (see build_)
        lab::Task _task;
        std::unique_ptr<brain::RawBrainNetwork> _network;
        brain::IType _seed = 1; // seed used (after the filter of the paths)
        brain::UIType _round = 0;
        std::vector<brain::Dataset> _points; // the points of the page whose outputs the snapshot gives (sent once, kept)
        bool _pointsSequence = false; // the points are sequences
        std::vector<std::string> _unsupported; // options of the page the library has not (the page shows them)

        // ---------- Pre-Function -------- //
        static std::unique_ptr<brain::RawBrainNetwork> build_(const nlohmann::json& options, const brain::IType seed, std::vector<std::string>& unsupported); // a network from the options and the seed
        void round_(const nlohmann::json& train); // one round: a score per sample, the mini-batch applied once, revive at the end

    public:
        // ---------- Pre-Function -------- //
        /* training */
        nlohmann::json train(const nlohmann::json& request); // up to <rounds> rounds, stopped after <budget> ms (0 = all of them), then the errors of each round and the snapshot
        nlohmann::json snapshot(const nlohmann::json& request) const; // positions, biases, inward links, states of one sample, outputs on the points kept

        /* batch */
        static nlohmann::json batch(const nlohmann::json& request); // many networks of the same options, one result each

        /* tools */
        static lab::Task task(const nlohmann::json& data); // the samples sent by the page
        static std::vector<brain::Dataset> samples(const nlohmann::json& list, bool& sequence); // the samples of a list: a list of steps each, or one step (sequence set when one has several levels)
        static brain::FType error(const brain::RawBrainNetwork& network, const std::vector<brain::Dataset>& inputs, const brain::Dataset& expected, const bool sequence); // mean of the mean absolute error of each sample
        static bool solved(const brain::RawBrainNetwork& network, const std::vector<brain::Dataset>& inputs, const brain::Dataset& expected, const bool regression, const bool sequence); // every sample good
        static std::pair<brain::IType, brain::IType> hops(const brain::RawBrainNetwork& network); // <nearest input -> every output, every input -> every output> in links (-1: no path)

        // ------------ Function ---------- //
        /* setter */
        _cold inline void points(const nlohmann::json& request) {if (request.contains("points") && request["points"].is_array()) this->_points = samples(request["points"], this->_pointsSequence);}; // keep the points of the request when it has some

        /* tools */
        _hot _nodiscard static inline brain::Values run(const brain::RawBrainNetwork& network, const brain::Dataset& sample, const bool sequence) {return sequence ? network.applySequence(sample) : network.apply(sample.front());}; // the outputs of one sample, nothing kept

        /* getter */
        _cold _nodiscard inline brain::IType seed(void) const                           {return this->_seed;}; // seed of the network (after the filter)
        _cold _nodiscard inline const std::vector<std::string>& unsupported(void) const {return this->_unsupported;}; // options ignored by the library

        // ---------- Constructor --------- //
        BrainSession(const nlohmann::json& options, lab::Task task); // the seed of the options, then the next ones until the filter of the paths is met
};

} // namespace end
#endif /* BRAINSESSION_H */
