/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 12/11/2026 by @author Tsukini

File Name:
##  @file ATrainHook.hpp

File Description:
##  Abstract training hook: every method defined but empty (fallthrough), a child only overrides what it needs
\**************************************************************/

#ifndef ATRAINHOOK_H
    #define ATRAINHOOK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _hot, _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::StopReason, brain::events
    #include "../NetworkType.hpp"      // brain::UIType, brain::TrainOptions, brain::TrainState
    #include "ITrainHook.hpp"          // brain::ITrainHook, brain::NeuralNetwork
    #include <cstdint>                 // std::uint32_t

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ATrainHook: public brain::ITrainHook {
    public:
        // ------------ Function ---------- //
        /* events */
        _cold _nodiscard std::uint32_t events(void) const override {return brain::events::DEFAULT;}; // every event except batch / sample (a child adds them when it overrides them)

        /* training */
        _cold void onStart(const brain::TrainOptions& options, const brain::UIType training, const brain::UIType validation) override { /* Nothing */ };
        _cold void onStop(const brain::TrainState& state, const brain::StopReason reason) override                                    { /* Nothing */ };

        /* loop */
        _hot bool onEpoch(const brain::TrainState& state) override  {return true;}; // never stops the training
        _hot void onBatch(const brain::TrainState& state) override  { /* Nothing */ };
        _hot void onSample(const brain::TrainState& state) override { /* Nothing */ };

        /* best network */
        _cold void onBest(const brain::TrainState& state, const brain::NeuralNetwork& nn) override { /* Nothing */ };
        _cold void onRollback(const brain::TrainState& state) override                             { /* Nothing */ };

        // ------------ Operator ---------- //
        ATrainHook& operator=(const ATrainHook& other) = delete;
        ATrainHook& operator=(ATrainHook&& other) = delete;

        // ---------- Constructor --------- //
        ATrainHook() = default;
        ATrainHook(const ATrainHook& other) = delete;
        ATrainHook(ATrainHook&& other) = delete;

        // ----------- Destructor --------- //
        ~ATrainHook() = default;
};

} // namespace end
#endif /* ATRAINHOOK_H */
