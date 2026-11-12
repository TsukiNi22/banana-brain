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
##  @file ITrainHook.hpp

File Description:
##  Interface of the training hooks: follow-up of a training (progress, logs, custom saves, stop)
\**************************************************************/

#ifndef ITRAINHOOK_H
    #define ITRAINHOOK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkDefine.hpp"   // brain::StopReason, brain::TrainEvent
    #include "../NetworkType.hpp"     // brain::UIType, brain::TrainOptions, brain::TrainState
    #include <cstdint>                // std::uint32_t

namespace brain { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

class NeuralNetwork; // NeuralNetwork.hpp (onBest)

//----------------------------------------------------------------//
/* CLASS */

// Every method pure virtual (ATrainHook for the empty defaults)
class ITrainHook {
    public:
        // ---------- Pre-Function -------- //
        /* events */
        virtual std::uint32_t events(void) const = 0; // brain::TrainEvent mask of the events called by the train (the others are skipped: no virtual call)

        /* training */
        virtual void onStart(const brain::TrainOptions& options, const brain::UIType training, const brain::UIType validation) = 0; // sizes of the sets
        virtual void onStop(const brain::TrainState& state, const brain::StopReason reason) = 0;

        /* loop */
        virtual bool onEpoch(const brain::TrainState& state) = 0; // false = stop (StopReason::Hook)
        virtual void onBatch(const brain::TrainState& state) = 0; // ON_BATCH only (can cost a lot)
        virtual void onSample(const brain::TrainState& state) = 0; // ON_SAMPLE only (can cost a lot)

        /* best network */
        virtual void onBest(const brain::TrainState& state, const brain::NeuralNetwork& nn) = 0; // new best network (custom save)
        virtual void onRollback(const brain::TrainState& state) = 0; // back on the best snapshot

        // ------------ Operator ---------- //
        ITrainHook& operator=(const ITrainHook& other) = delete;
        ITrainHook& operator=(ITrainHook&& other) = delete;

        // ---------- Constructor --------- //
        ITrainHook() = default;
        ITrainHook(const ITrainHook& other) = delete;
        ITrainHook(ITrainHook&& other) = delete;

        // ----------- Destructor --------- //
        virtual ~ITrainHook() = default;
};

} // namespace end
#endif /* ITRAINHOOK_H */
