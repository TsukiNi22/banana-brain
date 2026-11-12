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
##  @file ProgressHook.hpp

File Description:
##  Training hook drawing a progress bar with the epochs and the loss on an output stream
\**************************************************************/

#ifndef PROGRESSHOOK_H
    #define PROGRESSHOOK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkDefine.hpp"   // brain::StopReason
    #include "../NetworkType.hpp"     // brain::UIType, brain::TrainOptions, brain::TrainState
    #include "ATrainHook.hpp"         // brain::ATrainHook
    #include <iostream>               // std::cerr
    #include <ostream>                // std::ostream
    #include <cstdint>                // std::uint32_t

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ProgressHook: public brain::ATrainHook {
    private:
        std::ostream& _os; // where the bar is drawn
        brain::UIType _epochs = 0; // epochs of the training (0 = no limit: no bar, only the counters)
        brain::UIType _samples = 0; // samples of an epoch (the bar of the current epoch)

    public:
        // ---------- Pre-Function -------- //
        /* events */
        std::uint32_t events(void) const override; // the default ones + ON_BATCH (the bar moves on each batch)

        /* training */
        void onStart(const brain::TrainOptions& options, const brain::UIType training, const brain::UIType validation) override; // keep the sizes of the training
        void onStop(const brain::TrainState& state, const brain::StopReason reason) override; // last line with the reason

        /* loop */
        bool onEpoch(const brain::TrainState& state) override; // epoch and loss
        void onBatch(const brain::TrainState& state) override; // bar of the current epoch

        // ------------ Operator ---------- //
        ProgressHook& operator=(const ProgressHook& other) = delete;
        ProgressHook& operator=(ProgressHook&& other) = delete;

        // ---------- Constructor --------- //
        ProgressHook(std::ostream& os = std::cerr): _os{os} { /* Nothing */ };

        ProgressHook(const ProgressHook& other) = delete;
        ProgressHook(ProgressHook&& other) = delete;

        // ----------- Destructor --------- //
        ~ProgressHook() = default;
};

} // namespace end
#endif /* PROGRESSHOOK_H */
