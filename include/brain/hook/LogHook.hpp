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
##  @file LogHook.hpp

File Description:
##  Training hook writing one line per epoch (loss, validation, saved) on an output stream
\**************************************************************/

#ifndef LOGHOOK_H
    #define LOGHOOK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkDefine.hpp"   // brain::StopReason
    #include "../NetworkType.hpp"     // brain::TrainState
    #include "ATrainHook.hpp"         // brain::ATrainHook, brain::NeuralNetwork
    #include <iostream>               // std::cout
    #include <ostream>                // std::ostream

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class LogHook: public brain::ATrainHook {
    private:
        std::ostream& _os; // where the lines are written
        bool _saved = false; // a new best network since the last line ("(saved)" on the next one)

    public:
        // ---------- Pre-Function -------- //
        /* training */
        void onStop(const brain::TrainState& state, const brain::StopReason reason) override; // last line with the reason

        /* loop */
        bool onEpoch(const brain::TrainState& state) override; // epoch, loss, validation loss, (saved)

        /* best network */
        void onBest(const brain::TrainState& state, const brain::NeuralNetwork& nn) override; // marks the next line as saved
        void onRollback(const brain::TrainState& state) override; // line of the rollback

        // ------------ Operator ---------- //
        LogHook& operator=(const LogHook& other) = delete;
        LogHook& operator=(LogHook&& other) = delete;

        // ---------- Constructor --------- //
        LogHook(std::ostream& os = std::cout): _os{os} { /* Nothing */ };

        LogHook(const LogHook& other) = delete;
        LogHook(LogHook&& other) = delete;

        // ----------- Destructor --------- //
        ~LogHook() = default;
};

} // namespace end
#endif /* LOGHOOK_H */
