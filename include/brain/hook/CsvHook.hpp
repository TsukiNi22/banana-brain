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
##  @file CsvHook.hpp

File Description:
##  Training hook writing the history in a csv file on the go (no history kept in the ram)
\**************************************************************/

#ifndef CSVHOOK_H
    #define CSVHOOK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #include "../NetworkDefine.hpp"   // brain::StopReason
    #include "../NetworkType.hpp"     // brain::UIType, brain::TrainOptions, brain::TrainState
    #include "ATrainHook.hpp"         // brain::ATrainHook
    #include <filesystem>             // std::filesystem::path
    #include <fstream>                // std::ofstream

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class CsvHook: public brain::ATrainHook {
    private:
        std::filesystem::path _path; // the csv file (replaced at each training)
        std::ofstream _file; // opened by onStart, closed by onStop

    public:
        // ---------- Pre-Function -------- //
        /* training */
        void onStart(const brain::TrainOptions& options, const brain::UIType training, const brain::UIType validation) override; // open the file and write the header line
        void onStop(const brain::TrainState& state, const brain::StopReason reason) override; // close the file

        /* loop */
        bool onEpoch(const brain::TrainState& state) override; // one line: epoch, loss, validation loss, learning rate, seconds

        // ------------ Operator ---------- //
        CsvHook& operator=(const CsvHook& other) = delete;
        CsvHook& operator=(CsvHook&& other) = delete;

        // ---------- Constructor --------- //
        CsvHook(const std::filesystem::path& path): _path{path} { /* Nothing */ };

        CsvHook(const CsvHook& other) = delete;
        CsvHook(CsvHook&& other) = delete;

        // ----------- Destructor --------- //
        ~CsvHook() = default;
};

} // namespace end
#endif /* CSVHOOK_H */
