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
##  @file MultiHook.hpp

File Description:
##  Training hook calling several hooks at once (forwards every event to each hook)
\**************************************************************/

#ifndef MULTIHOOK_H
    #define MULTIHOOK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::StopReason
    #include "../NetworkType.hpp"      // brain::UIType, brain::TrainOptions, brain::TrainState
    #include "ITrainHook.hpp"          // brain::ITrainHook, brain::NeuralNetwork
    #include "ATrainHook.hpp"          // brain::ATrainHook
    #include <utility>                 // std::forward
    #include <cstdint>                 // std::uint32_t
    #include <memory>                  // std::shared_ptr, std::make_shared
    #include <vector>                  // std::vector

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class MultiHook: public brain::ATrainHook {
    private:
        /* storage */
        std::vector<std::shared_ptr<brain::ITrainHook>> _hooks; // shared with the caller, called in the order of add()

    public:
        // ---------- Pre-Function -------- //
        /* events */
        std::uint32_t events(void) const override; // the events of every hook (or of each mask)

        /* training */
        void onStart(const brain::TrainOptions& options, const brain::UIType training, const brain::UIType validation) override;
        void onStop(const brain::TrainState& state, const brain::StopReason reason) override;

        /* loop (an event only reaches the hooks that ask for it) */
        bool onEpoch(const brain::TrainState& state) override; // false when one of the hooks returns false (every hook is still called)
        void onBatch(const brain::TrainState& state) override;
        void onSample(const brain::TrainState& state) override;

        /* best network */
        void onBest(const brain::TrainState& state, const brain::NeuralNetwork& nn) override;
        void onRollback(const brain::TrainState& state) override;

        // ------------ Function ---------- //
        /* setter */
        _cold inline void add(std::shared_ptr<brain::ITrainHook> hook) {this->_hooks.push_back(hook);};
        template<class T, class... Args>
        _cold inline void add(Args&&... args) {this->add(std::make_shared<T>(std::forward<Args>(args)...));}; // builds it in place (make_shared<T>)
        _cold inline void clear(void)         {this->_hooks.clear();}; // remove every stored ptr (a hook is freed when nothing else holds it)

        /* getter */
        _cold _nodiscard inline const std::vector<std::shared_ptr<brain::ITrainHook>>& get(void) const {return this->_hooks;}; // the stored ptr, in the order of add()

        // ------------ Operator ---------- //
        MultiHook& operator=(const MultiHook& other) = delete;
        MultiHook& operator=(MultiHook&& other) = delete;

        // ---------- Constructor --------- //
        MultiHook(const std::vector<std::shared_ptr<brain::ITrainHook>>& hooks = {}): _hooks{hooks} { /* Nothing */ };

        MultiHook(const MultiHook& other) = delete;
        MultiHook(MultiHook&& other) = delete;

        // ----------- Destructor --------- //
        ~MultiHook() = default;
};

} // namespace end
#endif /* MULTIHOOK_H */
