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
##  @file MultiHook.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/hook/MultiHook.hpp"
#include <cstdint>

_cold std::uint32_t brain::MultiHook::events(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::MultiHook::onStart(const brain::TrainOptions& options, const brain::UIType training, const brain::UIType validation)
{
    /* Nothing */
}

_cold void brain::MultiHook::onStop(const brain::TrainState& state, const brain::StopReason reason)
{
    /* Nothing */
}

_hot bool brain::MultiHook::onEpoch(const brain::TrainState& state)
{
    /* Nothing */
    return true;
}

_hot void brain::MultiHook::onBatch(const brain::TrainState& state)
{
    /* Nothing */
}

_hot void brain::MultiHook::onSample(const brain::TrainState& state)
{
    /* Nothing */
}

_cold void brain::MultiHook::onBest(const brain::TrainState& state, const brain::NeuralNetwork& nn)
{
    /* Nothing */
}

_cold void brain::MultiHook::onRollback(const brain::TrainState& state)
{
    /* Nothing */
}
