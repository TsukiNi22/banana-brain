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
##  @file ProgressHook.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/hook/ProgressHook.hpp"
#include <cstdint>

_cold std::uint32_t brain::ProgressHook::events(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::ProgressHook::onStart(const brain::TrainOptions& options, const brain::UIType training, const brain::UIType validation)
{
    /* Nothing */
}

_cold void brain::ProgressHook::onStop(const brain::TrainState& state, const brain::StopReason reason)
{
    /* Nothing */
}

_hot bool brain::ProgressHook::onEpoch(const brain::TrainState& state)
{
    /* Nothing */
    return true;
}

_hot void brain::ProgressHook::onBatch(const brain::TrainState& state)
{
    /* Nothing */
}
