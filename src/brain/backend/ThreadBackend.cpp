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
##  @file ThreadBackend.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/backend/ThreadBackend.hpp"
#include <functional>

brain::ThreadBackend::ThreadBackend(const brain::UIType threads)
{
    /* Nothing */
}

_cold void brain::ThreadBackend::init(void)
{
    /* Nothing */
}

_cold void brain::ThreadBackend::release(void)
{
    /* Nothing */
}

_hot void brain::ThreadBackend::sync(void)
{
    /* Nothing */
}

_hot void brain::ThreadBackend::parallelFor(const brain::UIType count, const std::function<void(const brain::UIType begin, const brain::UIType end, const brain::UIType worker)>& fn)
{
    /* Nothing */
}

_cold void brain::ThreadBackend::threads(const brain::UIType threads)
{
    /* Nothing */
}
