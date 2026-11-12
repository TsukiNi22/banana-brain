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
##  @file Adam.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/optimizer/Adam.hpp"
#include <memory>

_cold void brain::Adam::init(const brain::UIType weights)
{
    /* Nothing */
}

_cold void brain::Adam::reset(void)
{
    /* Nothing */
}

_hot brain::FType brain::Adam::step(const brain::UIType index, const brain::FType gradient, const brain::FType learning_rate)
{
    /* Nothing */
    return {};
}

_cold brain::Bytes brain::Adam::serialize(void) const
{
    /* Nothing */
    return {};
}

_cold void brain::Adam::deserialize(const brain::Bytes& data)
{
    /* Nothing */
}

_cold std::unique_ptr<brain::IOptimizer> brain::Adam::clone(void) const
{
    /* Nothing */
    return {};
}
