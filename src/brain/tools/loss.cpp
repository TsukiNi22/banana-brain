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
##  @file loss.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/tools/loss.hpp"
#include <algorithm>
#include <cmath>

_hot brain::FType brain::tools::mse(const brain::Value output, const brain::Value expected)
{
    const brain::FType diff = output - expected;
    return diff * diff;
}

_hot brain::FType brain::tools::cross_entropy(const brain::Value output, const brain::Value expected)
{
    constexpr brain::FType epsilon = static_cast<brain::FType>(1e-7);
    const brain::FType clamped = std::clamp(output, epsilon, static_cast<brain::FType>(1.0) - epsilon); // avoid log(0)
    return -(expected * std::log(clamped) + (static_cast<brain::FType>(1.0) - expected) * std::log(static_cast<brain::FType>(1.0) - clamped));
}

_hot brain::FType brain::tools::mae(const brain::Value output, const brain::Value expected)
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::tools::huber(const brain::Value output, const brain::Value expected, const brain::FType delta)
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::tools::mse_d(const brain::Value output, const brain::Value expected)
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::tools::cross_entropy_d(const brain::Value output, const brain::Value expected)
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::tools::mae_d(const brain::Value output, const brain::Value expected)
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::tools::huber_d(const brain::Value output, const brain::Value expected, const brain::FType delta)
{
    /* Nothing */
    return {};
}

_hot brain::FType brain::tools::mean(const brain::LossFn& fn, const brain::Values& outputs, const brain::Values& expected)
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::tools::each(const brain::LossFn& fn, const brain::Values& outputs, const brain::Values& expected)
{
    /* Nothing */
    return {};
}
