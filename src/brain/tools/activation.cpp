/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 19/10/2026 by @author Tsukini

File Name:
##  @file activation.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/tools/activation.hpp"
#include <algorithm>
#include <cmath>

_hot brain::FType brain::tools::sigmoid(const brain::FType value)
{
    return static_cast<brain::FType>(1.0) / (static_cast<brain::FType>(1.0) + std::exp(-value));
}

_hot brain::Values brain::tools::softmax(const brain::Values& values)
{
    brain::Values result(values.size());
    brain::FType total = 0.0;

    // nothing to compute
    if (values.empty()) _unlikely return result;

    // exponentials (the max is removed first to avoid an overflow)
    const brain::FType max = *std::max_element(values.begin(), values.end());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = std::exp(values[i] - max);
        total += result[i];
    }

    // normalize
    for (brain::FType& value: result) value /= total;
    return result;
}

_hot brain::FType brain::tools::linear(const brain::FType value)
{
    return value;
}

_hot brain::FType brain::tools::relu(const brain::FType value)
{
    return value > 0 ? value : static_cast<brain::FType>(0.0);
}
