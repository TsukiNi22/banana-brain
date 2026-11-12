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
##  @file convertor.cpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#define _Attribute
#include <utils/utils.hpp>
#include "brain/tools/convertor.hpp"

_hot brain::Values brain::tools::int_to_bits(const brain::UIType value, const brain::UIType n)
{
    /* Nothing */
    return {};
}

_hot brain::UIType brain::tools::bits_to_int(const brain::Values& values, const brain::FType threshold)
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::tools::bytes_to_bits(const brain::Bytes& data, const brain::UIType n)
{
    /* Nothing */
    return {};
}

_hot brain::Bytes brain::tools::bits_to_bytes(const brain::Values& values, const brain::FType threshold)
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::tools::image_to_input(const brain::Bytes& pixels, const brain::UIType width, const brain::UIType height, const brain::UIType channels, const bool gray)
{
    /* Nothing */
    return {};
}

_hot brain::Values brain::tools::image_to_input(const brain::Image& image, const bool gray)
{
    /* Nothing */
    return {};
}

_hot brain::Image brain::tools::input_to_image(const brain::Values& values, const brain::UIType width, const brain::UIType height, const brain::UIType channels)
{
    /* Nothing */
    return {};
}
