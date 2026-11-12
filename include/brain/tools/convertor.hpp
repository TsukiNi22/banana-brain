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
##  @file convertor.hpp

File Description:
##  Conversions of the data of a network: bytes (serialization), text, bits and images to / from the values
\**************************************************************/

#ifndef CONVERTOR_H
    #define CONVERTOR_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _Exception
    #include <utils/utils.hpp>         // _hot, _cold, _nodiscard, _unlikely, utils::exception::ErrorException
    #include "../NetworkDefine.hpp"    // brain::Endian, brain::TextMode, BRAIN_DEFAULT_ENDIAN, BRAIN_DEFAULT_BITS_THRESHOLD
    #include "../NetworkType.hpp"      // brain::UIType, brain::FType, brain::Value, brain::Values, brain::Bytes, brain::Image
    #include <type_traits>             // std::is_arithmetic_v, std::is_same_v
    #include <algorithm>               // std::reverse, std::copy_n, std::clamp
    #include <cstdint>                 // std::uint8_t
    #include <cstring>                 // std::memcpy
    #include <string>                  // std::string, std::to_string
    #include <array>                   // std::array
    #include <cmath>                   // std::isfinite, std::lround
    #include <bit>                     // std::bit_cast, std::endian

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* bytes (the order of BRAIN_DEFAULT_ENDIAN whatever the machine: the save files stay the same between machines) */
// The type first: get can't deduce it, the endian keeps its default
template<typename T, brain::Endian endian = BRAIN_DEFAULT_ENDIAN> requires (std::is_arithmetic_v<T>)
_hot inline void put(brain::Bytes& out, const T value) // append the bytes of the value
{
    std::array<std::uint8_t, sizeof(T)> bytes = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);

    if constexpr ((endian == brain::Endian::Big) != (std::endian::native == std::endian::big)) std::reverse(bytes.begin(), bytes.end());
    out.insert(out.end(), bytes.begin(), bytes.end());
}

template<typename T, brain::Endian endian = BRAIN_DEFAULT_ENDIAN> requires (std::is_arithmetic_v<T>)
_hot _nodiscard inline T get(const brain::Bytes& data, brain::UIType& offset) // read the value at the offset then move the offset after it (OutOfBounds when the data are too short)
{
    std::array<std::uint8_t, sizeof(T)> bytes;

    if (offset > data.size() || data.size() - offset < sizeof(T)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfBounds, "truncated data: " + std::to_string(sizeof(T)) + " bytes expected at " + std::to_string(offset) + " of " + std::to_string(data.size()));
    }
    std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(offset), sizeof(T), bytes.begin());
    offset += sizeof(T);
    if constexpr ((endian == brain::Endian::Big) != (std::endian::native == std::endian::big)) std::reverse(bytes.begin(), bytes.end());
    if constexpr (std::is_same_v<T, bool>) return bytes[0] != 0; // any other byte than 0 / 1 is not a valid bool for a bit_cast
    else return std::bit_cast<T>(bytes);
}

_hot _nodiscard inline bool fits(const brain::Bytes& data, const brain::UIType offset, const brain::UIType count, const brain::UIType size) // do count elements of size bytes fit in the data after the offset (no overflow)
{
    return offset <= data.size() && (size == 0 || count <= (data.size() - offset) / size);
}

/* text */
template<brain::TextMode mode = brain::TextMode::Normal>
_hot _nodiscard inline brain::Values text_to_input(const std::string& text) // Normal: one value per character (byte / 255), Compact: the bytes copied as is in the values (4 per value, can be not what you want)
{
    if constexpr (mode == brain::TextMode::Normal) {
        brain::Values input(text.size());

        for (std::size_t i = 0; i < text.size(); ++i)
            input[i] = static_cast<brain::FType>(static_cast<unsigned char>(text[i])) / static_cast<brain::FType>(255.0);
        return input;
    } else {
        brain::Values input((text.size() + sizeof(brain::FType) - 1) / sizeof(brain::FType));

        std::memcpy(input.data(), text.data(), text.size());
        return input;
    }
}

template<brain::TextMode mode = brain::TextMode::Normal, bool round = false>
_hot _nodiscard inline std::string input_to_text(const brain::Values& input) // doesn't ensure the validity of the output, round: Compact only, removes the '\0' of the end (the padding of the last value)
{
    if constexpr (mode == brain::TextMode::Normal) {
        std::string text(input.size(), '\0');

        for (std::size_t i = 0; i < input.size(); ++i)
            text[i] = std::isfinite(input[i]) ? static_cast<char>(static_cast<unsigned char>(std::clamp<long>(std::lround(input[i] * static_cast<brain::FType>(255.0)), 0, 255))) : '\0';
        return text;
    } else {
        std::string text(input.size() * sizeof(brain::FType), '\0');

        std::memcpy(text.data(), input.data(), input.size() * sizeof(brain::FType));
        if constexpr (round) {
            while (!text.empty() && text.back() == '\0') text.pop_back();
        }
        return text;
    }
}

/* bits */
_hot _nodiscard brain::Values int_to_bits(const brain::UIType value, const brain::UIType n); // n values at 0 / 1, the lowest bit first (n <= the bits of UIType)
_hot _nodiscard brain::UIType bits_to_int(const brain::Values& values, const brain::FType threshold = BRAIN_DEFAULT_BITS_THRESHOLD); // a value over the threshold is a 1
_hot _nodiscard brain::Values bytes_to_bits(const brain::Bytes& data, const brain::UIType n); // n values at 0 / 1 from the first n bits of data (any n, not a multiple of 8)
_hot _nodiscard brain::Bytes bits_to_bytes(const brain::Values& values, const brain::FType threshold = BRAIN_DEFAULT_BITS_THRESHOLD); // the last byte is padded with 0

/* image */
_hot _nodiscard brain::Values image_to_input(const brain::Bytes& pixels, const brain::UIType width, const brain::UIType height, const brain::UIType channels, const bool gray = false); // 0 - 1 per channel, gray -> 1 channel
_hot _nodiscard brain::Values image_to_input(const brain::Image& image, const bool gray = false); // same as above, the shape taken from the image
_hot _nodiscard brain::Image input_to_image(const brain::Values& values, const brain::UIType width, const brain::UIType height, const brain::UIType channels); // clamped to 0 - 1, values.size() == width * height * channels (channels = 1 after a gray input)

} // namespace end
#endif /* CONVERTOR_H */
