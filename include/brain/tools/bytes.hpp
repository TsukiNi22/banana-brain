/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 22/10/2026 by @author Tsukini

File Name:
##  @file bytes.hpp

File Description:
##  Tools of the serialization: values written and read as little endian bytes
\**************************************************************/

#ifndef BYTES_H
    #define BYTES_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #define _Exception
    #include <utils/utils.hpp>      // _hot, _nodiscard, _unlikely, utils::exception::ErrorException
    #include "../NetworkType.hpp"   // brain::UIType, brain::Bytes
    #include <type_traits>          // std::is_arithmetic_v
    #include <algorithm>            // std::reverse
    #include <cstdint>              // std::uint8_t
    #include <string>               // std::to_string
    #include <array>                // std::array
    #include <bit>                  // std::bit_cast, std::endian

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

/* serialization (little endian whatever the machine) */
template<typename T> requires (std::is_arithmetic_v<T>)
_hot inline void put(brain::Bytes& out, const T value) // append the bytes of the value
{
    std::array<std::uint8_t, sizeof(T)> bytes = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);

    if constexpr (std::endian::native == std::endian::big) std::reverse(bytes.begin(), bytes.end());
    out.insert(out.end(), bytes.begin(), bytes.end());
}

template<typename T> requires (std::is_arithmetic_v<T>)
_hot _nodiscard inline T get(const brain::Bytes& data, brain::UIType& offset) // read the value at the offset then move the offset after it (OutOfBounds when the data are too short)
{
    std::array<std::uint8_t, sizeof(T)> bytes;

    if (offset > data.size() || data.size() - offset < sizeof(T)) _unlikely {
        throw utils::exception::ErrorException(utils::exception::InternalCode::OutOfBounds, "truncated data: " + std::to_string(sizeof(T)) + " bytes expected at " + std::to_string(offset) + " of " + std::to_string(data.size()));
    }
    std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(offset), sizeof(T), bytes.begin());
    offset += sizeof(T);
    if constexpr (std::endian::native == std::endian::big) std::reverse(bytes.begin(), bytes.end());
    if constexpr (std::is_same_v<T, bool>) return bytes[0] != 0; // any other byte than 0 / 1 is not a valid bool for a bit_cast
    else return std::bit_cast<T>(bytes);
}

_hot _nodiscard inline bool fits(const brain::Bytes& data, const brain::UIType offset, const brain::UIType count, const brain::UIType size) // do count elements of size bytes fit in the data after the offset (no overflow)
{
    return offset <= data.size() && (size == 0 || count <= (data.size() - offset) / size);
}

} // namespace end
#endif /* BYTES_H */
