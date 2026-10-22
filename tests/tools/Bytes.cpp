/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 17/10/2026 by @author Tsukini

File Name:
##  @file Bytes.cpp

File Description:
##  Unit tests of the brain neuron tools
\**************************************************************/

#include "brain/tools/bytes.hpp"
#include <gtest/gtest.h>
#include <cstdint>

/* ------------------------------- bytes ------------------------------- */
TEST(Bytes, WrittenLittleEndian) {
    brain::Bytes data;

    brain::tools::put<std::uint32_t>(data, 0x01020304u);
    EXPECT_EQ(data, (brain::Bytes{0x04, 0x03, 0x02, 0x01}));
}

TEST(Bytes, RoundTripOfEveryKind) {
    brain::Bytes data;
    brain::UIType offset = 0;

    brain::tools::put<std::uint64_t>(data, 123456789012345ull);
    brain::tools::put<float>(data, -1.5f);
    brain::tools::put<bool>(data, true);
    brain::tools::put<std::uint8_t>(data, 0xAB);
    EXPECT_EQ(brain::tools::get<std::uint64_t>(data, offset), 123456789012345ull);
    EXPECT_FLOAT_EQ(brain::tools::get<float>(data, offset), -1.5f);
    EXPECT_TRUE(brain::tools::get<bool>(data, offset));
    EXPECT_EQ(brain::tools::get<std::uint8_t>(data, offset), 0xAB);
    EXPECT_EQ(offset, data.size());
}

TEST(Bytes, ReadingTooFarThrows) {
    brain::Bytes data = {1, 2, 3};
    brain::UIType offset = 0;

    EXPECT_THROW((void)brain::tools::get<std::uint32_t>(data, offset), utils::exception::IException);
    EXPECT_EQ(offset, 0u); // not moved on an error
    offset = 99;
    EXPECT_THROW((void)brain::tools::get<std::uint8_t>(data, offset), utils::exception::IException);
}
