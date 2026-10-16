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
##  @file Random.cpp

File Description:
##  Unit tests of the random tools
\**************************************************************/

#include "brain/tools/random.hpp"
#include <gtest/gtest.h>
#include <type_traits>
#include <random>

/* ------------------------------- random ------------------------------- */
// The global generator is the one every randomness call uses
static_assert(std::is_same_v<decltype(brain::tools::generator), std::mt19937_64>);

TEST(Random, CallsDoNotThrow) {
    EXPECT_NO_THROW(brain::tools::rand(0));
    EXPECT_NO_THROW(brain::tools::rand(42));
    EXPECT_NO_THROW((void)brain::tools::rand_unsigned_integer());
    EXPECT_NO_THROW((void)brain::tools::rand_floating());
}
