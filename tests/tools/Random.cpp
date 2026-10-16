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
// The generator is the one every randomness call uses
static_assert(std::is_same_v<decltype(brain::tools::generator), std::mt19937_64>);

// Default scope: BRAIN_SEED_GLOBAL is defined by the header when nothing is asked
#ifndef BRAIN_SEED_GLOBAL
    #error "BRAIN_SEED_GLOBAL must be the default generator scope"
#endif
#ifdef BRAIN_SEED_LOCAL
    #error "BRAIN_SEED_LOCAL must not be defined by default"
#endif

const std::mt19937_64* generator_address_other_unit(void); // tools/RandomScope.cpp

TEST(Random, GlobalGeneratorByDefault) {
    // One generator shared by the whole program: the same object in every translation unit
    EXPECT_EQ(&brain::tools::generator, generator_address_other_unit());
}

TEST(Random, CallsDoNotThrow) {
    EXPECT_NO_THROW(brain::tools::rand(0));
    EXPECT_NO_THROW(brain::tools::rand(42));
    EXPECT_NO_THROW((void)brain::tools::rand_unsigned_integer());
    EXPECT_NO_THROW((void)brain::tools::rand_floating());
}
