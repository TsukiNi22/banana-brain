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
##  @file random.hpp

File Description:
##  Global random generator of the library and its randomness calls (change it here to change it everywhere)
\**************************************************************/

#ifndef RANDOM_H
    #define RANDOM_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _hot, _cold, _nodiscard
    #include "../NetworkType.hpp"   // brain::IType, brain::UIType, brain::FType
    #include <random>               // std::mt19937_64

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* GLOBAL */

inline std::mt19937_64 generator; // generator used by every randomness call of the library

//----------------------------------------------------------------//
/* PROTOTYPE */

/* randomness */
_cold inline void rand(const brain::IType seed) // setup the seed of the generator (0 = random seed)
{
    /* Nothing */
};

_hot _nodiscard inline brain::UIType rand_unsigned_integer(void) // next unsigned integer of the generator
{
    /* Nothing */
    return {};
};

_hot _nodiscard inline brain::FType rand_floating(void) // next floating value of the generator
{
    /* Nothing */
    return {};
};

} // namespace end
#endif /* RANDOM_H */
