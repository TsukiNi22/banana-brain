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
##  @file random.hpp

File Description:
##  Random generator of the library and its randomness calls: one shared by the
##  program (BRAIN_SEED_GLOBAL, default) or one per translation unit (BRAIN_SEED_LOCAL)
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

    //----------------------------------------------------------------//
    /* DEFINE */

    /* generator scope (compile time, before the include) */
    // BRAIN_SEED_GLOBAL (default): one generator shared by the whole program (extern, defined in src/brain/tools/random.cpp)
    // BRAIN_SEED_LOCAL: one generator per translation unit including this header (static), each file has its own sequence
    #if defined(BRAIN_SEED_GLOBAL) && defined(BRAIN_SEED_LOCAL)
        #error "banana-brain: BRAIN_SEED_GLOBAL and BRAIN_SEED_LOCAL can't be defined at the same time"
    #elif !defined(BRAIN_SEED_LOCAL) && !defined(BRAIN_SEED_GLOBAL)
        #define BRAIN_SEED_GLOBAL
    #endif
    #ifdef BRAIN_SEED_LOCAL
        #define BRAIN_SEED_INLINE static inline // the helpers of a translation unit use its own generator (no ODR issue)
    #else
        #define BRAIN_SEED_INLINE inline
    #endif

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* GLOBAL */

#ifdef BRAIN_SEED_LOCAL
    static std::mt19937_64 generator; // generator of this translation unit (BRAIN_SEED_LOCAL)
#else
    extern std::mt19937_64 generator; // generator shared by the whole program (BRAIN_SEED_GLOBAL), defined in random.cpp
#endif

//----------------------------------------------------------------//
/* PROTOTYPE */

/* randomness (use the active generator) */
_cold BRAIN_SEED_INLINE void rand(const brain::IType seed) // setup the seed of the generator (0 = random seed)
{
    /* Nothing */
};

_hot _nodiscard BRAIN_SEED_INLINE brain::UIType rand_unsigned_integer(void) // next unsigned integer of the generator
{
    /* Nothing */
    return {};
};

_hot _nodiscard BRAIN_SEED_INLINE brain::FType rand_floating(void) // next floating value of the generator
{
    /* Nothing */
    return {};
};

} // namespace end
#endif /* RANDOM_H */
