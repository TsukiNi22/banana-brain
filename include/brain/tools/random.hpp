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
##  @file random.hpp

File Description:
##  Random generator of the library and its randomness calls (one per thread): one shared by the
##  program (BRAIN_SEED_GLOBAL, default) or one per translation unit (BRAIN_SEED_LOCAL)
\**************************************************************/

#ifndef RANDOM_H
    #define RANDOM_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _hot, _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::RandInit
    #include "../NetworkType.hpp"      // brain::IType, brain::UIType, brain::FType
    #include <utility>                 // std::pair
    #include <cstdint>                 // std::uint64_t
    #include <random>                  // std::mt19937_64, std::random_device, std::generate_canonical
    #include <limits>                  // std::numeric_limits

    //----------------------------------------------------------------//
    /* DEFINE */

    /* generator scope (compile time, before the include) */
    // BRAIN_SEED_GLOBAL (default): one generator per thread shared by the whole program (extern, defined in src/brain/tools/random.cpp)
    // BRAIN_SEED_LOCAL: one generator per thread and per translation unit including this header (static), each file has its own sequence
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

// One generator per thread: the workers of a backend draw without lock
#ifdef BRAIN_SEED_LOCAL
    static thread_local std::mt19937_64 generator; // generator of this translation unit (BRAIN_SEED_LOCAL)
#else
    extern thread_local std::mt19937_64 generator; // generator shared by the whole program (BRAIN_SEED_GLOBAL), defined in random.cpp
#endif

//----------------------------------------------------------------//
/* PROTOTYPE */

/* generator (use the generator of the calling thread) */
_cold BRAIN_SEED_INLINE void rand(const brain::IType seed) // setup the seed of the generator (0 = random seed)
{
    std::random_device device;

    brain::tools::generator.seed(seed == 0 ? static_cast<std::uint64_t>(device()) : static_cast<std::uint64_t>(seed));
};

_cold BRAIN_SEED_INLINE void rand(const brain::IType seed, const brain::UIType worker) // seed of a worker: seed + worker (the same sequences with the same number of workers, 0 = random seed)
{
    /* Nothing */
};

_hot _nodiscard BRAIN_SEED_INLINE brain::UIType rand_unsigned_integer(void) // next unsigned integer of the generator
{
    return static_cast<brain::UIType>(brain::tools::generator());
};

_hot _nodiscard BRAIN_SEED_INLINE brain::FType rand_floating(void) // next floating value of the generator, in [0, 1)
{
    return std::generate_canonical<brain::FType, std::numeric_limits<brain::FType>::digits>(brain::tools::generator);
};

/* distribution */
_hot _nodiscard BRAIN_SEED_INLINE brain::FType rand_range(const brain::FType min, const brain::FType max) // uniform in [min, max)
{
    /* Nothing */
    return {};
};

_hot _nodiscard BRAIN_SEED_INLINE brain::FType rand_normal(const brain::FType mean = 0.0, const brain::FType deviation = 1.0) // normal distribution
{
    /* Nothing */
    return {};
};

/* weights init */
_hot _nodiscard BRAIN_SEED_INLINE brain::FType xavier(const brain::UIType fan_in, const brain::UIType fan_out) // uniform in [-sqrt(6 / (in + out)), sqrt(6 / (in + out))]
{
    /* Nothing */
    return {};
};

_hot _nodiscard BRAIN_SEED_INLINE brain::FType he(const brain::UIType fan_in) // normal with a deviation of sqrt(2 / in) (relu)
{
    /* Nothing */
    return {};
};

_hot _nodiscard BRAIN_SEED_INLINE brain::FType rand_init(const brain::RandInit mode, const brain::UIType fan_in, const brain::UIType fan_out, const std::pair<brain::FType, brain::FType>& range) // one entry point for every mode (range: Uniform / Normal)
{
    /* Nothing */
    return {};
};

} // namespace end
#endif /* RANDOM_H */
