/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 30/10/2026 by @author Tsukini

File Name:
##  @file sum.hpp

File Description:
##  Tools of the sum of the links of a neuron and of the end of the time steps:
##  four partial sums (the additions of a row do not wait for each other), the
##  same order everywhere so that every execution gives the same value
\**************************************************************/

#ifndef BRAIN_SUM_H
    #define BRAIN_SUM_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>      // _hot, _nodiscard
    #include "../NetworkType.hpp"   // brain::FType, brain::UIType, brain::BrainLink, brain::RawBrainLink
    #include <functional>           // std::function
    #include <cstring>              // std::memcmp
    #include <barrier>              // std::barrier
    #include <thread>               // std::jthread
    #include <vector>               // std::vector
    #include <cmath>                // std::abs

namespace brain::tools { // namespace start
//----------------------------------------------------------------//
/* PROTOTYPE */

// Sum of weight * state[id] over the packed links of a row of the reduced network (a weight of 0 ends it): the link j goes to the partial sum j % 4, then (0 + 1) + (2 + 3)
_hot _nodiscard inline brain::FType rowSum(const brain::BrainLink* row, const brain::UIType cap, const brain::FType* state)
{
    brain::FType s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    brain::UIType j = 0;

    for (; j + 4 <= cap && row[j + 3].weight != 0; j += 4) {
        s0 += row[j].weight * state[row[j].id];
        s1 += row[j + 1].weight * state[row[j + 1].id];
        s2 += row[j + 2].weight * state[row[j + 2].id];
        s3 += row[j + 3].weight * state[row[j + 3].id];
    }
    if (j < cap && row[j].weight != 0) {
        s0 += row[j].weight * state[row[j].id];
        if (j + 1 < cap && row[j + 1].weight != 0) {
            s1 += row[j + 1].weight * state[row[j + 1].id];
            if (j + 2 < cap && row[j + 2].weight != 0) s2 += row[j + 2].weight * state[row[j + 2].id];
        }
    }
    return (s0 + s1) + (s2 + s3);
}

// The same on the links of a neuron of the raw network: the slots[from .. to - 1] of its inward links (in the order of the slots)
_hot _nodiscard inline brain::FType listSum(const brain::RawBrainLink* links, const brain::UIType* slots, const brain::UIType from, const brain::UIType to, const brain::FType* state)
{
    brain::FType s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    brain::UIType k = from;

    for (; k + 4 <= to; k += 4) {
        s0 += links[slots[k]].weight * state[links[slots[k]].id];
        s1 += links[slots[k + 1]].weight * state[links[slots[k + 1]].id];
        s2 += links[slots[k + 2]].weight * state[links[slots[k + 2]].id];
        s3 += links[slots[k + 3]].weight * state[links[slots[k + 3]].id];
    }
    if (k < to) {
        s0 += links[slots[k]].weight * state[links[slots[k]].id];
        if (k + 1 < to) {
            s1 += links[slots[k + 1]].weight * state[links[slots[k + 1]].id];
            if (k + 2 < to) s2 += links[slots[k + 2]].weight * state[links[slots[k + 2]].id];
        }
    }
    return (s0 + s1) + (s2 + s3);
}

// Did the time steps settle? Every state equal to the one before to the bit (tolerance 0), else none moved by more than the tolerance (the signal threshold): the next steps are copies
_hot _nodiscard inline bool settled(const brain::FType* now, const brain::FType* before, const brain::UIType size, const brain::FType tolerance)
{
    if (tolerance <= 0) return std::memcmp(now, before, size * sizeof(brain::FType)) == 0;
    for (brain::UIType i = 0; i < size; ++i)
        if (!(std::abs(now[i] - before[i]) <= tolerance)) return false; // not (<=): a state that is not a number never settles
    return true;
}

// The time steps on several threads: body(thread, step) on each of the <threads> threads (the caller is the thread 0), then a barrier where done(step) runs once, alone (it returns true to stop: nothing moved any more); the same results as one thread whatever the number of threads (every neuron is computed by one thread, with its own sum)
template<typename Body, typename Done>
_hot inline void stepThreads(const brain::UIType threads, const brain::UIType steps, Body&& body, Done&& done)
{
    brain::UIType step = 0;
    bool stop = false;
    std::barrier sync(static_cast<std::ptrdiff_t>(threads), [&]() noexcept {stop = done(step); ++step;});
    const std::function<void(const brain::UIType)> loop = [&](const brain::UIType thread) {
        while (step < steps && !stop) {
            body(thread, step);
            sync.arrive_and_wait();
        }
    };
    std::vector<std::jthread> workers;

    for (brain::UIType thread = 1; thread < threads; ++thread) workers.emplace_back(loop, thread);
    loop(0);
}

} // namespace end
#endif /* BRAIN_SUM_H */
