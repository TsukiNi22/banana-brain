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
##  @file ThreadBackend.hpp

File Description:
##  Multi thread backend: the work split between workers, without touching the networks
\**************************************************************/

#ifndef THREADBACKEND_H
    #define THREADBACKEND_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>               // _cold, _nodiscard
    #include "../NetworkDefine.hpp"          // brain::BackendType, BRAIN_DEFAULT_THREADS
    #include "../NetworkType.hpp"            // brain::UIType
    #include "ABackend.hpp"                  // brain::ABackend
    #include <condition_variable>            // std::condition_variable
    #include <functional>                    // std::function
    #include <thread>                        // std::jthread
    #include <vector>                        // std::vector
    #include <mutex>                         // std::mutex
    #include <queue>                         // std::queue

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class ThreadBackend: public brain::ABackend {
    private:
        /* workers */
        std::vector<std::jthread> _workers; // started by init(), stopped by release()
        std::queue<std::function<void(void)>> _jobs; // chunks waiting for a worker
        std::condition_variable _condition; // a worker wakes up on a new job / the stop
        std::mutex _lock; // Handling of multithreading (_jobs)

    public:
        // ---------- Pre-Function -------- //
        /* lifecycle */
        void init(void) override; // start the workers
        void release(void) override; // stop the workers
        void sync(void) override; // wait for every pending job

        /* parallel */
        void parallelFor(const brain::UIType count, const std::function<void(const brain::UIType begin, const brain::UIType end, const brain::UIType worker)>& fn) override; // the range split in _threads chunks (at least _grain items each), the calling thread takes one chunk

        /* setter */
        using brain::ABackend::threads;
        void threads(const brain::UIType threads); // release + init again when ready (0 = hardware_concurrency())

        // ------------ Function ---------- //
        /* info */
        _cold _nodiscard brain::BackendType type(void) const override {return brain::BackendType::Threads;};

        // ------------ Operator ---------- //
        ThreadBackend& operator=(const ThreadBackend& other) = delete;
        ThreadBackend& operator=(ThreadBackend&& other) = delete;

        // ---------- Constructor --------- //
        ThreadBackend(const brain::UIType threads = BRAIN_DEFAULT_THREADS); // 0 = hardware_concurrency()
        ThreadBackend(const ThreadBackend& other) = delete;
        ThreadBackend(ThreadBackend&& other) = delete;

        // ----------- Destructor --------- //
        ~ThreadBackend() { /* Nothing */ }; // the workers stopped (release)
};

} // namespace end
#endif /* THREADBACKEND_H */
