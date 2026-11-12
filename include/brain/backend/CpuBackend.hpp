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
##  @file CpuBackend.hpp

File Description:
##  Default backend: everything on the calling thread (the code of brain::ABackend)
\**************************************************************/

#ifndef CPUBACKEND_H
    #define CPUBACKEND_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    #define _Attribute
    #include <utils/utils.hpp>         // _cold, _nodiscard
    #include "../NetworkDefine.hpp"    // brain::BackendType
    #include "IBackend.hpp"            // brain::IBackend
    #include "ABackend.hpp"            // brain::ABackend
    #include <memory>                  // std::shared_ptr

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class CpuBackend: public brain::ABackend {
    public:
        // ---------- Pre-Function -------- //
        /* shared */
        static std::shared_ptr<brain::IBackend> shared(void); // one instance for every network / NeuralNetwork without a backend of its own

        // ------------ Function ---------- //
        /* info */
        _cold _nodiscard brain::BackendType type(void) const override {return brain::BackendType::Cpu;};

        // ------------ Operator ---------- //
        CpuBackend& operator=(const CpuBackend& other) = delete;
        CpuBackend& operator=(CpuBackend&& other) = delete;

        // ---------- Constructor --------- //
        CpuBackend() = default;
        CpuBackend(const CpuBackend& other) = delete;
        CpuBackend(CpuBackend&& other) = delete;

        // ----------- Destructor --------- //
        ~CpuBackend() = default;
};

} // namespace end
#endif /* CPUBACKEND_H */
