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
##  @file Hook.hpp

File Description:
##  Global include of the training hook family (interface, abstract, implementations)
\**************************************************************/

#ifndef HOOK_H
    #define HOOK_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* interface */
    #include "ITrainHook.hpp"     // brain::ITrainHook

    /* abstract */
    #include "ATrainHook.hpp"     // brain::ATrainHook

    /* hook */
    #include "ProgressHook.hpp"   // brain::ProgressHook
    #include "MultiHook.hpp"      // brain::MultiHook
    #include "CsvHook.hpp"        // brain::CsvHook
    #include "LogHook.hpp"        // brain::LogHook

#endif /* HOOK_H */
