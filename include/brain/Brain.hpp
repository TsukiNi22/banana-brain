/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗ 
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 08/09/2026 by @author Tsukini

File Name:
##  @file Brain.hpp

File Description:
##  You know, I don t think there are good or bad descriptions,
##  for me, life is all about functions...
\**************************************************************/

#ifndef CORE_H
    #define CORE_H

    //----------------------------------------------------------------//
    /* INCLUDE */

    /* type */
    //#include <iostream>

namespace brain { // namespace start
//----------------------------------------------------------------//
/* CLASS */

class Brain {
    private:
        /* Nothing */

    public:
        // ---------- Pre-Function -------- //
        void none(void);

        // ------------ Function ---------- //

        // ------------ Operator ---------- //
        Brain& operator=(const Brain& other) = delete;
        Brain& operator=(Brain&& other) = delete;

        // ---------- Constructor --------- //
        Brain() = default;
        Brain(const Brain& other) = delete;
        Brain(Brain&& other) = delete;

        // ----------- Destructor --------- //
        ~Brain() = default;
};

} // namespace end
#endif /* CORE_H */
