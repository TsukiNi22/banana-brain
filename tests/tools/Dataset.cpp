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
##  @file Dataset.cpp

File Description:
##  Unit tests of the dataset tools
\**************************************************************/

#include "brain/tools/dataset.hpp"
#include <gtest/gtest.h>
#include <string>

/* ------------------------------- dataset ------------------------------- */
TEST(Dataset, CallsDoNotThrow) {
    brain::Dataset dataset = {{0.0, 1.0}, {1.0, 0.0}};
    EXPECT_NO_THROW(brain::tools::shuffle(dataset));
    EXPECT_NO_THROW((void)brain::tools::rand_sets(dataset, 0.5));
    EXPECT_NO_THROW((void)brain::tools::text_to_input("banana"));
    EXPECT_NO_THROW((void)brain::tools::input_to_text({0.0, 1.0}));
}
