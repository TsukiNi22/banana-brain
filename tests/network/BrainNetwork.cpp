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
##  @file BrainNetwork.cpp

File Description:
##  Unit tests of the BrainNetwork
\**************************************************************/

#include "brain/network/Network.hpp"
#include <gtest/gtest.h>
#include <memory>

/* ------------------------------- BrainNetwork ------------------------------- */
TEST(BrainNetwork, Construct) {
    EXPECT_NO_THROW(brain::BrainNetwork network(10););
    EXPECT_NO_THROW(brain::BrainNetwork network(10, {2.0, 3.0}););
}

TEST(BrainNetwork, UsableAsINetwork) {
    std::unique_ptr<brain::INetwork> network = std::make_unique<brain::BrainNetwork>(10);
    ASSERT_NE(network, nullptr);
    EXPECT_NO_THROW(network->inputs({0.0, 1.0}));
}
