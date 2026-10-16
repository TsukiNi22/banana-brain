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
##  @file LayerNetwork.cpp

File Description:
##  Unit tests of the LayerNetwork
\**************************************************************/

#include "brain/network/Network.hpp"
#include <gtest/gtest.h>
#include <memory>

/* ------------------------------- LayerNetwork ------------------------------- */
TEST(LayerNetwork, Construct) {
    EXPECT_NO_THROW(brain::LayerNetwork network(2, 3););
}

TEST(LayerNetwork, UsableAsINetwork) {
    std::unique_ptr<brain::INetwork> network = std::make_unique<brain::LayerNetwork>(2, 3);
    ASSERT_NE(network, nullptr);
    EXPECT_NO_THROW(network->inputs({0.0, 1.0}));
}
