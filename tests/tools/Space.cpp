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
##  @file Space.cpp

File Description:
##  Unit tests of the space tools
\**************************************************************/

#define _Exception
#include "brain/tools/space.hpp"
#include <gtest/gtest.h>
#include <vector>

static std::vector<brain::RawBrainNeuron> line(const std::vector<brain::FType>& positions)
{
    std::vector<brain::RawBrainNeuron> neurons(positions.size());

    for (std::size_t i = 0; i < positions.size(); ++i)
        neurons[i].pos = {positions[i], 0.0};
    return neurons;
}

/* ------------------------------- space ------------------------------- */
TEST(Space, ClosestReturnsTheNearestClosestFirst) {
    const std::vector<brain::RawBrainNeuron> neurons = line({0.0, 5.0, 1.0, 3.0});
    const std::vector<brain::UIType> nearest = brain::tools::closest(neurons, 0, 2);

    ASSERT_EQ(nearest.size(), 2u);
    EXPECT_EQ(nearest[0], 2u); // squared distance 1
    EXPECT_EQ(nearest[1], 3u); // 9 (the neuron 1 at 25 is over the count)
}

TEST(Space, ClosestNeverReturnsTheNeuronItself) {
    const std::vector<brain::RawBrainNeuron> neurons = line({0.0, 1.0, 2.0});
    const std::vector<brain::UIType> nearest = brain::tools::closest(neurons, 1, 10); // count above the other neurons: all of them

    ASSERT_EQ(nearest.size(), 2u);
    EXPECT_EQ(nearest[0], 0u); // tie (1 and 1): the lower id first
    EXPECT_EQ(nearest[1], 2u);
}

TEST(Space, ClosestWithoutCapOrNeighbours) {
    EXPECT_TRUE(brain::tools::closest(line({0.0, 1.0}), 0, 0).empty());
    EXPECT_TRUE(brain::tools::closest(line({0.0}), 0, 5).empty());
}

TEST(Space, ClosestRejectsABadIndex) {
    EXPECT_THROW((void)brain::tools::closest(line({0.0, 1.0, 2.0}), 3, 1), utils::exception::IException);
    EXPECT_THROW((void)brain::tools::closest(std::vector<brain::RawBrainNeuron>{}, 0, 1), utils::exception::IException);
}
