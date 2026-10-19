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
##  @file Neuron.cpp

File Description:
##  Unit tests of the brain neuron tools
\**************************************************************/

#define _Exception
#include "brain/tools/neuron.hpp"
#include <gtest/gtest.h>
#include <vector>

/* ------------------------------- neuron ------------------------------- */
TEST(Neuron, SortOrdersTheDistancesClosestFirst) {
    std::vector<brain::BrainNeuron> neurons(4);
    std::vector<brain::BrainDistance> ordered(4 * 3);
    const brain::FType positions[4] = {0.0, 5.0, 1.0, 3.0};

    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i].id = i;
        neurons[i].pos = {positions[i], 0.0};
    }
    ASSERT_NO_THROW(brain::tools::sort(0, neurons, ordered));
    EXPECT_EQ(neurons[0].orderedCount, 3u);
    EXPECT_EQ(ordered[0].id, 2u); // squared distance 1
    EXPECT_EQ(ordered[1].id, 3u); // 9
    EXPECT_EQ(ordered[2].id, 1u); // 25
    EXPECT_FLOAT_EQ(ordered[2].distance, 25.0);
}

TEST(Neuron, SortStoresTheDistanceOnTheOtherSideToo) {
    std::vector<brain::BrainNeuron> neurons(3);
    std::vector<brain::BrainDistance> ordered(3 * 2);

    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i].id = i;
        neurons[i].pos = {static_cast<brain::FType>(i), 0.0};
    }
    brain::tools::sort(0, neurons, ordered);
    EXPECT_EQ(ordered[neurons[2].ordered + 0].id, 0u); // neuron 2 knows the distance to neuron 0
    EXPECT_FLOAT_EQ(ordered[neurons[2].ordered + 0].distance, 4.0);
}

TEST(Neuron, SortRejectsABadIndexOrMemory) {
    std::vector<brain::BrainNeuron> neurons(3);
    std::vector<brain::BrainDistance> ordered(3 * 2);
    std::vector<brain::BrainDistance> small(1);

    EXPECT_THROW(brain::tools::sort(3, neurons, ordered), utils::exception::IException);
    EXPECT_THROW(brain::tools::sort(0, neurons, small), utils::exception::IException);
}
