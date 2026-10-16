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

#include "brain/tools/neuron.hpp"
#include <gtest/gtest.h>
#include <vector>

/* ------------------------------- neuron ------------------------------- */
TEST(Neuron, SortDoesNotThrow) {
    std::vector<brain::BrainNeuron> neurons(3);
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i].id = i;
        neurons[i].pos = {static_cast<brain::FType>(i), 0.0};
    }
    EXPECT_NO_THROW(brain::tools::sort(neurons[0], neurons));
}
