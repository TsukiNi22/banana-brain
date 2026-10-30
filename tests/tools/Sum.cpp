/**************************************************************\

 ██╗  ██╗ █████╗ ██████╗ ████████╗ █████╗ ███╗   ██╗██╗ █████╗
 ╚██╗██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗████╗  ██║██║██╔══██╗
  ╚███╔╝ ███████║██████╔╝   ██║   ███████║██╔██╗ ██║██║███████║
  ██╔██╗ ██╔══██║██╔══██╗   ██║   ██╔══██║██║╚██╗██║██║██╔══██║
 ██╔╝ ██╗██║  ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

Edition:
##  @date 30/10/2026 by @author Tsukini

File Name:
##  @file Sum.cpp

File Description:
##  Unit tests of the fast exponential and sigmoid, of the sums of the links
##  and of the end of the time steps
\**************************************************************/

#include "brain/tools/activation.hpp"
#include "brain/tools/sum.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(Sum, FastExpIsCloseToTheExponential) {
    brain::FType worst = 0;

    for (brain::FType x = -80.0f; x <= 80.0f; x += 0.0137f) {
        const double exact = std::exp(static_cast<double>(x));

        worst = std::max(worst, static_cast<brain::FType>(std::abs(static_cast<double>(brain::tools::fastExp(x)) - exact) / exact));
    }
    EXPECT_LT(worst, 3e-7f); // a few ulp of a float
    EXPECT_EQ(brain::tools::fastExp(0.0f), 1.0f);
    EXPECT_GT(brain::tools::fastExp(-1000.0f), 0.0f); // clamped: never a wrong sign or an overflow
    EXPECT_TRUE(std::isfinite(brain::tools::fastExp(1000.0f)));
}

TEST(Sum, FastSigmoidIsCloseToTheSigmoid) {
    brain::FType worst = 0;

    for (brain::FType x = -40.0f; x <= 40.0f; x += 0.0071f) worst = std::max(worst, std::abs(brain::tools::fastSigmoid(x) - brain::tools::sigmoid(x)));
    EXPECT_LT(worst, 2e-7f);
    EXPECT_FLOAT_EQ(brain::tools::fastSigmoid(0.0f), 0.5f);
    EXPECT_NEAR(brain::tools::fastSigmoid(-200.0f), 0.0f, 1e-30f);
    EXPECT_FLOAT_EQ(brain::tools::fastSigmoid(200.0f), 1.0f);
}

TEST(Sum, RowAndListSumsAreTheSameWhateverTheLayout) {
    // 11 links (a remainder of 3 after two groups of four), weights and states of different sizes: the same value on the two layouts, and the order (0 + 1) + (2 + 3) of the partial sums
    const brain::UIType cap = 16;
    std::vector<brain::BrainLink> row(cap, brain::BrainLink{0, 0.0f});
    std::vector<brain::RawBrainLink> raw(cap);
    std::vector<brain::UIType> slots;
    std::vector<brain::FType> state = {0.1f, 0.7f, 0.3f, 0.9f, 0.5f, 0.2f, 0.8f, 0.4f, 0.6f, 0.05f, 0.95f, 0.35f};
    float s[4] = {0, 0, 0, 0};

    for (brain::UIType j = 0; j < 11; ++j) {
        const brain::FType weight = (j % 2 ? -1.0f : 1.0f) * (0.3f + 0.17f * static_cast<brain::FType>(j));

        row[j] = {j, weight};
        raw[j] = {j, weight, true};
        slots.push_back(j);
        s[j % 4] += weight * state[j];
    }
    EXPECT_EQ(brain::tools::rowSum(row.data(), cap, state.data()), (s[0] + s[1]) + (s[2] + s[3]));
    EXPECT_EQ(brain::tools::listSum(raw.data(), slots.data(), 0, 11, state.data()), (s[0] + s[1]) + (s[2] + s[3]));
    EXPECT_EQ(brain::tools::rowSum(row.data(), cap, state.data()), brain::tools::listSum(raw.data(), slots.data(), 0, 11, state.data()));
    for (brain::UIType count = 0; count <= 11; ++count) { // every length: a row ends at its first free slot, a list at its end
        std::vector<brain::BrainLink> part(cap, brain::BrainLink{0, 0.0f});
        std::copy(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(count), part.begin());
        EXPECT_EQ(brain::tools::rowSum(part.data(), cap, state.data()), brain::tools::listSum(raw.data(), slots.data(), 0, count, state.data())) << "count " << count;
    }
}

TEST(Sum, SettledMeansNoStateMovedMoreThanTheTolerance) {
    const std::vector<brain::FType> before = {0.1f, 0.2f, 0.3f};
    std::vector<brain::FType> now = before;

    EXPECT_TRUE(brain::tools::settled(now.data(), before.data(), 3, 0.0f)); // to the bit
    now[1] += 1e-7f;
    EXPECT_FALSE(brain::tools::settled(now.data(), before.data(), 3, 0.0f));
    EXPECT_TRUE(brain::tools::settled(now.data(), before.data(), 3, 1e-6f)); // under the tolerance
    EXPECT_FALSE(brain::tools::settled(now.data(), before.data(), 3, 1e-8f));
    now[2] = std::nanf("");
    EXPECT_FALSE(brain::tools::settled(now.data(), before.data(), 3, 1.0f)); // a state that is not a number never settles
}
