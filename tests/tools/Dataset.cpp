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
}

/* ------------------------------- text ------------------------------- */
TEST(Dataset, TextNormalIsOneValuePerCharacter) {
    const brain::Values input = brain::tools::text_to_input("ab");

    ASSERT_EQ(input.size(), 2u);
    EXPECT_FLOAT_EQ(input[0], 97.0 / 255.0);
    EXPECT_FLOAT_EQ(input[1], 98.0 / 255.0);
    for (const brain::FType value: input) {
        EXPECT_GE(value, 0.0); // small values only: usable as the inputs of a network
        EXPECT_LE(value, 1.0);
    }
}

TEST(Dataset, TextNormalRoundTrip) {
    const std::string text = "banana \xff\x01 end";

    EXPECT_EQ(brain::tools::input_to_text(brain::tools::text_to_input(text)), text);
    EXPECT_TRUE(brain::tools::text_to_input("").empty());
}

TEST(Dataset, TextNormalHandlesBadValues) {
    const brain::Values input = {-3.0, 2.0, std::nanf("")};

    EXPECT_EQ(brain::tools::input_to_text(input), std::string({'\0', static_cast<char>(255), '\0'}));
}

TEST(Dataset, TextCompactRoundTrip) {
    const std::string text = "banana";
    const brain::Values input = brain::tools::text_to_input<brain::TextMode::Compact>(text);

    EXPECT_EQ(input.size(), 2u); // 4 characters per value
    EXPECT_EQ((brain::tools::input_to_text<brain::TextMode::Compact, true>(input)), text); // the padding is removed with round
    EXPECT_EQ((brain::tools::input_to_text<brain::TextMode::Compact>(input)).size(), 8u);
}
