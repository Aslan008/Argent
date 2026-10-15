// templates/cpp/l1_contract_test.cpp — Layer 1: Contract & Known Coordinates
#include <gtest/gtest.h>

TEST(L1ContractTest, ExactKnownCoordinates) {
    int input = 100;
    int expected = 250;
    int actual = input * 2 + 50; // Replace with target function
    ASSERT_EQ(actual, expected);
}
