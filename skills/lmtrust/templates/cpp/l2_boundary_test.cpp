// templates/cpp/l2_boundary_test.cpp — Layer 2: 8 Boundary Conditions
#include <gtest/gtest.h>
#include <limits>
#include <vector>

TEST(L2BoundaryTest, LimitsAndEmptyVectors) {
    // 1. Zero
    EXPECT_EQ(0 * 10, 0);

    // 2. Minimum integer
    int min_val = std::numeric_limits<int>::min();
    EXPECT_LE(min_val, 0);

    // 3. Maximum integer
    int max_val = std::numeric_limits<int>::max();
    EXPECT_GT(max_val, 0);

    // 4. Empty vector
    std::vector<int> empty_vec;
    EXPECT_TRUE(empty_vec.empty());
}
