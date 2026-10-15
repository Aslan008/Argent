// templates/cpp/l4_adversarial_test.cpp — Layer 4: Adversarial error handling
#include <gtest/gtest.h>
#include <stdexcept>

TEST(L4AdversarialTest, ControlledExceptionOnBadInput) {
    auto safe_divide = [](int a, int b) -> int {
        if (b == 0) throw std::invalid_argument("Division by zero");
        return a / b;
    };

    EXPECT_THROW(safe_divide(10, 0), std::invalid_argument);
}
