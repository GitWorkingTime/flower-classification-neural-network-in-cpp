#include "doctest.h"
#include "activations.h"

#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// ─── relu ────────────────────────────────────────────────────────────────────

TEST_CASE("relu of positive number returns same number") {
    CHECK(approxEqual(relu(3.0f), 3.0f));
}

TEST_CASE("relu of negative number returns zero") {
    CHECK(approxEqual(relu(-3.0f), 0.0f));
}

TEST_CASE("relu of zero returns zero") {
    CHECK(approxEqual(relu(0.0f), 0.0f));
}

// ─── reluDerivative ──────────────────────────────────────────────────────────

TEST_CASE("relu derivative of positive number returns one") {
    CHECK(approxEqual(reluDerivative(3.0f), 1.0f));
}

TEST_CASE("relu derivative of negative number returns zero") {
    CHECK(approxEqual(reluDerivative(-3.0f), 0.0f));
}

TEST_CASE("relu derivative of zero returns zero") {
    CHECK(approxEqual(reluDerivative(0.0f), 0.0f));
}