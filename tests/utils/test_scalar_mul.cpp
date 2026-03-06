#include "doctest.h"
#include "utils.h"

#include <vector>
#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

TEST_CASE("scalar multiplication basic case") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};

    std::vector<float> result = scalarMul(a, 2.0f);

    CHECK(approxEqual(result[0], 2.0f));
    CHECK(approxEqual(result[1], 4.0f));
    CHECK(approxEqual(result[2], 6.0f));
}

TEST_CASE("scalar multiplication by zero returns zero vector") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};

    std::vector<float> result = scalarMul(a, 0.0f);

    CHECK(approxEqual(result[0], 0.0f));
    CHECK(approxEqual(result[1], 0.0f));
    CHECK(approxEqual(result[2], 0.0f));
}

TEST_CASE("scalar multiplication by one returns original") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};

    std::vector<float> result = scalarMul(a, 1.0f);

    CHECK(approxEqual(result[0], 1.0f));
    CHECK(approxEqual(result[1], 2.0f));
    CHECK(approxEqual(result[2], 3.0f));
}

TEST_CASE("scalar multiplication returns empty on empty vector") {
    std::vector<float> a = {};

    std::vector<float> result = scalarMul(a, 2.0f);

    CHECK(result.empty());
}