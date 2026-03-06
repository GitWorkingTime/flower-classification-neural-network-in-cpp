#include "doctest.h"
#include "utils.h"

#include <vector>
#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

TEST_CASE("dot product of two simple vectors") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    CHECK(approxEqual(dot(a, b), 32.0f));
}

TEST_CASE("dot product of orthogonal vectors is zero") {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};

    CHECK(approxEqual(dot(a, b), 0.0f));
}

TEST_CASE("dot product is commutative") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    CHECK(approxEqual(dot(a, b), dot(b, a)));
}

TEST_CASE("dot product returns zero on dimension mismatch") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f};

    CHECK(approxEqual(dot(a, b), 0.0f));
}