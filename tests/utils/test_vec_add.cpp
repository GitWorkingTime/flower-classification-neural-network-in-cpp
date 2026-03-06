#include "doctest.h"
#include "utils.h"

#include <vector>
#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

TEST_CASE("vector addition basic case") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    std::vector<float> result = vecAdd(a, b);

    CHECK(result.size() == 3);
    CHECK(approxEqual(result[0], 5.0f));
    CHECK(approxEqual(result[1], 7.0f));
    CHECK(approxEqual(result[2], 9.0f));
}

TEST_CASE("vector addition with zero vector returns original") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> zeros = {0.0f, 0.0f, 0.0f};

    std::vector<float> result = vecAdd(a, zeros);

    CHECK(approxEqual(result[0], 1.0f));
    CHECK(approxEqual(result[1], 2.0f));
    CHECK(approxEqual(result[2], 3.0f));
}

TEST_CASE("vector addition is commutative") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    std::vector<float> ab = vecAdd(a, b);
    std::vector<float> ba = vecAdd(b, a);

    CHECK(approxEqual(ab[0], ba[0]));
    CHECK(approxEqual(ab[1], ba[1]));
    CHECK(approxEqual(ab[2], ba[2]));
}

TEST_CASE("vector addition returns empty on dimension mismatch") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f};

    std::vector<float> result = vecAdd(a, b);

    CHECK(result.empty());
}