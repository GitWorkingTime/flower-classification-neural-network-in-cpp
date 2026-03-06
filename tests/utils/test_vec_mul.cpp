#include "doctest.h"
#include "utils.h"

#include <vector>
#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

TEST_CASE("element wise vector multiplication basic case") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    std::vector<float> result = vecMul(a, b);

    CHECK(result.size() == 3);
    CHECK(approxEqual(result[0], 4.0f));
    CHECK(approxEqual(result[1], 10.0f));
    CHECK(approxEqual(result[2], 18.0f));
}

TEST_CASE("element wise multiplication by ones returns original") {
    std::vector<float> a = {3.0f, 5.0f, 7.0f};
    std::vector<float> ones = {1.0f, 1.0f, 1.0f};

    std::vector<float> result = vecMul(a, ones);

    CHECK(approxEqual(result[0], 3.0f));
    CHECK(approxEqual(result[1], 5.0f));
    CHECK(approxEqual(result[2], 7.0f));
}

TEST_CASE("element wise multiplication by zeros returns zeros") {
    std::vector<float> a = {3.0f, 5.0f, 7.0f};
    std::vector<float> zeros = {0.0f, 0.0f, 0.0f};

    std::vector<float> result = vecMul(a, zeros);

    CHECK(approxEqual(result[0], 0.0f));
    CHECK(approxEqual(result[1], 0.0f));
    CHECK(approxEqual(result[2], 0.0f));
}

TEST_CASE("element wise multiplication returns empty on dimension mismatch") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f};

    std::vector<float> result = vecMul(a, b);

    CHECK(result.empty());
}