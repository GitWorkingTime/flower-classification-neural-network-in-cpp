#include "doctest.h"
#include "utils.h"

#include <vector>
#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

TEST_CASE("transpose of a 2x3 matrix produces a 3x2 matrix") {
    std::vector<std::vector<float>> W = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    std::vector<std::vector<float>> result = transpose(W);

    CHECK(result.size() == 3);
    CHECK(result[0].size() == 2);
}

TEST_CASE("transpose swaps rows and columns correctly") {
    std::vector<std::vector<float>> W = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    std::vector<std::vector<float>> result = transpose(W);

    CHECK(approxEqual(result[0][0], 1.0f));
    CHECK(approxEqual(result[1][0], 2.0f));
    CHECK(approxEqual(result[2][0], 3.0f));
    CHECK(approxEqual(result[0][1], 4.0f));
    CHECK(approxEqual(result[1][1], 5.0f));
    CHECK(approxEqual(result[2][1], 6.0f));
}

TEST_CASE("transpose of transpose returns original matrix") {
    std::vector<std::vector<float>> W = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    std::vector<std::vector<float>> result = transpose(transpose(W));

    CHECK(approxEqual(result[0][0], 1.0f));
    CHECK(approxEqual(result[0][1], 2.0f));
    CHECK(approxEqual(result[0][2], 3.0f));
    CHECK(approxEqual(result[1][0], 4.0f));
    CHECK(approxEqual(result[1][1], 5.0f));
    CHECK(approxEqual(result[1][2], 6.0f));
}

TEST_CASE("transpose of empty matrix returns empty") {
    std::vector<std::vector<float>> W = {};

    std::vector<std::vector<float>> result = transpose(W);

    CHECK(result.empty());
}