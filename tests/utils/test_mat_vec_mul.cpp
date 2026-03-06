#include "doctest.h"
#include "utils.h"

#include <vector>
#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

TEST_CASE("matrix vector multiplication basic case") {
    // W = [[1, 2], [3, 4]], x = [1, 1]
    // result = [1*1 + 2*1, 3*1 + 4*1] = [3, 7]
    std::vector<std::vector<float>> W = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    std::vector<float> x = {1.0f, 1.0f};

    std::vector<float> result = matVecMul(W, x);

    CHECK(result.size() == 2);
    CHECK(approxEqual(result[0], 3.0f));
    CHECK(approxEqual(result[1], 7.0f));
}

TEST_CASE("matrix vector multiplication with identity matrix") {
    std::vector<std::vector<float>> I = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };
    std::vector<float> x = {5.0f, 3.0f, 2.0f};

    std::vector<float> result = matVecMul(I, x);

    CHECK(result.size() == 3);
    CHECK(approxEqual(result[0], 5.0f));
    CHECK(approxEqual(result[1], 3.0f));
    CHECK(approxEqual(result[2], 2.0f));
}

TEST_CASE("matrix vector multiplication output size matches row count") {
    std::vector<std::vector<float>> W = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f}
    };
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<float> result = matVecMul(W, x);

    CHECK(result.size() == 3);
}

TEST_CASE("matrix vector multiplication returns empty on dimension mismatch") {
    std::vector<std::vector<float>> W = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    std::vector<float> x = {1.0f, 2.0f};

    std::vector<float> result = matVecMul(W, x);

    CHECK(result.empty());
}

TEST_CASE("matrix vector multiplication returns empty on empty matrix") {
    std::vector<std::vector<float>> W = {};
    std::vector<float> x = {1.0f, 2.0f};

    std::vector<float> result = matVecMul(W, x);

    CHECK(result.empty());
}