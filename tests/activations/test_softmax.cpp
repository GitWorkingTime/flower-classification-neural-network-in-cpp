#include "doctest.h"
#include "activations.h"

#include <vector>
#include <cmath>
#include <numeric>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// ─── softmax ─────────────────────────────────────────────────────────────────

TEST_CASE("softmax outputs sum to 1") {
    std::vector<float> z = {2.1f, 5.3f, 0.7f};
    std::vector<float> result = softmax(z);

    float sum = 0.0f;
    for (float val : result) sum += val;

    CHECK(approxEqual(sum, 1.0f));
}

TEST_CASE("softmax outputs are all between 0 and 1") {
    std::vector<float> z = {2.1f, 5.3f, 0.7f};
    std::vector<float> result = softmax(z);

    for (float val : result) {
        CHECK(val > 0.0f);
        CHECK(val < 1.0f);
    }
}

TEST_CASE("softmax largest input gets largest probability") {
    std::vector<float> z = {2.1f, 5.3f, 0.7f};
    std::vector<float> result = softmax(z);

    // index 1 has the largest input so it should have the largest probability
    CHECK(result[1] > result[0]);
    CHECK(result[1] > result[2]);
}

TEST_CASE("softmax of equal inputs produces uniform distribution") {
    std::vector<float> z = {1.0f, 1.0f, 1.0f};
    std::vector<float> result = softmax(z);

    // all outputs should be equal to 1/3
    CHECK(approxEqual(result[0], 1.0f / 3.0f));
    CHECK(approxEqual(result[1], 1.0f / 3.0f));
    CHECK(approxEqual(result[2], 1.0f / 3.0f));
}

TEST_CASE("softmax output size matches input size") {
    std::vector<float> z = {1.0f, 2.0f, 3.0f};
    std::vector<float> result = softmax(z);

    CHECK(result.size() == z.size());
}

TEST_CASE("softmax does not change ranking of inputs") {
    std::vector<float> z = {1.0f, 3.0f, 2.0f};
    std::vector<float> result = softmax(z);

    // ranking should be preserved: index 1 > index 2 > index 0
    CHECK(result[1] > result[2]);
    CHECK(result[2] > result[0]);
}