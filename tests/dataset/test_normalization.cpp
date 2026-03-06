#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "dataset.h"

#include <cmath>

// helper to compare floats within a small tolerance
// we cannot use == directly because floating point arithmetic
// introduces tiny rounding errors
static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

TEST_CASE("min value normalizes to 0.0") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-setosa"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-setosa"}
    };

    std::vector<IrisSample> normalized = minMaxNormalize(dataset);

    // for each feature, the minimum value across all samples should be 0.0
    for (std::size_t feature = 0; feature < 4; ++feature) {
        float minVal = normalized[0].features[feature];
        for (const IrisSample& sample : normalized) {
            if (sample.features[feature] < minVal) {
                minVal = sample.features[feature];
            }
        }
        CHECK(approxEqual(minVal, 0.0f));
    }
}

TEST_CASE("max value normalizes to 1.0") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-setosa"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-setosa"}
    };

    std::vector<IrisSample> normalized = minMaxNormalize(dataset);

    for (std::size_t feature = 0; feature < 4; ++feature) {
        float maxVal = normalized[0].features[feature];
        for (const IrisSample& sample : normalized) {
            if (sample.features[feature] > maxVal) {
                maxVal = sample.features[feature];
            }
        }
        CHECK(approxEqual(maxVal, 1.0f));
    }
}

TEST_CASE("midpoint value normalizes to 0.5") {
    // min=0, max=10, mid=5 → should normalize to exactly 0.5
    std::vector<IrisSample> dataset = {
        {{0.0f, 0.0f, 0.0f, 0.0f}, "Iris-setosa"},
        {{5.0f, 5.0f, 5.0f, 5.0f}, "Iris-setosa"},
        {{10.0f, 10.0f, 10.0f, 10.0f}, "Iris-setosa"}
    };

    std::vector<IrisSample> normalized = minMaxNormalize(dataset);

    // sample at index 1 had the midpoint value in every feature
    for (std::size_t feature = 0; feature < 4; ++feature) {
        CHECK(approxEqual(normalized[1].features[feature], 0.5f));
    }
}

TEST_CASE("all normalized values are between 0.0 and 1.0") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-versicolor"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-virginica"}
    };

    std::vector<IrisSample> normalized = minMaxNormalize(dataset);

    for (const IrisSample& sample : normalized) {
        for (float val : sample.features) {
            CHECK(val >= 0.0f);
            CHECK(val <= 1.0f);
        }
    }
}

TEST_CASE("labels are preserved after normalization") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-versicolor"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-virginica"}
    };

    std::vector<IrisSample> normalized = minMaxNormalize(dataset);

    CHECK(normalized[0].label == "Iris-setosa");
    CHECK(normalized[1].label == "Iris-versicolor");
    CHECK(normalized[2].label == "Iris-virginica");
}

TEST_CASE("constant feature column does not divide by zero") {
    // all values in feature 0 are identical - min == max
    std::vector<IrisSample> dataset = {
        {{5.0f, 1.0f, 2.0f, 3.0f}, "Iris-setosa"},
        {{5.0f, 4.0f, 5.0f, 6.0f}, "Iris-setosa"},
        {{5.0f, 7.0f, 8.0f, 9.0f}, "Iris-setosa"}
    };

    // should not crash - your implementation must handle min == max
    std::vector<IrisSample> normalized = minMaxNormalize(dataset);

    // when min == max, the convention is to set normalized value to 0.0
    for (const IrisSample& sample : normalized) {
        CHECK(approxEqual(sample.features[0], 0.0f));
    }
}