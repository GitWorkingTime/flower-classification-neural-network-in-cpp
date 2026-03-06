#include "doctest.h"
#include "layer.h"

#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

TEST_CASE("createLayer produces correct weight matrix dimensions") {
    Layer layer = createLayer(4, 8, ActivationType::ReLU);

    // W should be outputSize x inputSize = 8 x 4
    CHECK(layer.W.size() == 8);
    for (const std::vector<float>& row : layer.W) {
        CHECK(row.size() == 4);
    }
}

TEST_CASE("createLayer produces correct bias vector size") {
    Layer layer = createLayer(4, 8, ActivationType::ReLU);

    CHECK(layer.b.size() == 8);
}

TEST_CASE("createLayer initializes biases to zero") {
    Layer layer = createLayer(4, 8, ActivationType::ReLU);

    for (float bias : layer.b) {
        CHECK(approxEqual(bias, 0.0f));
    }
}

TEST_CASE("createLayer initializes weights within Xavier range") {
    // layer 1: inputSize=4, limit = 1/sqrt(4) = 0.5
    Layer layer = createLayer(4, 8, ActivationType::ReLU);

    float limit = 1.0f / std::sqrt(4.0f);

    for (const std::vector<float>& row : layer.W) {
        for (float w : row) {
            CHECK(w >= -limit);
            CHECK(w <= limit);
        }
    }
}

TEST_CASE("createLayer stores correct activation type") {
    Layer reluLayer = createLayer(4, 8, ActivationType::ReLU);
    Layer softmaxLayer = createLayer(8, 3, ActivationType::Softmax);

    CHECK(reluLayer.activation == ActivationType::ReLU);
    CHECK(softmaxLayer.activation == ActivationType::Softmax);
}

TEST_CASE("createLayer weights are not all zero") {
    // Xavier init should produce non-zero weights
    // probability of all 32 weights being exactly zero is negligible
    Layer layer = createLayer(4, 8, ActivationType::ReLU);

    bool allZero = true;
    for (const std::vector<float>& row : layer.W) {
        for (float w : row) {
            if (w != 0.0f) {
                allZero = false;
                break;
            }
        }
    }

    CHECK(allZero == false);
}