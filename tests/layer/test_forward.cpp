#include "doctest.h"
#include "layer.h"

#include <cmath>
#include <numeric>

static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// helper - build a layer with known weights for deterministic testing
static Layer buildKnownLayer(ActivationType activation) {
    Layer layer;
    layer.activation = activation;

    // W = [[1, 0, 0, 0],
    //      [0, 1, 0, 0],
    //      [0, 0, 1, 0]]
    // identity-like matrix for predictable output
    layer.W = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f}
    };
    layer.b = {0.0f, 0.0f, 0.0f};

    return layer;
}

TEST_CASE("forward output size matches number of neurons") {
    Layer layer = createLayer(4, 8, ActivationType::ReLU);
    std::vector<float> x = {0.5f, 0.3f, 0.8f, 0.1f};

    std::vector<float> output = forward(layer, x);

    CHECK(output.size() == 8);
}

TEST_CASE("forward stores lastInput correctly") {
    Layer layer = createLayer(4, 8, ActivationType::ReLU);
    std::vector<float> x = {0.5f, 0.3f, 0.8f, 0.1f};

    forward(layer, x);

    CHECK(approxEqual(layer.lastInput[0], 0.5f));
    CHECK(approxEqual(layer.lastInput[1], 0.3f));
    CHECK(approxEqual(layer.lastInput[2], 0.8f));
    CHECK(approxEqual(layer.lastInput[3], 0.1f));
}

TEST_CASE("forward stores lastZ correctly") {
    // use known weights so we can predict z exactly
    Layer layer = buildKnownLayer(ActivationType::ReLU);
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    forward(layer, x);

    // z = Wx + b
    // z[0] = 1*1 + 0*2 + 0*3 + 0*4 + 0 = 1.0
    // z[1] = 0*1 + 1*2 + 0*3 + 0*4 + 0 = 2.0
    // z[2] = 0*1 + 0*2 + 1*3 + 0*4 + 0 = 3.0
    CHECK(approxEqual(layer.lastZ[0], 1.0f));
    CHECK(approxEqual(layer.lastZ[1], 2.0f));
    CHECK(approxEqual(layer.lastZ[2], 3.0f));
}

TEST_CASE("forward with ReLU zeroes negative pre-activations") {
    Layer layer;
    layer.activation = ActivationType::ReLU;
    layer.W = {
        { 1.0f, 0.0f},
        {-1.0f, 0.0f}
    };
    layer.b = {0.0f, 0.0f};

    std::vector<float> x = {1.0f, 0.0f};
    std::vector<float> output = forward(layer, x);

    // neuron 0: z = 1.0 → relu = 1.0
    // neuron 1: z = -1.0 → relu = 0.0
    CHECK(approxEqual(output[0], 1.0f));
    CHECK(approxEqual(output[1], 0.0f));
}

TEST_CASE("forward with ReLU passes positive pre-activations unchanged") {
    Layer layer = buildKnownLayer(ActivationType::ReLU);
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<float> output = forward(layer, x);

    CHECK(approxEqual(output[0], 1.0f));
    CHECK(approxEqual(output[1], 2.0f));
    CHECK(approxEqual(output[2], 3.0f));
}

TEST_CASE("forward with Softmax outputs sum to 1") {
    Layer layer = buildKnownLayer(ActivationType::Softmax);
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<float> output = forward(layer, x);

    float sum = 0.0f;
    for (float val : output) sum += val;

    CHECK(approxEqual(sum, 1.0f));
}

TEST_CASE("forward with Softmax outputs are all between 0 and 1") {
    Layer layer = buildKnownLayer(ActivationType::Softmax);
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<float> output = forward(layer, x);

    for (float val : output) {
        CHECK(val > 0.0f);
        CHECK(val < 1.0f);
    }
}