#include "doctest.h"
#include "layer.h"

#include <cmath>

static bool approxEqual(float a, float b, float epsilon = 1e-4f) {
    return std::fabs(a - b) < epsilon;
}

// helper - build a layer with known weights and run a forward pass
// so lastInput and lastZ are populated before backward is called
static Layer buildAndForward() {
    Layer layer;
    layer.activation = ActivationType::ReLU;
    layer.W = {
        {0.5f, 0.5f},
        {0.5f, 0.5f}
    };
    layer.b = {0.0f, 0.0f};

    std::vector<float> x = {1.0f, 1.0f};
    forward(layer, x);

    return layer;
}

TEST_CASE("backward returns delta for previous layer with correct size") {
    Layer layer = buildAndForward();
    std::vector<float> delta = {0.1f, 0.2f};

    std::vector<float> deltaPrev = backward(layer, delta, 0.01f);

    // deltaPrev should be same size as number of inputs (2)
    CHECK(deltaPrev.size() == 2);
}

TEST_CASE("backward updates weights in correct direction") {
    Layer layer = buildAndForward();

    // record original weights
    float originalW00 = layer.W[0][0];

    std::vector<float> delta = {1.0f, 1.0f};
    backward(layer, delta, 0.1f);

    // with positive delta and positive input, weight should decrease
    // W = W - lr * delta * input
    // W[0][0] = 0.5 - 0.1 * 1.0 * 1.0 = 0.4
    CHECK(layer.W[0][0] < originalW00);
}

TEST_CASE("backward updates biases in correct direction") {
    Layer layer = buildAndForward();

    float originalB0 = layer.b[0];
    std::vector<float> delta = {1.0f, 1.0f};
    backward(layer, delta, 0.1f);

    // b = b - lr * delta
    // b[0] = 0.0 - 0.1 * 1.0 = -0.1
    CHECK(layer.b[0] < originalB0);
}

TEST_CASE("backward weight update matches manual calculation") {
    Layer layer = buildAndForward();

    // W[0][0] = 0.5, lastInput[0] = 1.0, delta[0] = 1.0, lr = 0.1
    // dW[0][0] = delta[0] * lastInput[0] = 1.0 * 1.0 = 1.0
    // W[0][0] = 0.5 - 0.1 * 1.0 = 0.4
    std::vector<float> delta = {1.0f, 1.0f};
    backward(layer, delta, 0.1f);

    CHECK(approxEqual(layer.W[0][0], 0.4f));
}

TEST_CASE("backward with zero delta does not change weights") {
    Layer layer = buildAndForward();

    float originalW00 = layer.W[0][0];
    float originalW01 = layer.W[0][1];

    std::vector<float> delta = {0.0f, 0.0f};
    backward(layer, delta, 0.1f);

    CHECK(approxEqual(layer.W[0][0], originalW00));
    CHECK(approxEqual(layer.W[0][1], originalW01));
}

TEST_CASE("backward with zero learning rate does not change weights") {
    Layer layer = buildAndForward();

    float originalW00 = layer.W[0][0];
    std::vector<float> delta = {1.0f, 1.0f};
    backward(layer, delta, 0.0f);

    CHECK(approxEqual(layer.W[0][0], originalW00));
}