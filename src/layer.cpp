#include "layer.h"
#include "activations.h"
#include "utils.h"

#include <random>
#include <cmath>
#include <iostream>

Layer createLayer(int inputSize, int outputSize, ActivationType activation) {
    Layer layer;

    std::random_device rd;
    std::mt19937 rng(rd());

    float limit = 1.0f / std::sqrt((float)inputSize);
    std::uniform_real_distribution<float> dist(-limit, limit);

    layer.W.resize(outputSize, std::vector<float>(inputSize));
    for (int row = 0; row < outputSize; ++row) {
        for (int col = 0; col < inputSize; ++col) {
            layer.W[row][col] = dist(rng);
        }
    }

    layer.b.resize(outputSize, 0.0f);
    layer.activation = activation;

    return layer;
}

std::vector<float> forward(Layer& layer, const std::vector<float>& x) {
    layer.lastInput = x;

    std::vector<float> z = vecAdd(matVecMul(layer.W, x), layer.b);

    layer.lastZ = z;

    if (layer.activation == ActivationType::Softmax) {
        return softmax(z);
    }

    std::vector<float> a;
    for (float val : z) {
        a.push_back(relu(val));
    }

    return a;
}

std::vector<float> backward(Layer& layer, const std::vector<float>& delta, float lr) {

    // relaxed clip value - allows larger gradients early in training
    const float clipValue = 5.0f;

    std::vector<float> clippedDelta = delta;
    for (float& d : clippedDelta) {
        if (d >  clipValue) d =  clipValue;
        if (d < -clipValue) d = -clipValue;
    }

    // ── Part 1 - weight gradients via outer product ──────────────────────────
    std::vector<std::vector<float>> dW(
        layer.W.size(),
        std::vector<float>(layer.lastInput.size(), 0.0f)
    );

    for (std::size_t i = 0; i < clippedDelta.size(); ++i) {
        for (std::size_t j = 0; j < layer.lastInput.size(); ++j) {
            dW[i][j] = clippedDelta[i] * layer.lastInput[j];
        }
    }

    // ── Part 2 - update weights and biases ───────────────────────────────────
    for (std::size_t i = 0; i < layer.W.size(); ++i) {
        for (std::size_t j = 0; j < layer.W[i].size(); ++j) {
            layer.W[i][j] -= lr * dW[i][j];
        }
    }

    for (std::size_t i = 0; i < layer.b.size(); ++i) {
        layer.b[i] -= lr * clippedDelta[i];
    }

    // ── Part 3 - propagate error signal to previous layer ───────────────────
    std::vector<float> deltaPrev = matVecMul(transpose(layer.W), clippedDelta);

    // only apply ReLU derivative for hidden layers
    if (layer.activation == ActivationType::ReLU) {
        for (std::size_t i = 0; i < deltaPrev.size(); ++i) {
            deltaPrev[i] *= reluDerivative(layer.lastZ[i]);
        }
    }

    return deltaPrev;
}