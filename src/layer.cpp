#include "layer.h"
#include "activations.h"
#include "utils.h"

#include <random>
#include <cmath>
#include <iostream>

Layer createLayer(int inputSize, int outputSize, ActivationType activation) {
    Layer layer;

    // set up the RNG
    std::random_device rd;
    std::mt19937 rng(rd());

    // Xavier range
    float limit = 1.0f / std::sqrt((float)inputSize);
    std::uniform_real_distribution<float> dist(-limit, limit);

    // initialize W as outputSize x inputSize matrix
    layer.W.resize(outputSize, std::vector<float>(inputSize));
    for (int row = 0; row < outputSize; ++row) {
        for (int col = 0; col < inputSize; ++col) {
            layer.W[row][col] = dist(rng);
        }
    }

    // initialize b as outputSize vector of zeros
    layer.b.resize(outputSize, 0.0f);

    // store activation type
    layer.activation = activation;

    return layer;
}

std::vector<float> forward(Layer& layer, const std::vector<float>& x) {
    // store input for backprop
    layer.lastInput = x;

    // z = Wx + b
    std::vector<float> z = vecAdd(matVecMul(layer.W, x), layer.b);

    // store pre-activation for backprop
    layer.lastZ = z;

    // apply activation function
    if (layer.activation == ActivationType::Softmax) {
        return softmax(z);
    }

    // apply relu element-wise
    std::vector<float> a;
    for (float val : z) {
        a.push_back(relu(val));
    }

    return a;
}

std::vector<float> backward(Layer& layer, const std::vector<float>& delta, float lr) {
    // ── Part 1 - compute weight gradients via outer product ──────────────────
    // dW[i][j] = delta[i] * lastInput[j]
    std::vector<std::vector<float>> dW(
        layer.W.size(),
        std::vector<float>(layer.lastInput.size(), 0.0f)
    );

    for (std::size_t i = 0; i < delta.size(); ++i) {
        for (std::size_t j = 0; j < layer.lastInput.size(); ++j) {
            dW[i][j] = delta[i] * layer.lastInput[j];
        }
    }

    // ── Part 2 - update weights and biases ───────────────────────────────────
    for (std::size_t i = 0; i < layer.W.size(); ++i) {
        for (std::size_t j = 0; j < layer.W[i].size(); ++j) {
            layer.W[i][j] -= lr * dW[i][j];
        }
    }

    for (std::size_t i = 0; i < layer.b.size(); ++i) {
        layer.b[i] -= lr * delta[i];
    }

    // ── Part 3 - propagate error signal to previous layer ───────────────────
    // W^T · delta
    std::vector<float> deltaPrev = matVecMul(transpose(layer.W), delta);

    // element-wise multiply by ReLU derivative of lastZ
    for (std::size_t i = 0; i < deltaPrev.size(); ++i) {
        deltaPrev[i] *= reluDerivative(layer.lastZ[i]);
    }

    return deltaPrev;
}