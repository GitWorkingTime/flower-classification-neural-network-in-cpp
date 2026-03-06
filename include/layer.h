#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "activations.h"
#include "utils.h"

// ─── Activation Type ─────────────────────────────────────────────────────────

enum class ActivationType {
    ReLU,
    Softmax
};

// ─── Layer ───────────────────────────────────────────────────────────────────

struct Layer {
    std::vector<std::vector<float>> W;  // weight matrix [outputSize x inputSize]
    std::vector<float> b;               // bias vector [outputSize]
    std::vector<float> lastInput;       // input received during forward pass - stored for backprop
    std::vector<float> lastZ;           // pre-activation z = Wx + b - stored for backprop
    ActivationType activation;          // type of activation function for this layer
};

// ─── Functions ───────────────────────────────────────────────────────────────

// initialize a layer with random weights using Xavier initialization
Layer createLayer(int inputSize, int outputSize, ActivationType activation);

// forward pass - computes a = f(Wx + b), stores lastInput and lastZ
std::vector<float> forward(Layer& layer, const std::vector<float>& x);

// backward pass - computes gradients, updates weights, returns delta for previous layer
std::vector<float> backward(Layer& layer, const std::vector<float>& delta, float lr);

#endif // LAYER_H