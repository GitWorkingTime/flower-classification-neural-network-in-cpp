// activations.cpp
#include "activations.h"
#include <cmath>
#include <algorithm>

float relu(float x) {
    return std::max(0.0f, x);
}

float reluDerivative(float x) {
    if (x > 0.0f) {
        return 1.0f;
    } else {
        return 0.0f;
    }
}

std::vector<float> softmax(const std::vector<float>& x) {
    if (x.empty()) {
        return {};
    }

    std::vector<float> transformed = x;
    float sum = 0.0f;
    for (std::size_t i = 0; i < transformed.size(); ++i ) {
        transformed[i] = std::exp(transformed[i]);
        sum += transformed[i];
    }

    for (std::size_t i = 0; i < transformed.size(); ++i ) {
        transformed[i] = transformed[i] / sum;
    }

    return transformed;
}