#include "network.h"
#include "layer.h"
#include "dataset.h"
#include "activations.h"
#include "utils.h"

#include <cmath>
#include <iostream>
#include <algorithm>

Network createNetwork() {
    Network network;

    network.layers.push_back(createLayer(4, 8, ActivationType::ReLU));
    network.layers.push_back(createLayer(8, 8, ActivationType::ReLU));
    network.layers.push_back(createLayer(8, 3, ActivationType::Softmax));

    return network;
}

std::vector<float> predict(Network& network, const std::vector<float>& x) {
    std::vector<float> current = x;

    for (Layer& layer : network.layers) {
        current = forward(layer, current);
    }

    return current;
}

float crossEntropyLoss(const std::vector<float>& yHat, const std::vector<float>& yTrue) {
    float loss = 0.0f;
    const float epsilon = 1e-7f;

    for (std::size_t i = 0; i < yHat.size(); ++i) {
        float clipped = std::max(yHat[i], epsilon);
        loss += yTrue[i] * std::log(clipped);
    }

    return -loss;
}

void trainSample(Network& network, const std::vector<float>& x, const std::vector<float>& yTrue, float lr) {
    // step 1 - forward pass
    std::vector<float> yHat = predict(network, x);

    // step 2 - output layer error: delta = yHat - yTrue
    std::vector<float> delta(yHat.size());
    for (std::size_t i = 0; i < yHat.size(); ++i) {
        delta[i] = yHat[i] - yTrue[i];
    }

    // step 3 - backward pass right to left
    for (int l = network.layers.size() - 1; l >= 0; --l) {
        delta = backward(network.layers[l], delta, lr);
    }
}

void trainEpoch(Network& network, const std::vector<IrisSample>& trainSet, float lr) {
    for (const IrisSample& sample : trainSet) {
        std::vector<float> yTrue = oneHotEncode(sample.label);
        trainSample(network, sample.features, yTrue, lr);
    }
}

float evaluate(Network& network, const std::vector<IrisSample>& testSet) {
    int correct = 0;

    for (const IrisSample& sample : testSet) {
        std::vector<float> yHat  = predict(network, sample.features);
        std::vector<float> yTrue = oneHotEncode(sample.label);

        int predicted = std::max_element(yHat.begin(),  yHat.end())  - yHat.begin();
        int actual    = std::max_element(yTrue.begin(), yTrue.end()) - yTrue.begin();

        if (predicted == actual) correct++;
    }

    return (float)correct / (float)testSet.size();
}