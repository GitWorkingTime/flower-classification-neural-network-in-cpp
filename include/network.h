#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "layer.h"
#include "dataset.h"

// ─── Network ─────────────────────────────────────────────────────────────────

struct Network {
    std::vector<Layer> layers;
};

// ─── Functions ───────────────────────────────────────────────────────────────

// build the network with 3 layers for Iris
// Input(4) → Dense(8, ReLU) → Dense(8, ReLU) → Dense(3, Softmax)
Network createNetwork();

// feed one sample through the full network - returns probability vector
std::vector<float> predict(Network& network, const std::vector<float>& x);

// compute cross entropy loss between predicted and true label
float crossEntropyLoss(const std::vector<float>& yHat, const std::vector<float>& yTrue);

// train for one sample - forward pass + backward pass + weight updates
void trainSample(Network& network, const std::vector<float>& x, const std::vector<float>& yTrue, float lr);

// train for one full epoch over the entire training set
void trainEpoch(Network& network, const std::vector<IrisSample>& trainSet, float lr);

// evaluate accuracy on the test set - returns value between 0.0 and 1.0
float evaluate(Network& network, const std::vector<IrisSample>& testSet);

#endif // NETWORK_H