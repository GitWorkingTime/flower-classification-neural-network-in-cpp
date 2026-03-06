#include "dataset.h"
#include "network.h"

#include <iostream>
#include <vector>

int main() {

    // ── Step 1 - Load Data ───────────────────────────────────────────────────

    std::vector<IrisSample> dataset = loadCSVFile("data/Iris.csv");

    if (dataset.empty()) {
        std::cerr << "Error: dataset is empty - check file path" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << dataset.size() << " samples" << std::endl;

    // ── Step 2 - Normalize ───────────────────────────────────────────────────

    std::vector<IrisSample> normalized = minMaxNormalize(dataset);

    std::cout << "Normalized " << normalized.size() << " samples" << std::endl;

    // ── Step 3 - Train/Test Split ────────────────────────────────────────────

    std::vector<IrisSample> trainSet;
    std::vector<IrisSample> testSet;
    trainTestSplit(normalized, trainSet, testSet, 0.2f);

    std::cout << "Train samples: " << trainSet.size() << std::endl;
    std::cout << "Test samples:  " << testSet.size()  << std::endl;

    // ── Step 4 - Build Network ───────────────────────────────────────────────

    Network network = createNetwork();

    std::cout << "Network created" << std::endl;
    std::cout << "Architecture: Input(4) -> Dense(8, ReLU) -> Dense(8, ReLU) -> Dense(3, Softmax)" << std::endl;

    // ── Step 5 - Training Loop ───────────────────────────────────────────────

    const int   epochs   = 2000;
    const float lr       = 0.01f;
    const float decay    = 0.02f;

    std::cout << "\nTraining for " << epochs << " epochs with initial learning rate " << lr << "\n" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {

        // learning rate decay - reduces lr gradually to stabilize training
        float currentLr = lr / (1.0f + decay * epoch);

        trainEpoch(network, trainSet, currentLr);

        if (epoch % 50 == 0) {
            float accuracy = evaluate(network, testSet);
            std::cout << "Epoch " << epoch
                      << " \t lr: "       << currentLr
                      << " \t accuracy: " << accuracy * 100.0f << "%" << std::endl;
        }
    }

    // ── Step 6 - Final Evaluation ────────────────────────────────────────────

    float finalAccuracy = evaluate(network, testSet);
    std::cout << "\nFinal accuracy: " << finalAccuracy * 100.0f << "%" << std::endl;

    return 0;
}