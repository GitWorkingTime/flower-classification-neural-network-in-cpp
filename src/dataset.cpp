// Imports
#include "../include/dataset.h"
#include <algorithm>
#include <random>

std::vector<IrisSample> loadCSVFile(const std::string& filePath) {
    std::vector<IrisSample> dataset;
    std::ifstream file(filePath);

    // Edge Case: File isn't found or couldn't be opened
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file" << std::endl;
        return dataset;
    }

    // Read the file line by line
    std::string line = "";

    // Skip the header row
    std::getline(file, line);

    while (std::getline(file, line)) {

        // Skip a line if it's empty
        if (line.empty()) {
            continue;
        }

        // Apply "stream behaviour" to the std::string
        std::stringstream ss(line);
        std::string token = "";
        IrisSample sample;

        // Skip the ID column
        std::getline(ss, token, ',');

        // Read the 4 features
        for (std::size_t i = 0; i < 4; ++i) {
            std::getline(ss, token, ',');
            sample.features.push_back(std::stof(token));
        }

        // Read the label
        std::getline(ss, token);

        // Handling \r\n on Windows
        if (!token.empty() && token.back() == '\r') {
            token.pop_back();
        }
        sample.label = token;
        dataset.push_back(sample);
    }

    // Clean up
    file.close();
    return dataset;
}

std::vector<IrisSample> minMaxNormalize(const std::vector<IrisSample>& dataset) {
    std::vector<IrisSample> normalized = dataset;
    for (std::size_t feat = 0; feat < 4; ++feat ) {
        float min = dataset[0].features[feat];
        float max = dataset[0].features[feat];

        for (const IrisSample& sample : normalized) {
            if (sample.features[feat] < min) {
                min = sample.features[feat];
            }

            if (sample.features[feat] > max ) {
                max = sample.features[feat];
            }
        }

        for (IrisSample& sample : normalized) {
            if (min == max) {
                sample.features[feat] = 0.0f;
                continue;
            }
            sample.features[feat] = (sample.features[feat] - min) / (max - min);
        }
    }

    return normalized;
}

void trainTestSplit(
    const std::vector<IrisSample>& dataset,
    std::vector<IrisSample>& trainSet,
    std::vector<IrisSample>& testSet,
    float testRatio
) {
    std::vector<IrisSample> copy = dataset;
    std::random_device rd;
    std::mt19937 rng(rd());
    
    std::shuffle(copy.begin(), copy.end(), rng);
    std::size_t splitIndex = copy.size() * (1.0f - testRatio);

    for (std::size_t i = 0; i < splitIndex; ++i ) {
        trainSet.push_back(copy[i]);
    }

    for (std::size_t i = splitIndex; i < copy.size(); ++i ) {
        testSet.push_back(copy[i]);
    }
}