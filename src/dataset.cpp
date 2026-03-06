// Imports
#include "../include/dataset.h"

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

float minMaxNormalization(const float value, const float min, const float max) {
    float numerator = value - min;
    float denominator = max - min;
    float result = numerator/denominator;

    return result;
}