#include <iostream>
#include "../include/dataset.h"

inline std::ostream& operator<<(std::ostream& os, const IrisSample& sample) {
    os << "Features: ";
    for (float f : sample.features) {
        os << f << " ";
    }
    os << "| Label: " << sample.label;
    return os;
}

int main() {
    std::vector<IrisSample> dataset = loadCSVFile("data/Iris.csv");

    std::cout << "Loaded " << dataset.size() << " samples" << std::endl;

    if (dataset.empty()) {
        std::cerr << "Dataset is empty - check file path" << std::endl;
        return 1;
    }

    // safe to access now
    std::cout << "First sample features: ";
    for (float f : dataset[0].features) {
        std::cout << f << " ";
    }
    std::cout << "\nLabel: " << dataset[0].label << std::endl;

    std::vector<IrisSample> normalized = minMaxNormalize(dataset);
    std::cout << "\nOriginal: " << dataset[0] << std::endl;
    std::cout << "Normalized: " << normalized[0] << std::endl;

    return 0;
}