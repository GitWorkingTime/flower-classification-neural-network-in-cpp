#include <iostream>
#include "../include/dataset.h"

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

    return 0;
}