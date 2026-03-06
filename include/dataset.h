#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

struct IrisSample {
    std::vector<float> features;
    std::string label;
};

std::vector<IrisSample> loadCSVFile(const std::string& filePath);

float minMaxNormalization(const float value, const float min, const float max);

#endif