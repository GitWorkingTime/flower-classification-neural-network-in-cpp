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
std::vector<IrisSample> minMaxNormalize(const std::vector<IrisSample>& dataset);

void trainTestSplit(
    const std::vector<IrisSample>& dataset,
    std::vector<IrisSample>& trainSet,
    std::vector<IrisSample>& testSet,
    float testRatio = 0.2f
);

std::vector<float> oneHotEncode(const std::string& label);

#endif