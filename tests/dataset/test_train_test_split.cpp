#include "doctest.h"
#include "dataset.h"

#include <algorithm>
#include <vector>

TEST_CASE("train and test sets cover the full dataset") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-setosa"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-setosa"},
        {{13.0f, 14.0f, 15.0f, 16.0f}, "Iris-versicolor"},
        {{17.0f, 18.0f, 19.0f, 20.0f}, "Iris-versicolor"},
        {{21.0f, 22.0f, 23.0f, 24.0f}, "Iris-virginica"},
        {{25.0f, 26.0f, 27.0f, 28.0f}, "Iris-virginica"},
        {{29.0f, 30.0f, 31.0f, 32.0f}, "Iris-virginica"},
        {{33.0f, 34.0f, 35.0f, 36.0f}, "Iris-virginica"},
        {{37.0f, 38.0f, 39.0f, 40.0f}, "Iris-virginica"}
    };

    std::vector<IrisSample> trainSet;
    std::vector<IrisSample> testSet;
    trainTestSplit(dataset, trainSet, testSet, 0.2f);

    CHECK(trainSet.size() + testSet.size() == dataset.size());
}

TEST_CASE("split ratio produces correct sizes") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-setosa"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-setosa"},
        {{13.0f, 14.0f, 15.0f, 16.0f}, "Iris-versicolor"},
        {{17.0f, 18.0f, 19.0f, 20.0f}, "Iris-versicolor"},
        {{21.0f, 22.0f, 23.0f, 24.0f}, "Iris-virginica"},
        {{25.0f, 26.0f, 27.0f, 28.0f}, "Iris-virginica"},
        {{29.0f, 30.0f, 31.0f, 32.0f}, "Iris-virginica"},
        {{33.0f, 34.0f, 35.0f, 36.0f}, "Iris-virginica"},
        {{37.0f, 38.0f, 39.0f, 40.0f}, "Iris-virginica"}
    };

    std::vector<IrisSample> trainSet;
    std::vector<IrisSample> testSet;
    trainTestSplit(dataset, trainSet, testSet, 0.2f);

    // 10 samples, 0.2 ratio → 8 train, 2 test
    CHECK(trainSet.size() == 8);
    CHECK(testSet.size() == 2);
}

TEST_CASE("original dataset is not modified after split") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-setosa"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-setosa"},
        {{13.0f, 14.0f, 15.0f, 16.0f}, "Iris-versicolor"},
        {{17.0f, 18.0f, 19.0f, 20.0f}, "Iris-versicolor"}
    };

    std::vector<IrisSample> trainSet;
    std::vector<IrisSample> testSet;
    trainTestSplit(dataset, trainSet, testSet, 0.2f);

    CHECK(dataset.size() == 5);
    CHECK(dataset[0].features[0] == 1.0f);
}

TEST_CASE("no sample is lost or duplicated after split") {
    std::vector<IrisSample> dataset = {
        {{1.0f, 2.0f, 3.0f, 4.0f}, "Iris-setosa"},
        {{5.0f, 6.0f, 7.0f, 8.0f}, "Iris-setosa"},
        {{9.0f, 10.0f, 11.0f, 12.0f}, "Iris-setosa"},
        {{13.0f, 14.0f, 15.0f, 16.0f}, "Iris-versicolor"},
        {{17.0f, 18.0f, 19.0f, 20.0f}, "Iris-versicolor"},
        {{21.0f, 22.0f, 23.0f, 24.0f}, "Iris-virginica"},
        {{25.0f, 26.0f, 27.0f, 28.0f}, "Iris-virginica"},
        {{29.0f, 30.0f, 31.0f, 32.0f}, "Iris-virginica"},
        {{33.0f, 34.0f, 35.0f, 36.0f}, "Iris-virginica"},
        {{37.0f, 38.0f, 39.0f, 40.0f}, "Iris-virginica"}
    };

    std::vector<IrisSample> trainSet;
    std::vector<IrisSample> testSet;
    trainTestSplit(dataset, trainSet, testSet, 0.2f);

    std::vector<float> allValues;
    for (const IrisSample& s : trainSet) allValues.push_back(s.features[0]);
    for (const IrisSample& s : testSet)  allValues.push_back(s.features[0]);

    std::vector<float> originalValues;
    for (const IrisSample& s : dataset) originalValues.push_back(s.features[0]);

    std::sort(allValues.begin(), allValues.end());
    std::sort(originalValues.begin(), originalValues.end());

    CHECK(allValues == originalValues);
}