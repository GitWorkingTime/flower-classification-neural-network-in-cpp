#include "doctest.h"
#include "dataset.h"

#include <vector>

TEST_CASE("Iris-setosa encodes to [1, 0, 0]") {
    std::vector<float> result = oneHotEncode("Iris-setosa");

    CHECK(result.size() == 3);
    CHECK(result[0] == 1.0f);
    CHECK(result[1] == 0.0f);
    CHECK(result[2] == 0.0f);
}

TEST_CASE("Iris-versicolor encodes to [0, 1, 0]") {
    std::vector<float> result = oneHotEncode("Iris-versicolor");

    CHECK(result.size() == 3);
    CHECK(result[0] == 0.0f);
    CHECK(result[1] == 1.0f);
    CHECK(result[2] == 0.0f);
}

TEST_CASE("Iris-virginica encodes to [0, 0, 1]") {
    std::vector<float> result = oneHotEncode("Iris-virginica");

    CHECK(result.size() == 3);
    CHECK(result[0] == 0.0f);
    CHECK(result[1] == 0.0f);
    CHECK(result[2] == 1.0f);
}

TEST_CASE("one hot vector always has exactly one 1") {
    std::vector<std::string> labels = {
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    };

    for (const std::string& label : labels) {
        std::vector<float> result = oneHotEncode(label);

        int countOnes = 0;
        for (float val : result) {
            if (val == 1.0f) countOnes++;
        }

        CHECK(countOnes == 1);
    }
}

TEST_CASE("one hot vector always has exactly two 0s") {
    std::vector<std::string> labels = {
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    };

    for (const std::string& label : labels) {
        std::vector<float> result = oneHotEncode(label);

        int countZeros = 0;
        for (float val : result) {
            if (val == 0.0f) countZeros++;
        }

        CHECK(countZeros == 2);
    }
}

TEST_CASE("unknown label returns empty vector") {
    std::vector<float> result = oneHotEncode("Iris-unknown");

    CHECK(result.empty());
}