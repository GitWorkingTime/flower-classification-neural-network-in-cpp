#include "../include/utils.h"

// dot product of two vectors
float dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: dot product dimension mismatch - "
                  << "a (" << a.size() << ") "
                  << "must match b (" << b.size() << ")" << std::endl;
        return 0.0f;
    }

    float sum = 0.0f;
    if (a.size() != b.size()) {
        return 0.0f;
    } else {
        for (std::size_t i = 0; i < a.size(); ++i ) {
            sum += a[i] * b[i];
        }
    }

    return sum;
}

// matrix-vector multiplication
// matrix is represented as a vector of rows
std::vector<float> matVecMul(const std::vector<std::vector<float>>& W, const std::vector<float>& x) {
    if (W.empty()) {
        std::cerr << "Error: matVecMul called with empty matrix" << std::endl;
        return {};
    }

    if (W[0].size() != x.size()) {
        std::cerr << "Error: matVecMul dimension mismatch - "
                  << "matrix columns (" << W[0].size() << ") "
                  << "must match vector size (" << x.size() << ")" << std::endl;
        return {};
    }

    std::vector<float> result;
    for (const std::vector<float>& row : W) {
        result.push_back(dot(row, x));
    }

    return result;
}

// element-wise vector addition
std::vector<float> vecAdd(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result = {};
    if (a.size() != b.size()) {
        return {};
    } else {
        for (std::size_t i = 0; i < a.size(); ++i) {
            result.push_back(a[i] + b[i]);
        }
    }

    return result;
}

// element-wise vector multiplication
std::vector<float> vecMul(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result;

    if (a.size() != b.size()) {
        return {};
    } 

    for (std::size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] * b[i]);
    }

    return result;
}

// transpose a matrix
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& W) {
    if (W.empty()) {
        return {};
    }

    std::size_t rows = W.size();
    std::size_t cols = W[0].size();

    // initialize result as cols x rows filled with 0
    std::vector<std::vector<float>> result(cols, std::vector<float>(rows, 0.0f));

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result[j][i] = W[i][j];
        }
    }

    return result;
}

// scalar multiplication on a vector
std::vector<float> scalarMul(const std::vector<float>& a, float scalar) {
    if (a.empty()) {
        std::cerr << "Error: scalarMul called with empty vector" << std::endl;
        return {};
    }

    std::vector<float> result = {};
    if (a.size() == 0) {
        return {};
    } else {
        for (std::size_t i = 0; i < a.size() ; ++i) {
            result.push_back(a[i] * scalar);
        }
    }

    return result;
}