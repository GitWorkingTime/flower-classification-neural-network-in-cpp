#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>

// dot product of two vectors
float dot(const std::vector<float>& a, const std::vector<float>& b);

// matrix-vector multiplication
// matrix is represented as a vector of rows
std::vector<float> matVecMul(const std::vector<std::vector<float>>& W, const std::vector<float>& x);

// element-wise vector addition
std::vector<float> vecAdd(const std::vector<float>& a, const std::vector<float>& b);

// element-wise vector multiplication
std::vector<float> vecMul(const std::vector<float>& a, const std::vector<float>& b);

// transpose a matrix
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& W);

// scalar multiplication on a vector
std::vector<float> scalarMul(const std::vector<float>& a, float scalar);

#endif