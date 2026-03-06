// activations.h
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>

float relu(float x);
float reluDerivative(float x);
std::vector<float> softmax(const std::vector<float>& x);

#endif // ACTIVATIONS_H