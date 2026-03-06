cmake_minimum_required(VERSION 3.10)
project(iris_nn)

set(CMAKE_CXX_STANDARD 17)

add_executable(iris_nn
    src/main.cpp
    src/dataset.cpp
    src/utils.cpp
)
target_include_directories(iris_nn PRIVATE include)

add_executable(run_tests
    tests/dataset/test_normalization.cpp
    tests/dataset/test_train_test_split.cpp
    tests/dataset/test_one_hot_encode.cpp
    tests/utils/test_dot.cpp
    tests/utils/test_mat_vec_mul.cpp
    tests/utils/test_vec_add.cpp
    tests/utils/test_vec_mul.cpp
    tests/utils/test_transpose.cpp
    tests/utils/test_scalar_mul.cpp
    tests/activations/test_relu.cpp
    tests/activations/test_softmax.cpp
    src/dataset.cpp
    src/utils.cpp
    src/activations.cpp
)
target_include_directories(run_tests PRIVATE include tests)