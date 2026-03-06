cmake_minimum_required(VERSION 3.10)
project(iris_nn)

set(CMAKE_CXX_STANDARD 17)

# main executable
add_executable(iris_nn
    src/main.cpp
    src/dataset.cpp
)
target_include_directories(iris_nn PRIVATE include)

# test executable
add_executable(run_tests
    tests/test_normalization.cpp
    tests/test_train_test_split.cpp
    src/dataset.cpp
)
target_include_directories(run_tests PRIVATE include tests)