cmake_minimum_required(VERSION 3.24)

project(iranges
    VERSION 1.0.0
    DESCRIPTION "Building the rds shared library"
    LANGUAGES CXX)

find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(iranges
    src/coverage.cpp
    src/interranges.cpp
    src/init.cpp
)

set_property(TARGET iranges PROPERTY CXX_STANDARD 17)

target_link_libraries(iranges PRIVATE pybind11::pybind11)

set_target_properties(iranges PROPERTIES
    OUTPUT_NAME lib_iranges
    PREFIX ""
)
