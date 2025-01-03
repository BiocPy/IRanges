cmake_minimum_required(VERSION 3.24)

project(iranges
    VERSION 1.0.0
    DESCRIPTION "Building the rds shared library"
    LANGUAGES CXX)

# Defining the targets.
set(TARGET iranges)

find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(${TARGET}
    src/iranges.cpp
)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)

target_link_libraries(${TARGET} PRIVATE pybind11::pybind11)

set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME lib_iranges
    PREFIX ""
)
