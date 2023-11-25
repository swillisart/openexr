# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenEXR Project.
cmake_minimum_required(VERSION 3.4)
project(
    PyOpenEXR
    VERSION 0.0.1
    LANGUAGES CXX
)
find_package(OpenEXR REQUIRED)

# PYBIND11 --------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
  GIT_TAG "master"
  OVERRIDE_FIND_PACKAGE
)

# The following is automatically redirected to FetchContent_MakeAvailable(Catch2)
find_package(pybind11)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(SOURCES
    PyOpenEXR.cpp
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(PyOpenEXR PRIVATE OpenEXR::OpenEXR)
