# This file is part of the EnvSim Open Source Environment Simulation codes, which is developed and 
# maintained by the University of Utah and the University of Minnesota Duluth.
#
# Copyright (c) 2018 University of Utah and the Regents of the University of Minnesota
# This software is distributed under the MIT License, expressed in: LICENSE.txt.

cmake_minimum_required (VERSION 3.9)

# cmake_policy(SET CMP0048 NEW)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/Modules/")

message(STATUS ${CMAKE_MODULE_PATH})

project(cudaUrb)

# set(CUDAURB_VERSION_INFO -${cudaUrb_Toolkit_VERSION_MAJOR}.${MinVR_Toolkit_VERSION_MINOR})

# MESSAGE(STATUS "Building the '" ${CMAKE_BUILD_TYPE} "' version of cudaUrb.")
# MESSAGE(STATUS "If you want something different, use -DCMAKE_BUILD_TYPE=Debug, Release, RelWithDebInfo, MinSizeRel.")

add_subdirectory(src)

add_subdirectory(solverTestCases)


