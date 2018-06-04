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

# ----------------------------------------------------------
# Boost
#
# This section tests for Boost support on the system. Boost is
# required as it is used heavily with this code. Boost also allows the
# code to be nicely portable across Operating Systems.
# ----------------------------------------------------------
# SET (Boost_MULTITHREADED TRUE)

# If you want to get more info on what the Find Boost macro is
# doing...  
# SET(Boost_DEBUG 1)

# Necessary to get Win32 version to work, plus, not a bad idea to link
# statically for our purposes.
SET(Boost_USE_STATIC_LIBS TRUE)

FIND_PACKAGE(Boost REQUIRED date_time program_options system thread unit_test_framework chrono timer)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
  MESSAGE(STATUS "Boost Libs: ${Boost_LIBRARIES}")
ENDIF()

add_subdirectory(src)

add_subdirectory(solverTestCases)

add_subdirectory(test)


