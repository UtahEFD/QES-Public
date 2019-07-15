# This file is part of the EnvSim Open Source Environment Simulation codes, which is developed and 
# maintained by the University of Utah and the University of Minnesota Duluth.
#
# Copyright (c) 2018 University of Utah and the Regents of the University of Minnesota
# This software is distributed under the MIT License, expressed in: LICENSE.txt.

CMAKE_MINIMUM_REQUIRED (VERSION 3.9)

# cmake_policy(SET CMP0048 NEW)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/Modules/")
MESSAGE(STATUS "CMake Module Path: ${CMAKE_MODULE_PATH}")

PROJECT(cudaUrb)

SET(CUDAURB_VERSION_INFO -${cudaUrb_Toolkit_VERSION_MAJOR}.${MinVR_Toolkit_VERSION_MINOR})

MESSAGE(STATUS "Building the '" ${CMAKE_BUILD_TYPE} "' version of cudaUrb.")
MESSAGE(STATUS "If you want something different, use -DCMAKE_BUILD_TYPE=Debug, Release, RelWithDebInfo, MinSizeRel.")

#
# This code used C++ 11 and 14 features so enable these on CXX Flags
#
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ----------------------------------------------------------
# Boost
#
# This section tests for Boost support on the system. Boost is
# required as it is used heavily with this code. Boost also allows the
# code to be nicely portable across Operating Systems.
# ----------------------------------------------------------
# SET (Boost_MULTITHREADED TRUE)

if(DEFINED ENV{BOOST_DIR})
    message(STATUS "Setting BOOST_DIR to $ENV{BOOST_DIR}")
    SET(BOOST_ROOT $ENV{BOOST_DIR})
endif()

# If you want to get more info on what the Find Boost macro is
# doing...  
# SET(Boost_DEBUG 1)

# May be necessary to get Win32 version to work
# SET(Boost_USE_STATIC_LIBS TRUE)

FIND_PACKAGE(Boost REQUIRED date_time program_options system thread unit_test_framework chrono timer)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
  MESSAGE(STATUS "Found Boost Libraries in ${Boost_LIBRARY_DIR}, Version ${Boost_VERSION}")
ENDIF()

SET(NETCDF_CXX "YES")
FIND_PACKAGE(NetCDF REQUIRED)
IF(NetCDF_FOUND)
  MESSAGE(STATUS "Found NetCDF: ${NETCDF_INCLUDES}: ${NETCDF_LIBRARIES}, ${NETCDF_INCLUDES_CXX}, ${NETCDF_LIBRARIES_CXX}")
  INCLUDE_DIRECTORIES(${NETCDF_INCLUDES})
  INCLUDE_DIRECTORIES(${NETCDF_INCLUDES_CXX})
ENDIF(NetCDF_FOUND)

# SET(CMAKE_PREFIX_PATH /uufs/chpc.utah.edu/sys/installdir/gdal/2.1.3-c7)
FIND_PACKAGE(GDAL REQUIRED)
IF (${GDAL_FOUND})
  INCLUDE_DIRECTORIES(${GDAL_INCLUDE_DIR})
  MESSAGE(STATUS "Found GDAL: ${GDAL_INCLUDE_DIR}")
ENDIF(${GDAL_FOUND})

SET(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall --std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_61;-rdc=true;" )

FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
#   INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/samples/common/inc)
  LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib64)
  MESSAGE(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)

#
# Find doxygen so we can add a target for building
#
FIND_PACKAGE(Doxygen)
if (DOXYGEN_FOUND)

    # set input and output files
    MESSAGE(STATUS "Doc config file: ${CMAKE_CURRENT_SOURCE_DIR}/docs/DoxygenConfig.in")

    # request to configure the file
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/DoxygenConfig.in
      ${CMAKE_CURRENT_BINARY_DIR}/docs
      @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( urbdoc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

#
#	Unit Testing:
#
# This "unit test" is only a simple test to show the system
# working. It simply returns a true value to indicate that a null
# program has ended OK. I pass on this test really just means the
# cmake system constructed the build environment OK.
enable_testing() # once
include(CTest)


add_subdirectory(util)
add_subdirectory(src)

add_subdirectory(scratch)

add_subdirectory(cudaUrb)

