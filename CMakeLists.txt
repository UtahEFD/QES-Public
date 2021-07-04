# This file is part of the EnvSim Open Source Environment Simulation codes, which is developed and 
# maintained by the University of Utah and the University of Minnesota Duluth.
#
# Copyright (c) 2018 University of Utah and the Regents of the University of Minnesota
# This software is distributed under the MIT License, expressed in: LICENSE.txt.

CMAKE_MINIMUM_REQUIRED (VERSION 3.9)

# cmake_policy(SET CMP0048 NEW)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/Modules/")
MESSAGE(STATUS "CMake Module Path: ${CMAKE_MODULE_PATH}")

PROJECT(qesWinds)

SET(QESWINDS_VERSION_INFO -${qesWinds_Toolkit_VERSION_MAJOR}.${MinVR_Toolkit_VERSION_MINOR})

MESSAGE(STATUS "Building the '" ${CMAKE_BUILD_TYPE} "' version of qesWinds.")
MESSAGE(STATUS "If you want something different, use -DCMAKE_BUILD_TYPE=Debug, Release, RelWithDebInfo, MinSizeRel.")

# Force VERBOSE Output -- used in all branches except master
# Can be commented out in master branches.
SET(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
  
# 
# Detect COMPILER
#  This may be useful for helping us determine the level of C++
#  support available.  For instance on GCC 4.4.7, -std=c++11 is not
#  available, but -std=c++0x is available.  Newer compilers
#  can handle the -std=c++11, -std=c++14, etc...
#
include(CheckCXXCompilerFlag)
MESSAGE(STATUS "Compiling with ${CMAKE_CXX_COMPILER_ID}, Version: ${CMAKE_CXX_COMPILER_VERSION}")
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

  #
  # This code used C++ 11 and 14 features so enable these on CXX Flags
  #
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()


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

FIND_PACKAGE(Boost REQUIRED filesystem date_time program_options system thread unit_test_framework chrono timer)
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

# SET(CMAKE_PREFIX_PATH /uufs/chpc.utah.edu/sys/installdir/gdal/2.4.0)
FIND_PACKAGE(GDAL REQUIRED)
IF (${GDAL_FOUND})
  INCLUDE_DIRECTORIES(${GDAL_INCLUDE_DIR})
  MESSAGE(STATUS "Found GDAL: ${GDAL_INCLUDE_DIR}")
ENDIF(${GDAL_FOUND})

SET(CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_52;-rdc=true;" )

#use fast math option is required for OptiX
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_61;-dlink;-rdc=true;--use_fast_math;--keep-device-functions")

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
# Allow the compilation to override including OptiX if desired
#
OPTION(USE_OPTIX_SUPPORT "Determines if OptiX functionality is compiled into the code base" ON)

#
# if CUDA 10.2 or higher is found, then OptiX support can be enabled, so search for it.
#
MESSAGE(STATUS "Using CUDA Version: ${CUDA_VERSION}")
if (${CUDA_VERSION} VERSION_GREATER_EQUAL 10.2)
  MESSAGE(STATUS "Found CUDA 10.2 or higher! Now searching for OptiX support...")
  
  #
  # NVIDIA OptiX
  #   current version used in project: 7.0
  #
  FIND_PACKAGE(OptiX 7.0)

  if(NOT OptiX_INCLUDE)

    MESSAGE(STATUS "  OptiX is used for accelerating some computations with this project. If you wish to run the project with OptiX GPU acceleration, please look into installing NVIDIA CUDA 10.2 or higher and NVIDIA OptiX 7.0 or higher.")    
    SET(HAS_OPTIX_SUPPORT OFF BOOL "Determines if OptiX functionality is compiled into the code base")
    REMOVE_DEFINITIONS(-DHAS_OPTIX)

  else()

    MESSAGE(STATUS "Found NVIDIA OptiX 7.0")

    # Set the preprocessor definitions to conditonally compile the optix code
    SET(HAS_OPTIX_SUPPORT ON BOOL "Determines if OptiX functionality is compiled into the code base")
    ADD_DEFINITIONS(-DHAS_OPTIX)

    MESSAGE(STATUS "Using OptiX Include location: ${OptiX_INCLUDE}")
    include_directories(${OptiX_INCLUDE})

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ptx") #ptx files located here
    add_definitions(-DPTX_DIR="${CMAKE_BINARY_DIR}/ptx") #PTX_DIR
    set(PTX_DIR "${CMAKE_BINARY_DIR}/ptx")
    set(CUDA_GENERATED_OUTPUT_DIR ${PTX_DIR})

    find_program(BIN2C bin2c DOC "Path to the cuda-sdk bin2c executable.")

    # this macro defines cmake rules that execute the following four steps:
    # 1) compile the given cuda file ${cuda_file} to an intermediary PTX file
    # 2) use the 'bin2c' tool (that comes with CUDA) to
    #    create a second intermediary (.c-)file which defines a const string variable
    #    (named '${c_var_name}') whose (constant) value is the PTX output
    #    from the previous step.
    # 3) compile the given .c file to an intermediary object file (why thus has
    #    that PTX string 'embedded' as a global constant.
    # 4) assign the name of the intermediary .o file to the cmake variable
    #    'output_var', which can then be added to cmake targets.
    macro(cuda_compile_and_embed output_var cuda_file)
      set(c_var_name ${output_var})
      cuda_compile_ptx(ptx_files ${cuda_file})
      list(GET ptx_files 0 ptx_file)
      set(embedded_file ${ptx_file}_embedded.c)
      #  message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
      add_custom_command(
	OUTPUT ${embedded_file}
	COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
	DEPENDS ${ptx_file}
	COMMENT "compiling (and embedding ptx from) ${cuda_file}"
	)
      set(${output_var} ${embedded_file})
    endmacro()

  endif()  

else()
  MESSAGE(STATUS "Did NOT find CUDA 10.2 or higher!  Disabling NVIDIA OptiX acceleration.")
  MESSAGE(STATUS "  OptiX is used for accelerating some computations with this project. If you wish to run the project with OptiX GPU acceleration, please look into installing NVIDIA CUDA 10.2 or higher and NVIDIA OptiX 7.0 or higher.")    
  # Disable OptiX Support
  SET(HAS_OPTIX_SUPPORT OFF BOOL "Determines if OptiX functionality is compiled into the code base")
  REMOVE_DEFINITIONS(-DHAS_OPTIX)
endif()

MESSAGE(STATUS "CUDA/OptiX Configuration Complete.")

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
    add_custom_target( windsdoc
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

add_subdirectory(qesWinds)
add_subdirectory(qesCanopy)
add_subdirectory(tests)

option(ENABLE_ALL_TESTS "Enables all tests for an extensive test run" OFF) # will not override CPU/GPU selection
option(ENABLE_SANITY_TESTS "Enables tests that simply check if qesWinds runs on input w/o runtime errors" ON)
option(ENABLE_LONG_SANITY_TESTS "Enables the longer sanity tests" OFF)
option(ENABLE_CPU_TESTS "Enable CPU tests" OFF)
option(ENABLE_GPU_TESTS "Enable GPU tests" ON)
