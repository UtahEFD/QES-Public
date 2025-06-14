# ====================================================================================
# Quick Environmental Simulation (QES)
#
# This file is part of the QES Open Source framework, which is developed and
# maintained by the University of Utah and the University of Minnesota Duluth.
#
# Copyright (c) 2023 University of Utah and the Regents of the University of Minnesota
# This software is distributed under the MIT License, expressed in: LICENSE.txt.
# ====================================================================================

CMAKE_MINIMUM_REQUIRED (VERSION 3.18)
CMAKE_POLICY(SET CMP0074 NEW)
# cmake_policy(SET CMP0048 NEW)

# This creates a variable we could use for tacking on to various folders
# or install locations based on the version info.  It pulls the Verson information
# from CMake based on the Version number provided in the PROJECT command.
PROJECT(QES VERSION 2.3.0 LANGUAGES CXX)

# The following will pass the version into the code as a preprocessor variable
SET(QES_VERSION_INFO v${QES_VERSION_MAJOR}.${QES_VERSION_MINOR}.${QES_VERSION_PATCH})
ADD_DEFINITIONS(-DQES_VERSION_INFO="${QES_VERSION_INFO}")
MESSAGE(STATUS "Building QES ${QES_VERSION_INFO}")

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/Modules/")
MESSAGE(STATUS "CMake Module Path: ${CMAKE_MODULE_PATH}")


# use some variables to acquire lists of install targets
set(QES_INSTALL_EXECUTABLES "")


# ----------------------------------------------------------
# BUILD OPTIONS
# ----------------------------------------------------------
option(ENABLE_DEV_MODE "Enable developer mode - turns on important flags for developing and debugging." OFF)
option(ENABLE_OPENMP "Enable multithreaded support with OpenMP." OFF)
option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(ENABLE_TESTS "Enable Testing suite" OFF)

# ----------------------------------------------------------
# CLANG TIDY
# ----------------------------------------------------------
if(ENABLE_CLANG_TIDY)
  find_program(CLANGTIDY clang-tidy)
  if(CLANGTIDY)
    set(CMAKE_C_CLANG_TIDY ${CLANGTIDY})
    set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
  else()
    message(SEND_ERROR "clang-tidy could not be located.")
  endif()
endif()


# ----------------------------------------------------------
# Detect COMPILER
#  This may be useful for helping us determine the level of C++
#  support available.  For instance on GCC 4.4.7, -std=c++11 is not
#  available, but -std=c++0x is available.  Newer compilers
#  can handle the -std=c++11, -std=c++14, etc...
# ----------------------------------------------------------
include(CheckCXXCompilerFlag)
MESSAGE(STATUS "Compiling with ${CMAKE_CXX_COMPILER_ID}, Version: ${CMAKE_CXX_COMPILER_VERSION}")

#if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#   MESSAGE(STATUS "MSVC")
#   CHECK_CXX_COMPILER_FLAG("/std:c++17" COMPILER_SUPPORTS_CXX17)
#else()
#   CHECK_CXX_COMPILER_FLAG("--std=c++17" COMPILER_SUPPORTS_CXX17)
#endif()

#if(COMPILER_SUPPORTS_CXX17)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

#
# This code used C++ 17 features so enable these on CXX Flags
# Portable way to do this.
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#else()
#  message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++17 support.")
#endif()

# ----------------------------------------------------------
# OPENMP
# ----------------------------------------------------------
FIND_PACKAGE(OpenMP)
if (ENABLE_OPENMP)
  MESSAGE(STATUS "OpenMP Found: ${OpenMP_FOUND}, ${OpenMP_CXX_FLAGS}, ${OpenMP_CXX_LIBRARIES}")

  If (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    MESSAGE(STATUS "Enabling OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # using CLANG (MACOS)
    if(APPLE)
      MESSAGE(STATUS "Enabling OpenMP")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${OpenMP_CXX_INCLUDE_DIR}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  # -Xpreprocessor -fopenmp")
    else()
      MESSAGE(STATUS "Enabling OpenMP")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    #MESSAGE(WARNING "CLANG support of openmp limitied or not avaiable on macos")
    #SET(ENABLE_OPENMP OFF CACHE BOOL "Disable openmp support" FORCE)
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
    MESSAGE(STATUS "Enabling OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    MESSAGE(STATUS "Enabling OpenMP")
    add_compile_options(/openmp)
  endif()
endif()
  
# ----------------------------------------------------------
# DEV MODE:
# - show warnings
# - build type set to debug
#
# - highly recommanded to turn dev mode off for production
# ----------------------------------------------------------
if (ENABLE_DEV_MODE)
  MESSAGE(STATUS "Enabling Developer Mode")
  
  # Force VERBOSE Output -- used in all branches except master
  # Can be commented out in master branches.
  SET(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)

  # Export the commands for the build so they can be inspected
  SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Turn on debug
  SET(CMAKE_BUILD_TYPE Debug)

  # Force unit tests and Catch2 to compile
  SET(ENABLE_UNITTESTS ON CACHE BOOL "Enables unitests" FORCE)

  #
  # Turn on important warnings and the sanitizers
  #
  if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    # using GCC or CLANG

    # Both, GCC and Clang support the same option set
    # We first turn on Warnings
    # --> https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
    #
    # -Wall stands for All Warnings
    # -Wextra pushes that a bit further
    # -pedantic sets the compiler to use strict ISO C++
    # -Werror treats all warnings as errors so you have to fix them
    add_compile_options(-Wall -Wextra -pedantic) # -Werror)

    # Sanitizer options
    # The following will turn on Debugging (-g flag) and the Address Sanitizer
    # which is turned on with the -fsanitize=address option to the compiler.
#    add_compile_options(-g -fsanitize=address,undefined)

    # When we turn on the sanitizer, you also need to instruct the linker that it
    # will be used which happens with the following link option command.
#    add_link_options(-g -fsanitize=address,undefined)
    
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
    
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    # MSVC uses a different warning flags
#    add_compile_options(/W4 /WX)
#    add_compile_options(/fsanitize=address /MTd)
  endif()
  
ELSE (ENABLE_DEV_MODE)

  # Remove verbose output
  SET(CMAKE_VERBOSE_MAKEFILE OFF CACHE BOOL "OFF" FORCE)

  IF((CMAKE_BUILD_TYPE NOT DEFINED) OR (CMAKE_BUILD_TYPE STREQUAL ""))
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  ENDIF()
  MESSAGE(STATUS "Building the '" ${CMAKE_BUILD_TYPE} "' version of QES.")
  MESSAGE(STATUS "If you want something different, use -DCMAKE_BUILD_TYPE=Debug, Release, RelWithDebInfo, MinSizeRel.")

ENDIF (ENABLE_DEV_MODE)  

# ----------------------------------------------------------
# Boost
#
# This section tests for Boost support on the system. Boost is
# required as it is used heavily with this code. Boost also allows the
# code to be nicely portable across Operating Systems.
# ----------------------------------------------------------
# SET (Boost_MULTITHREADED TRUE)

#set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
#set(Boost_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
#set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
#set(Boost_USE_MULTITHREADED      ON)
#set(Boost_USE_STATIC_RUNTIME     ON)  # link Boost Static libraries

if(DEFINED ENV{BOOST_DIR})
  message(STATUS "Setting BOOST_DIR to $ENV{BOOST_DIR}")
  SET(BOOST_ROOT $ENV{BOOST_DIR})
endif()

FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options date_time) # property_tree)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
  MESSAGE(STATUS "Found Boost Libraries in ${Boost_LIBRARIES}, Version ${Boost_VERSION}")
ENDIF()

# ----------------------------------------------------------
# NetCDF
# ----------------------------------------------------------
if (WIN32 OR CMAKE_TOOLCHAIN_FILE) # MATCHES "vcpkg.cmake$"))

  # clear the module path as the findnetcdf.cmake that we need on
  # unix at the moment causes major issues finding netcdf here.
  set(CMAKE_MODULE_PATH "")

  SET(NETCDF_CXX "YES")
  FIND_PACKAGE(netCDFCxx CONFIG REQUIRED)
  IF(netCDFCxx_FOUND)
    MESSAGE(STATUS "Found NetCDF CXX: ${NETCDF_C_INCLUDE_DIR} ${NETCDF_C_LIBRARY}")
    INCLUDE_DIRECTORIES(${NETCDF_C_INCLUDE_DIR})
  ENDIF(netCDFCxx_FOUND)

  FIND_PACKAGE(netCDF CONFIG REQUIRED)
  IF(netCDF_FOUND)
    MESSAGE(STATUS "Found NetCDF - Version: ${NetCDFVersion}, ${netCDF_INCLUDE_DIR}, ${netCDF_LIB_DIR}, ${netCDF_LIBRARIES}, ${netCDF_LDFLAGS}, ${netCDF_C_CPP_FLAGS}")
    #  ADD_LINK_OPTIONS("${netCDF_LDFLAGS}")
    INCLUDE_DIRECTORIES(${netCDF_INCLUDE_DIR})
    LINK_DIRECTORIES(${netCDF_LIB_DIR})
  ENDIF(netCDF_FOUND)

  # reset the module path...
  LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/Modules/")
  MESSAGE(STATUS "CMake Module Path: ${CMAKE_MODULE_PATH}")
else()
  # What we had before - but doesn't work so well with vcpkg or Windows
  # NetCDF is odd in how it gets installed...
  SET(NETCDF_CXX "YES")
  FIND_PACKAGE(NetCDF REQUIRED)
  IF(NetCDF_FOUND)
    MESSAGE(STATUS "Found NetCDF: ${NETCDF_INCLUDES}: ${NETCDF_LIBRARIES}, ${NETCDF_INCLUDES_CXX}, ${NETCDF_LIBRARIES_CXX}")
    INCLUDE_DIRECTORIES(${NETCDF_INCLUDES})
    INCLUDE_DIRECTORIES(${NETCDF_INCLUDES_CXX})
  ENDIF(NetCDF_FOUND)
endif()

# ----------------------------------------------------------
# GDAL
# ----------------------------------------------------------
# SET(CMAKE_PREFIX_PATH /uufs/chpc.utah.edu/sys/installdir/gdal/2.4.0)
FIND_PACKAGE(GDAL REQUIRED)
IF (${GDAL_FOUND})
  MESSAGE(STATUS "Found GDAL: ${GDAL_INCLUDE_DIR}")
  INCLUDE_DIRECTORIES(${GDAL_INCLUDE_DIR})
ENDIF(${GDAL_FOUND})

# ----------------------------------------------------------
# CUDA
# - check if CUDA is available
# ----------------------------------------------------------
INCLUDE(CheckLanguage)
CHECK_LANGUAGE(CUDA)

FIND_PACKAGE(CUDAToolkit)

If (CMAKE_CUDA_COMPILER)

   MESSAGE(STATUS "CUDA is available; CUDA_COMPILER=${CMAKE_CUDA_COMPILER}, Version=${CMAKE_CUDA_COMPILER_VERSION}")
   ENABLE_LANGUAGE( CUDA )

   IF(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
   ENDIF()
   
ELSE()
  MESSAGE(STATUS "CUDA is not available.")
ENDIF()

# SET(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-rdc=true;-shared;-Xcompiler;-fPIC" )

#use fast math option is required for OptiX
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_61;-dlink;-rdc=true;--use_fast_math;--keep-device-functions")

# Attempt to find CUDA
IF(CMAKE_CUDA_COMPILER)
  MESSAGE(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
  SET(HAS_CUDA_SUPPORT ON CACHE BOOL "Determines if CUDA/GPU functionality is compiled into the code base" )

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  
  INCLUDE_DIRECTORIES(${CUDAToolkit_INCLUDE_DIRS})
  #INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/samples/common/inc)
  LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib64)


ELSE(CUDA_FOUND)
  #
  # CUDA was not found, so disable CUDA/GPU support
  #
  SET(HAS_CUDA_SUPPORT OFF CACHE BOOL "Determines if CUDA/GPU functionality is compiled into the code base" FORCE)
  SET(HAS_OPTIX_SUPPORT OFF CACHE BOOL "Determines if OptiX functionality is compiled into the code base" FORCE)
  
  MESSAGE(WARNING "CUDA is required for compiling GPU accelerated computations into this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories if you need these features.")
  MESSAGE(WARNING "GPU and related accelerations will be disable for this build.")
ENDIF()

# ----------------------------------------------------------
# CUDA and OPTIX
# if CUDA 10.2 or higher is found, then OptiX support can be enabled, so search for it.
# ----------------------------------------------------------
if(HAS_CUDA_SUPPORT)

  ADD_DEFINITIONS(-DHAS_CUDA)

  # May need this only on Windows to get around an odd error from
  # Boost where __builtin_FUNCSIG is undefined
  ADD_DEFINITIONS(-DBOOST_DISABLE_CURRENT_LOCATION)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
   endif()
   
  MESSAGE(STATUS "Using CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
  if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 10.2)
    MESSAGE(STATUS "Found CUDA 10.2 or higher! Now searching for OptiX support...")
    
    #
    # NVIDIA OptiX
    #   current version used in project: 7.0
    #
    FIND_PACKAGE(OptiX 7.0)
    if(NOT OptiX_INCLUDE)

      MESSAGE(STATUS "  OptiX is used for accelerating some computations with this project. If you wish to run the project with OptiX GPU acceleration, please look into installing NVIDIA CUDA 10.2 or higher and NVIDIA OptiX 7.0 or higher.")    
      SET(HAS_OPTIX_SUPPORT OFF CACHE BOOL "Determines if OptiX functionality is compiled into the code base" FORCE)
      REMOVE_DEFINITIONS(-DHAS_OPTIX)

    else(NOT OptiX_INCLUDE)
      MESSAGE(STATUS "Found NVIDIA OptiX 7.0")

      # 
      # include the OptiX macros
      #
      include(${CMAKE_SOURCE_DIR}/cmake/Optix_macros.cmake)

      # Set the preprocessor definitions to conditonally compile the optix code
      SET(HAS_OPTIX_SUPPORT ON CACHE BOOL "Determines if OptiX functionality is compiled into the code base" FORCE)
      ADD_DEFINITIONS(-DHAS_OPTIX)

      MESSAGE(STATUS "Using OptiX Include location: ${OptiX_INCLUDE}")
      include_directories(${OptiX_INCLUDE})

      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ptx") #ptx files located here
      add_definitions(-DPTX_DIR="${CMAKE_BINARY_DIR}/ptx") #PTX_DIR
      set(PTX_DIR "${CMAKE_BINARY_DIR}/ptx")
      set(CUDA_GENERATED_OUTPUT_DIR ${PTX_DIR})

      find_program(BIN2C bin2c DOC "Path to the cuda-sdk bin2c executable.")

      # the macros needed to load Optix code is in the src/winds folder CMakeLists.txt
      # as this is the only part of the code base that currently relies on OptiX.

    endif(NOT OptiX_INCLUDE)

  else()
    MESSAGE(STATUS "Did NOT find CUDA 10.2 or higher!  Disabling NVIDIA OptiX acceleration.")
    MESSAGE(STATUS "  OptiX is used for accelerating some computations with this project. If you wish to run the project with OptiX GPU acceleration, please look into installing NVIDIA CUDA 10.2 or higher and NVIDIA OptiX 7.0 or higher.")    

    SET(HAS_CUDA_SUPPORT OFF FORCE BOOL "Determines if CUDA/GPU functionality is compiled into the code base")
    REMOVE_DEFINITIONS(-DHAS_CUDA)

    SET(HAS_OPTIX_SUPPORT OFF FORCE BOOL "Determines if OptiX functionality is compiled into the code base")
    REMOVE_DEFINITIONS(-DHAS_OPTIX)
  endif()

  MESSAGE(STATUS "GPU CUDA/OptiX Configuration Complete.")

else(HAS_CUDA_SUPPORT)
  MESSAGE(STATUS "Disabling CUDA Support.")
  SET(HAS_CUDA_SUPPORT OFF FORCE BOOL "Determines if CUDA/GPU functionality is compiled into the code base")
  REMOVE_DEFINITIONS(-DHAS_CUDA)
  SET(HAS_OPTIX_SUPPORT OFF FORCE BOOL "Determines if OptiX functionality is compiled into the code base")
  REMOVE_DEFINITIONS(-DHAS_OPTIX)
endif(HAS_CUDA_SUPPORT)

# ----------------------------------------------------------
# DOXYGEN
# ----------------------------------------------------------
# Find doxygen so we can add a target for building
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
  MESSAGE(WARNING "Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)


# ----------------------------------------------------------
# LIBRARY LINK
# - define a function to link all external library needed to
#   build the code
# ----------------------------------------------------------
function(link_external_libraries exec)
  target_link_libraries(${exec} PRIVATE ${Boost_LIBRARIES})
#  target_link_libraries(${exec} PRIVATE ${Boost_DATE_TIME_LIBRARIES})
  #target_link_libraries(${exec} PRIVATE ${Boost_FILESYSTEM_LIBRARIES})
  target_link_libraries(${exec} PRIVATE ${GDAL_LIBRARY})
  # want to clean this up!!! - Pete
  if (WIN32 OR CMAKE_TOOLCHAIN_FILE) # MATCHES "vcpkg.cmake$"))
    target_link_libraries(${exec} PRIVATE "netCDF::netcdf-cxx4")
  else()
    target_link_libraries(${exec} PRIVATE ${NETCDF_LIBRARIES_CXX})
  endif()
  target_link_libraries(${exec} PRIVATE ${NETCDF_C_LIBRARY})
  IF ($CACHE{HAS_CUDA_SUPPORT})
    # target_link_libraries(${exec} cudadevrt)
    target_link_libraries(${exec} PRIVATE ${CUDA_LIBRARIES})
  ENDIF()

  IF (ENABLE_OPENMP AND ${OpenMP_FOUND})
    target_link_libraries(${exec} PRIVATE ${OpenMP_CXX_LIBRARIES})    
  ENDIF()
endfunction()

# ----------------------------------------------------------
# SOURCE CODES
# ----------------------------------------------------------
add_subdirectory(src/util)
add_subdirectory(src/winds)
add_subdirectory(src/plume)
add_subdirectory(src/fire)

# add_subdirectory(scratch)
add_subdirectory(examples)

add_subdirectory(qesWinds)
list(APPEND QES_INSTALL_EXECUTABLES qesWinds)

add_subdirectory(qesPlume)
add_subdirectory(qesFire)
add_subdirectory(qes)

# ----------------------------------------------------------
#  Testing Suite
# ----------------------------------------------------------
IF (ENABLE_TESTS)
  enable_testing() # once
  include(CTest)

  add_subdirectory(tests)

  # ----------------------------------------------------------
  # TESTING OPTION
  # ----------------------------------------------------------
  option(TESTS_ENABLE_UNIT_TESTS "Enables unit tests" OFF)
  option(TESTS_ENABLE_REGRESSION_TESTS "Enables regression tests" OFF)

  option(TESTS_ENABLE_SANITY_CHECKS_CPU "Enables tests that simply check if qesWinds runs on input w/o runtime errors" OFF)
  option(TESTS_ENABLE_SANITY_CHECKS_GPU "Enables tests that simply check if qesWinds runs on input w/o runtime errors" OFF)

  # obsolete tests.... need updated
  #option(TESTS_ENABLE_ALL_COMPARISON_TESTS "Enables test of comparison between serial and GPU solvers" OFF) # overrides individual comparison selections
  #option(TESTS_ENABLE_DYNAMIC_PARALLELISM_COMPARISON_TESTS "(NOT FINISHED) Enables tests between CPU and Dynamic Parallel solvers" OFF)
  #option(TESTS_ENABLE_GLOBAL_COMPARISON_TESTS "(NOT FINISHED) Enables tests between CPU and Global Memory solvers" OFF)
  #option(TESTS_ENABLE_SHARED_COMPARISON_TESTS "(NOT FINISHED) Enables tests between CPU and Shared Memory solvers" OFF)
  #option(TESTS_ENABLE_LONG_COMPARISON_TESTS "Enables longer comparison tests on all other comparison test options (adds around 5 minutes)" OFF)
  #option(TESTS_ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS "Enables the longest comparison tests (adds around 15 minutes)." OFF)
ENDIF()


set(CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "UtahEFD")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "QES - Quick Environmental Simulation")
set(CPACK_PACKAGE_DESCRIPTION "The Quick Environmental Simulation (QES) code is a low-computational-cost framework designed to compute high-resolution wind and concentration fields in complex atmospheric-boundary-layer environments. It contains the following modules: QES-Winds, QES-TURB, QES-Plume and QES-Fire.")
set(CPACK_PACKAGE_VERSION ${QES_VERSION_INFO})
set(CPACK_PACKAGE_CONTACT "https://qes-documentation.readthedocs.io/en/latest")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/UtahEFD")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")

install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION "${CPACK_PACKAGE_INSTALL_DIRECTORY}")
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION "${CPACK_PACKAGE_INSTALL_DIRECTORY}")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/data" DESTINATION "${CPACK_PACKAGE_INSTALL_DIRECTORY}")
install(TARGETS ${QES_INSTALL_EXECUTABLES} RUNTIME DESTINATION ${CPACK_PACKAGE_INSTALL_DIRECTORY}/bin)

if(WIN32)

  # need to include some DLLs for Windows to work correctly: gdal and boost and others
  # - this copies the DLLs from the build location into the bin of the package
  install(DIRECTORY "${CMAKE_BINARY_DIR}/qesWinds/" DESTINATION ${CPACK_PACKAGE_INSTALL_DIRECTORY}/bin FILES_MATCHING PATTERN "*.dll")
  
  set(CPACK_GENERATOR "NSIS")  # Windows Installer
  set(CPACK_NSIS_EXECUTABLES_DIRECTORY bin)
  set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
  set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
  set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
  
elseif(APPLE)
  set(CPACK_GENERATOR "DragNDrop")  # macOS DMG package
  set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_SOURCE_DIR}/data/qesBackground.png")
elseif(UNIX)
  # unix package gen has not been tested
  set(CPACK_GENERATOR "DEB;RPM")  # Linux: Deb for Debian/Ubuntu, RPM for Fedora/RHEL
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "UtahEFD <s@utah.edu>")
  # set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.28), libstdc++6 (>= 8)")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")
endif()

include( CPack )
