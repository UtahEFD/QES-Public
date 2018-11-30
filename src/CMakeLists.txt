CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
project(cudaUrb)

SET( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH} )

SET(Boost_USE_STATIC_LIBS TRUE)

FIND_PACKAGE(Boost REQUIRED date_time program_options system thread unit_test_framework chrono timer)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ENDIF()

FIND_PACKAGE(NetCDF)
IF(NetCDF_FOUND)
  MESSAGE(STATUS "Found NetCDF: ${NetCDF_INCLUDE_DIRECTORIES}, ${NetCDF_LIBRARIES}")
ENDIF(NetCDF_FOUND)

FIND_PACKAGE(GDAL)
IF (${GDAL_FOUND})
  INCLUDE_DIRECTORIES(${GDAL_INCLUDE_DIR})
ENDIF(${GDAL_FOUND})

SET(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
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
    add_custom_target( doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

include_directories(${CMAKE_SOURCE_DIR}/util)
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory( util )

# CUDA_ADD_LIBRARY( urbgpucore SHARED 
#   DynamicParallelism.cu
# )

CUDA_ADD_EXECUTABLE( cudaUrb
  DynamicParallelism.cu
  handleURBArgs.cpp
  Solver.cpp
  CPUSolver.cpp
  NetCDFData.cpp
  main/cudaUrbMain.cpp
  Mesh.cpp
  Triangle.cpp
  BVH.cpp
  DTEHeightField.cpp
  Cell.cpp
  Canopy.cpp
  Cut_cell.cpp
  Sensor.cpp
  )

target_link_libraries(cudaUrb qesutil)
target_link_libraries(cudaUrb ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaUrb netcdf)
target_link_libraries(cudaUrb netcdf_c++4)
target_link_libraries(cudaUrb ${GDAL_LIBRARY})
# target_link_libraries(cudaUrb urbgpucore )
target_link_libraries(cudaUrb cudadevrt)
target_link_libraries(cudaUrb ${CUDA_LIBRARIES})

