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
  MESSAGE(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)

include_directories(${CMAKE_SOURCE_DIR}/util)
include_directories(${CMAKE_SOURCE_DIR})

include_directories("/home/cs/willemsn/Research/envsim_research/netcdf-cxx4-4.3.0.dist/include")

add_subdirectory( util )

link_directories("/home/cs/willemsn/Research/envsim_research/netcdf-cxx4-4.3.0.dist/lib")

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
  )

target_link_libraries(cudaUrb qesutil)
target_link_libraries(cudaUrb ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaUrb netcdf)
target_link_libraries(cudaUrb netcdf_c++4)
target_link_libraries(cudaUrb ${GDAL_LIBRARY})
# target_link_libraries(cudaUrb urbgpucore )
target_link_libraries(cudaUrb cudadevrt)
target_link_libraries(cudaUrb ${CUDA_LIBRARIES})

