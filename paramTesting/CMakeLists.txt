CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
project(gpuParams)

SET( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH} )

SET(Boost_USE_STATIC_LIBS TRUE)

FIND_PACKAGE(Boost REQUIRED date_time program_options system thread unit_test_framework chrono timer)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ENDIF()


FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/samples/common/inc)

ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)



SET(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_61;-rdc=true;" )

include_directories(${CMAKE_SOURCE_DIR})

cuda_add_executable( gpuParams
  DataA.cpp
  DataB.cpp
  main.cpp
  kernelizer.cu
  )

target_link_libraries(gpuParams /home/csgrads/software/sivelab/cuda_9.1/lib64/libcudadevrt.a)
target_link_libraries(gpuParams ${Boost_PROGRAM_OPTIONS_LIBRARIES})
