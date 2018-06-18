#Set cuda version to 9.1 in ccmake set cuda host compiler to gcc-6
CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
project(cudaUrb)

FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/samples/common/inc)

ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

cuda_add_executable( cudaUrb
  main.cpp
  Interface.cu
  )
