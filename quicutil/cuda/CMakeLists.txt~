CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/include)
ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)

CUDA_ADD_LIBRARY(sive-cudautil
#  lcgStandAlone.cu lcgKernel.h
	pack.cu
	showerror.cu
)

CUDA_ADD_EXECUTABLE(sum
  sum.cu
)

CUDA_ADD_EXECUTABLE(computeVersion computeVersion.cpp)

# Where is this?
#CUDA_ADD_EXECUTABLE(lcgTest
#  lcgTest.cu
#)
#TARGET_LINK_LIBRARIES(lcgTest cudautil)

INSTALL(TARGETS computeVersion 
  RUNTIME
  DESTINATION bin)
