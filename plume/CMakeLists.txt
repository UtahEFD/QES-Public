cmake_minimum_required (VERSION 2.8) 
PROJECT (plume) 

SET(BUNDLEOPTS MACOSX_BUNDLE)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

SET(SDK_PATH "/home/alex/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/")
INCLUDE_DIRECTORIES (${SDK_PATH})

SET(SDK_LIB "/home/alex/NVIDIA_GPU_Computing_SDK/C/lib/")
LINK_DIRECTORIES (${SDK_LIB}) 

SET(LIBSIVELAB_PATH "${CMAKE_SOURCE_DIR}/../libsivelab" CACHE STRING "Path to the directory where the libsivelab repository is located (assumes code is built into a directory named 'build').")
INCLUDE_DIRECTORIES (${LIBSIVELAB_PATH}) 
LINK_DIRECTORIES (${LIBSIVELAB_PATH}/lib) 

FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)

  # Add the need flags for atomic operations in CUDA.
  # SET(CUDA_NVCC_FLAGS -arch=compute_13)

#  ADD_SUBDIRECTORY(utils/computeVersion)

#  ADD_CUSTOM_COMMAND(
#    COMMAND utils/computeVersion 
#    DEPENDS computeVersion
#    )
# ADD_CUSTOM_TARGET(computeVersion ALL)

#  EXEC_PROGRAM(utils/computeVersion/computeVersion
#    OUTPUT_VARIABLE COMPUTE_VERSION)
#  SET(COMPUTE_VERSION "${COMPUTE_VERSION}" CACHE INTERNAL "")
#  SET(CUDA_NVCC_FLAGS -arch=compute_${COMPUTE_VERSION})
#  MESSAGE(STATUS "Setting CUDA Compute Version to compute_${COMPUTE_VERSION}")
ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)

SET(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE(Boost REQUIRED)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ENDIF()

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
IF(${Boost_PROGRAM_OPTIONS_FOUND})
ENDIF()

cuda_add_executable (plume 
  Shader/Particle/shaders.cpp Kernel/particleSystem.cu Shader/Particle/render_particles.cpp 
  particleSystem.cpp main.cpp
)

target_link_libraries (plume sive-quicutil) 
target_link_libraries (plume sive-util)
target_link_libraries (plume cutil_i386)
target_link_libraries (plume GLEW) 
target_link_libraries (plume cuda) 
target_link_libraries (plume GL) 
target_link_libraries (plume glut) 
target_link_libraries (plume GLU) 
target_link_libraries (plume ${Boost_PROGRAM_OPTIONS_LIBRARIES} )
