#cmake_minimum_required (VERSION 2.8) 
# PROJECT (cudaplume) 

SET(BUNDLEOPTS MACOSX_BUNDLE)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

SET(CMAKE_CXX_FLAGS "-Wall")

FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_FOUND) 
#  set(CUDA_VERBOSE_BUILD ON)
#      list(APPEND CUDA_NVCC_FLAGS "-g -G")
#  SET(CUDA_NVCC_FLAGS "-arch;sm_11")

  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
#ignore this################################## 
#SET(SDK_LIB "/home/alex/NVIDIA_GPU_Computing_SDK") 
#INCLUDE_DIRECTORIES (${SDK_PATH})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/samples/common/inc)
  LINK_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/lib/)

  # Add the need flags for atomic operations in CUDA.
   #SET(CUDA_NVCC_FLAGS -arch=compute_13)
   #SET(CUDA_NVCC_FLAGS "-g -G")

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

INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  
LINK_DIRECTORIES (${CMAKE_BINARY_DIR}/quicutil/)

cuda_add_executable (plume 
   shader/particle/shaders.cpp kernel/kernel_interface.cu 
   shader/particle/render_particles.cpp  
   plumeSystem.cpp  
  
  bw/Eulerian.cpp
  bw/Util.cpp
  bw/advectPar.cpp
  bw/Random.cpp
  bw/Dispersion.cpp
  bw/Turbulence.cpp
  bw/LocalMixing.cpp
  bw/NonLocalMixing.cpp
  bw/NonLocalMixing2.cpp
  main.cpp
)

target_link_libraries (plume quicloader) 
target_link_libraries (plume util)
target_link_libraries (plume ${Boost_PROGRAM_OPTIONS_LIBRARIES} )  

# IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
#  target_link_libraries (plume cutil_x86_64)
#ELSE(CMAKE_SIZEOF_VOID_P EQUAL 8)
#  target_link_libraries (plume cutil_i386)
#ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)

if (APPLE)
  find_package(OpenGL)
  INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
  find_package(GLUT)
  INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR})
  target_link_libraries (plume ${OPENGL_LIBRARIES})
  target_link_libraries (plume ${GLUT_LIBRARIES})
else (APPLE)
  target_link_libraries (plume GL)
  target_link_libraries (plume glut)
  target_link_libraries (plume GLU)
endif (APPLE)
target_link_libraries (plume GLEW) 
