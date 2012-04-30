CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
PROJECT (Common) 

SET(Common_MAJOR_VERSION 0)
SET(Common_MINOR_VERSION 0)
SET(Common_PATCH_VERSION 0)
SET(Common_SOVERSION 0)
SET(Common_VERSION ${Common_MAJOR_VERSION}.${Common_MINOR_VERSION}.${Common_PATCH_VERSION})

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../CMake/Modules/")

# OPTION (BUILD_SHARED_LIBS "Build Shared Libraries (default is OFF)" OFF)
# Set the library type to static
SET (LIB_TYPE STATIC)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

#Determine if we're building for visuals or not (GLUT, GLEW, OpenGL...)
# How to set the option from the terminal:
# cmake -D QUIC_BUILD_DEV=[On \ Off] .. 
#
# You can set the option manually here:
# SET(QUIC_BUILD_DEV On)
# 
# If not set or passed through -D then it's default value is 'Off'.
OPTION(QUIC_BUILD_DEV "On - buildings everything; Off - builds console only" On)

#SET(LIBSIVELAB_PATH "${CMAKE_SOURCE_DIR}/../libsivelab" CACHE STRING "Path to the directory where the libsivelab repository is located.")
#SET(OPTIX_PATH /Developer/OptiX CACHE STRING "Path the to directory where OptiX is installed (defaults to Apple install location).")

FIND_PACKAGE(CUDA REQUIRED)
IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)

  # Add the need flags for atomic operations in CUDA.
  SET(CUDA_NVCC_FLAGS -arch sm_11)

ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)


# If dev, then build all.
# if not, then only build the basic console executables.
if(QUIC_BUILD_DEV)
  # Find OpenGL, GLUT, CUDA, and other related packages
  find_package(OpenGL REQUIRED)
  find_package(GLUT REQUIRED)
  find_package(GLEW REQUIRED)

  include_directories(${OPENGL_INCLUDE_DIR})
  include_directories(${GLUT_INCLUDE_DIRS})
  include_directories(${GLEW_INCLUDE_PATH})

  set(LIBS ${LIBS} ${OPENGL_LIBRARIES})
  set(LIBS ${LIBS} ${GLUT_LIBRARIES})
  set(LIBS ${LIBS} ${GLEW_LIBRARY})
  
  link_libraries (${OPENGL_LIBRARY})
  link_libraries (${GLUT_LIBRARY})
  link_libraries (${GLEW_LIBRARY})
endif()

#INCLUDE_DIRECTORIES (${LIBSIVELAB_PATH}) 
#LINK_DIRECTORIES (${LIBSIVELAB_PATH}/lib/) 

add_subdirectory (cuda)
add_subdirectory (quic)
add_subdirectory (tests)
add_subdirectory (tools)
add_subdirectory (util)
