CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
PROJECT (CUDAPlume) 

SET(gpuurbplume_MAJOR_VERSION 0)
SET(gpuurbplume_MINOR_VERSION 2)
SET(gpuurbplume_PATCH_VERSION 0)
SET(gpuurbplume_SOVERSION 0)
SET(gpuurbplume_VERSION ${gpuurbplume_MAJOR_VERSION}.${gpuurbplume_MINOR_VERSION}.${gpuurbplume_PATCH_VERSION})

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../CMake/Modules/")

FIND_PACKAGE(CUDA REQUIRED)

# Force 32-bit.  ??? not needed anymore, is it?
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
SET(CMAKE_C_FLAGS "-Wall")

IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/samples/common/inc)
ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)

IF(APPLE)
   # Currently, SNOW LEOPARD/10.6 has problems with CUDA and
   # 64-bit..., so shut off 64 bit compiling and make sure we're in
   # 32-bit ode across the board.
   # SET(CUDA_64_BIT_DEVICE_CODE ON)
#   SET(CMAKE_CXX_FLAGS "-m64")
#   SET(CMAKE_C_FLAGS "-m64")

   FIND_LIBRARY(CARBON_LIBRARY Carbon)
   FIND_LIBRARY(COCOA_LIBRARY Cocoa)
ENDIF(APPLE)


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

# present the packaging option only if we have the cpack command defined (effectively >= 2.6.0)
IF(CMAKE_CPACK_COMMAND)
    OPTION(BUILD_QUIC_PACKAGES "Set to ON to generate CPack configuration files and packaging targets" OFF)
    IF(BUILD_QUIC_PACKAGES)
      # From the OSG CPack structure...
      # resolve architecture. The reason i "change" i686 to i386 is that debian packages
      # require i386 so this is for the future
      IF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "i686")
	SET(SYSTEM_ARCH "i386")
      ELSE()
	SET(SYSTEM_ARCH ${CMAKE_SYSTEM_PROCESSOR})
      ENDIF()

      # set a default system name - use CMake setting (Linux|Windows|...)
      SET(SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
      
      # Get all defined components
      GET_CMAKE_PROPERTY(CPACK_COMPONENTS_ALL COMPONENTS)
      IF(NOT CPACK_COMPONENTS_ALL)
	# cmake 2.6.0 don't supply the COMPONENTS property.
	# I set it manually to be the packages that can always be packaged
	MESSAGE("When building packages please consider using cmake version 2.6.1 or above")
	SET(CPACK_COMPONENTS_ALL libopenscenegraph libopenthreads openscenegraph libopenscenegraph-dev libopenthreads-dev)
      ENDIF()

      INCLUDE(CPack)
    ENDIF(BUILD_QUIC_PACKAGES)
ENDIF(CMAKE_CPACK_COMMAND)


# INCLUDE_DIRECTORIES (${LIBSIVELAB_PATH}) 
# LINK_DIRECTORIES (${LIBSIVELAB_PATH}/lib/) 

include_directories( ${CMAKE_SOURCE_DIR}/quicutil )

add_subdirectory (quicutil)
add_subdirectory (plume)

# from bw's directory
ADD_EXECUTABLE(glecode
  plume/bw/main.cpp 
  plume/bw/Eulerian.cpp
  plume/bw/Util.cpp
  plume/bw/advectPar.cpp
  plume/bw/Random.cpp
  plume/bw/Dispersion.cpp
  plume/bw/Turbulence.cpp
  plume/bw/LocalMixing.cpp
  plume/bw/NonLocalMixing.cpp
  plume/bw/NonLocalMixing2.cpp
  )
target_link_libraries (glecode quicloader)
target_link_libraries (glecode util)
