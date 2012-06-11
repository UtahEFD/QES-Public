cmake_minimum_required (VERSION 2.8)
PROJECT (SIVELib)

# OPTION (BUILD_SHARED_LIBS "Build Shared Libraries (default is OFF)" OFF)
# Set the library type to static
SET (LIB_TYPE STATIC)

# Ensure the CMAKE_LIBRARY_OUTPUT_DIRECTORY is set
IF(WIN32)
  SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
ELSE(WIN32)
  SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
ENDIF(WIN32)

INCLUDE(ExternalProject)

FIND_PACKAGE(PNG REQUIRED)
IF (PNG_FOUND )
  MESSAGE(${PNG_INCLUDE_DIR})
  MESSAGE(${PNG_LIBRARIES})
  INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIR})
ELSE (PNG_FOUND)
  IF( CMAKE_MAJOR_VERSION GREATER 1  AND CMAKE_MINOR_VERSION GREATER 7  )  # Need 2.8 or higher
    ExternalProject_Add(
      zlib-1.2.5
      URL ${CMAKE_CURRENT_SOURCE_DIR}/Externals/zlib-1.2.5.tar.gz
      URL_MD5 95cd21d8abbb37b7616432fc8bc58e6c
      CMAKE_COMMAND cmake
      CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
      INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
      UPDATE_COMMAND ""
      )

    ExternalProject_Add(
      libpng-1.5.7
      URL ${CMAKE_CURRENT_SOURCE_DIR}/Externals/libpng-1.5.7.tar.gz
      URL_MD5 944b56a84b65d94054cc73d7ff965de8
      CMAKE_COMMAND cmake
      CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=Release -DPNG_SHARED:BOOL=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
      INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
      UPDATE_COMMAND ""
      )
  ELSE()
    MESSAGE(FATAL ERROR "PNG could not be located and is required for built-in PNG support.\n"
      "You have two options: (1) Upgrade your version of CMake to something greater than 2.8.5, or\n"
      "(2) install libpng from http://www.libpng.org.  Either should remedy this issue for you.")
  ENDIF()

  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/include)
  SET(PNG_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib)
ENDIF (PNG_FOUND)

SET(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE(Boost REQUIRED)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ELSE()
  MESSAGE(FATAL ERROR "Boost could not be located and is required for libsivelab.")
ENDIF()

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
IF(${Boost_PROGRAM_OPTIONS_FOUND})
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

# Include stuff in the common library of newurbCUDA.
FIND_PACKAGE(CUDA REQUIRED)
IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)

  # Add the need flags for atomic operations in CUDA.
  SET(CUDA_NVCC_FLAGS -arch sm_11)
ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)

add_subdirectory(cuda)
add_subdirectory(util)
add_subdirectory(network)
add_subdirectory(quicutil)

add_subdirectory(tests)
add_subdirectory(tools)

ENABLE_TESTING()
INCLUDE(CTest)
ADD_TEST(vectorTest ${CMAKE_BINARY_DIR}/tests/util_Vector3D)

# INSTALL(FILES ${CMAKE_SOURCE_DIR}/util/*.h DESTINATION include/util)
# Another example
# INSTALL(FILES file5.txt ${CMAKE_CURRENT_BINARY_DIR}/file6.cmake
# DESTINATION share/special_files)

# INSTALL(TARGETS SIVELAB
#   BUNDLE DESTINATION . COMPONENT libsivelab
#   RUNTIME DESTINATION bin COMPONENT libsivelab
# )

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "SIVELib Code Base - Utility libraries for SIVELab projects.")
SET(CPACK_PACKAGE_VENDOR "UMDSIVELab")

SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
# SET(CPACK_PACKAGE_VERSION_PATCH "2")

SET(CPACK_SOURCE_PACKAGE_FILE_NAME "sivelab-1.0.0")
SET(CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}")
SET(CPACK_PACKAGE_CONTACT "envsim@d.umn.edu")

# SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
# SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt")

# SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")

IF(LINUX)
  SET(CPACK_GENERATOR "DEB")
  SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Pete Willemsen")
ENDIF(LINUX)

IF (APPLE)
ENDIF(APPLE)

INCLUDE(InstallRequiredSystemLibraries)
INCLUDE(CPack)

# To use this:
# make package
# sudo dpkg -i DistanceBetweenPoints-0.1.1-Linux.deb
 
# This will result in the file:
#/usr/distance/DistanceBetweenPoints
