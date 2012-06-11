# Install script for directory: /home/alex/Desktop/project/libsivelab/util

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/util" TYPE FILE FILES
    "/home/alex/Desktop/project/libsivelab/util/GLSL.h"
    "/home/alex/Desktop/project/libsivelab/util/Random.h"
    "/home/alex/Desktop/project/libsivelab/util/handlePlumeArgs.h"
    "/home/alex/Desktop/project/libsivelab/util/handleNetworkArgs.h"
    "/home/alex/Desktop/project/libsivelab/util/logstream.h"
    "/home/alex/Desktop/project/libsivelab/util/Vector3D.h"
    "/home/alex/Desktop/project/libsivelab/util/normal.h"
    "/home/alex/Desktop/project/libsivelab/util/handleGraphicsArgs.h"
    "/home/alex/Desktop/project/libsivelab/util/handleQUICArgs.h"
    "/home/alex/Desktop/project/libsivelab/util/fileHandling.h"
    "/home/alex/Desktop/project/libsivelab/util/ArgumentParsing.h"
    "/home/alex/Desktop/project/libsivelab/util/point.h"
    "/home/alex/Desktop/project/libsivelab/util/angle.h"
    "/home/alex/Desktop/project/libsivelab/util/Timer.h"
    "/home/alex/Desktop/project/libsivelab/util/StopWatch.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/alex/Desktop/project/libsivelab/lib/libsive-util.a")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

