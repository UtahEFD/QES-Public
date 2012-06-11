# Install script for directory: /home/alex/Desktop/repos/libsivelab/quicutil

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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/quicutil" TYPE FILE FILES
    "/home/alex/Desktop/repos/libsivelab/quicutil/fileParser.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUICBaseProject.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/legacyFileParser.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUSimparams.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/element.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/emitterArrayElement.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUICReader.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QPBuildout.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUVelocities.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUBuildings.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/floatMatrixElement.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/bisect.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QPParams.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/datamList.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/velocities.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QPTurbulenceField.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUFileOptions.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/floatVectorElement.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/constants.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/standardFileParser.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/datam.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/attribute.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/buildingArrayElement.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUMetParams.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUScreenout.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/peekline.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/basicElements.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUICDataFile.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QPSource.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/standardElements.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/attributeList.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/velocityMatrixElement.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUICProject.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/dimensionElements.h"
    "/home/alex/Desktop/repos/libsivelab/quicutil/QUSensor.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/alex/Desktop/repos/libsivelab/lib/libsive-quicutil.a")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

