# Install script for directory: /uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/util" TYPE FILE FILES
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/ArgumentParsing.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/NetCDFInput.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/NetCDFOutput.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/ParseException.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/ParseInterface.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/QESNetCDFOutput.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/Vec3D.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/Vector3.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/Vector3Double.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/Vector3Int.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/calcTime.h"
    "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/util/doesFolderExist.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/util/libqesutil.a")
endif()

