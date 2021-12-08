# Install script for directory: /uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/util/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/src/winds/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/src/plume/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/scratch/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/qesWinds/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/qesCanopy/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/qesPlume/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/qes/cmake_install.cmake")
  include("/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/tests/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/uufs/chpc.utah.edu/common/home/stoll-group3/lucasulmer/QES-Winds/buildPIV/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
